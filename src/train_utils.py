"""
Shared training and evaluation utilities for Geometry-R1.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from PIL import Image
from peft import LoraConfig, PeftConfig, PeftModel, TaskType
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"<image>\s*", re.IGNORECASE)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def coerce_numeric_fields(config: dict[str, Any], field_names: list[str]) -> dict[str, Any]:
    """Convert selected config fields from strings to numeric types when needed."""
    normalized = dict(config)
    for field in field_names:
        value = normalized.get(field)
        if isinstance(value, str):
            try:
                if any(ch in value.lower() for ch in [".", "e"]):
                    normalized[field] = float(value)
                else:
                    normalized[field] = int(value)
            except ValueError:
                pass
    return normalized


def strip_image_placeholder(text: str) -> str:
    """Remove dataset-level image placeholders from prompts."""
    normalized = IMAGE_PLACEHOLDER_PATTERN.sub("", text).strip()
    return normalized or text.strip()


def resolve_dtype(bf16: bool = False, fp16: bool = False):
    """Resolve torch dtype lazily to avoid importing torch at module import time."""
    import torch

    if bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if fp16 and torch.cuda.is_available():
        return torch.float16
    return "auto"


def build_lora_config(model_config: dict[str, Any]) -> LoraConfig | None:
    """Construct a LoRA config if PEFT is enabled."""
    if not model_config.get("use_peft"):
        return None

    peft_kwargs = dict(model_config.get("peft_config", {}))
    return LoraConfig(task_type=TaskType.CAUSAL_LM, **peft_kwargs)


def load_processor(model_name_or_path: str, trust_remote_code: bool = True):
    """Load the processor and ensure padding is configured."""
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return processor


def is_peft_adapter_path(model_name_or_path: str | Path) -> bool:
    """Return True if the path looks like a PEFT adapter checkpoint."""
    path = Path(model_name_or_path)
    return path.exists() and (path / "adapter_config.json").exists()


def resolve_base_model_name(model_name_or_path: str | Path) -> str:
    """Resolve the underlying base model for either a full model or adapter path."""
    if is_peft_adapter_path(model_name_or_path):
        peft_config = PeftConfig.from_pretrained(str(model_name_or_path))
        return peft_config.base_model_name_or_path
    return str(model_name_or_path)


def load_model(model_name_or_path: str, trust_remote_code: bool = True, bf16: bool = False, fp16: bool = False):
    """Load the VLM with a dtype that matches the current device."""
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    model_dtype = resolve_dtype(bf16=bf16, fp16=fp16)
    if model_dtype != "auto":
        model_kwargs["dtype"] = model_dtype

    if is_peft_adapter_path(model_name_or_path):
        peft_config = PeftConfig.from_pretrained(str(model_name_or_path))
        base_model = AutoModelForImageTextToText.from_pretrained(peft_config.base_model_name_or_path, **model_kwargs)
        return PeftModel.from_pretrained(base_model, str(model_name_or_path))

    return AutoModelForImageTextToText.from_pretrained(str(model_name_or_path), **model_kwargs)


def merge_peft_adapter(
    adapter_path: str | Path,
    output_path: str | Path,
    trust_remote_code: bool = True,
    bf16: bool = False,
    fp16: bool = False,
) -> Path:
    """Merge a PEFT adapter into its base model and save a standalone checkpoint."""
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)

    if not is_peft_adapter_path(adapter_path):
        raise ValueError(f"Expected a PEFT adapter directory, got: {adapter_path}")

    model = load_model(str(adapter_path), trust_remote_code=trust_remote_code, bf16=bf16, fp16=fp16)
    processor = load_processor(str(adapter_path), trust_remote_code=trust_remote_code)

    if not isinstance(model, PeftModel):
        raise TypeError(f"Expected a PeftModel when loading adapter path {adapter_path}")

    merged_model = model.merge_and_unload()
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    return output_path


def _load_image(image_path: str | Path) -> Image.Image:
    """Load an image as RGB."""
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _make_user_message(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": strip_image_placeholder(prompt)}]


class JsonlVisionDataset(Dataset):
    """A lightweight dataset that lazily loads images from Geometry-R1 JSONL rows."""

    def __init__(self, jsonl_path: str | Path, mode: str):
        self.jsonl_path = Path(jsonl_path)
        self.mode = mode
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.rows[index]
        if self.mode == "sft":
            messages = _make_user_message(example["prompt"])
            messages.append({"role": "assistant", "content": example["completion"]})
            return {
                "messages": messages,
                "images": [_load_image(example["image"])],
                "ground_truth": example["ground_truth"],
                "id": example["id"],
            }
        if self.mode == "grpo":
            return {
                "prompt": _make_user_message(example["prompt"]),
                "images": [_load_image(example["image"])],
                "answer": example["ground_truth"],
                "ground_truth": example["ground_truth"],
                "id": example["id"],
            }
        raise ValueError(f"Unsupported dataset mode: {self.mode}")


def prepare_sft_dataset(jsonl_path: str | Path) -> Dataset:
    """Convert the project JSONL into TRL-compatible multimodal SFT rows."""
    return JsonlVisionDataset(jsonl_path, mode="sft")


def prepare_grpo_dataset(jsonl_path: str | Path) -> Dataset:
    """Convert the project JSONL into GRPO prompt rows."""
    return JsonlVisionDataset(jsonl_path, mode="grpo")


def build_generation_prompt(processor, prompt_messages: list[dict[str, Any]], images: list[Image.Image]) -> str:
    """Render a multimodal chat prompt for generation."""
    prompt_copy = copy.deepcopy(prompt_messages)
    image_included = False
    for message in prompt_copy:
        content = message.get("content", "")
        if not isinstance(content, str):
            continue

        if message["role"] == "system":
            message["content"] = [{"type": "text", "text": content}]
        elif message["role"] == "user":
            if not image_included:
                placeholders = [{"type": "image"} for _ in range(len(images))]
                message["content"] = [*placeholders, {"type": "text", "text": content}]
                image_included = True
            else:
                message["content"] = [{"type": "text", "text": content}]
        else:
            message["content"] = [{"type": "text", "text": content}]

    return processor.apply_chat_template(prompt_copy, tokenize=False, add_generation_prompt=True)


def extract_assistant_text(generated_text: str, prompt_text: str) -> str:
    """Remove the prompt prefix from decoded generations when present."""
    if generated_text.startswith(prompt_text):
        return generated_text[len(prompt_text) :].strip()
    return generated_text.strip()


def save_metrics(output_path: str | Path, payload: dict[str, Any]) -> None:
    """Persist metrics JSON with stable formatting."""
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
