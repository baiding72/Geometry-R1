"""
Run SFT training for Geometry-R1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_utils import (
    build_lora_config,
    load_model,
    load_processor,
    load_yaml_config,
    prepare_sft_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT for Geometry-R1")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    model_cfg = config["model"]
    training_cfg = dict(config["training"])
    data_cfg = config["data"]

    model_name = args.model_name_or_path or model_cfg["name"]
    train_file = Path(args.train_file or data_cfg["train_file"])
    output_dir = args.output_dir or training_cfg.pop("output_dir")
    max_length = data_cfg.get("max_length", 2048)

    dataset = prepare_sft_dataset(train_file)
    processor = load_processor(model_name, trust_remote_code=model_cfg.get("trust_remote_code", True))
    model = load_model(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        bf16=bool(training_cfg.get("bf16")),
        fp16=bool(training_cfg.get("fp16")),
    )
    peft_config = build_lora_config(model_cfg)

    dataset_num_proc = data_cfg.get("preprocessing_num_workers")
    if not hasattr(dataset, "map"):
        dataset_num_proc = None

    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=max_length,
        remove_unused_columns=False,
        report_to="none",
        dataset_num_proc=dataset_num_proc,
        **training_cfg,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
