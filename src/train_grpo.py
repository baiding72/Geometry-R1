"""
Run GRPO training for Geometry-R1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trl import GRPOConfig, GRPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reward import accuracy_reward, format_reward
from src.train_utils import (
    build_lora_config,
    load_processor,
    load_yaml_config,
    prepare_grpo_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO for Geometry-R1")
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    model_cfg = config["model"]
    grpo_cfg = dict(config["grpo"])
    training_cfg = dict(config["training"])
    reward_cfg = config["reward"]
    data_cfg = config["data"]

    model_name = args.model_name_or_path or model_cfg["name"]
    train_file = Path(args.train_file or data_cfg["train_file"])
    output_dir = args.output_dir or training_cfg.pop("output_dir")

    dataset = prepare_grpo_dataset(train_file)
    processor = load_processor(model_name, trust_remote_code=model_cfg.get("trust_remote_code", True))
    peft_config = build_lora_config(model_cfg)

    reward_weights = [
        reward_cfg.get("correct_answer", 1.0),
        reward_cfg.get("format_bonus", 0.1),
    ]

    training_args = GRPOConfig(
        output_dir=output_dir,
        remove_unused_columns=False,
        report_to="none",
        max_prompt_length=data_cfg.get("max_length", 2048),
        max_completion_length=grpo_cfg.pop("max_new_tokens", 512),
        beta=grpo_cfg.pop("kl_coef", 0.0),
        reward_weights=reward_weights,
        **grpo_cfg,
        **training_cfg,
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[accuracy_reward, format_reward],
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
