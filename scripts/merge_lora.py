#!/usr/bin/env python3
"""
Merge a LoRA adapter checkpoint into its base model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_utils import merge_peft_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into a standalone model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the LoRA adapter checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged standalone model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_path = merge_peft_adapter(args.adapter_path, args.output_path, bf16=True, fp16=False)
    print(f"Merged model saved to: {merged_path}")


if __name__ == "__main__":
    main()
