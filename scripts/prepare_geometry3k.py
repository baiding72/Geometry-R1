#!/usr/bin/env python3
"""
Geometry-R1 Data Preparation Script

Downloads and processes the hiyouga/geometry3k dataset for the two-stage
training pipeline:
- Phase 1 (SFT): First 500 samples with pseudo-CoT from Qwen-VL API
- Phase 2 (RL): Remaining samples for GRPO training

Usage:
    # Local (mock mode for testing)
    uv run python scripts/prepare_geometry3k.py --sft-samples 500

    # With real API distillation
    export DASHSCOPE_API_KEY="your_key"
    uv run python scripts/prepare_geometry3k.py --sft-samples 500 --use-api

    # AutoDL (custom data directory)
    export DATA_DIR=/root/autodl-tmp/data
    python scripts/prepare_geometry3k.py --sft-samples 500 --use-api
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import dashscope
from dashscope import MultiModalConversation
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from src.config import (
    get_data_dir,
    get_sft_image_dir,
    get_rl_image_dir,
    setup_directories,
)


def generate_pseudo_cot_via_api(
    image_path: str,
    question: str,
    ground_truth: str,
    max_retries: int = 3,
) -> str | None:
    """
    Call Qwen-VL-Max API to generate pseudo chain-of-thought reasoning.

    This function takes an image path, question, and ground truth answer,
    and returns a formatted response with thinking process.

    Args:
        image_path: Absolute path to the image file
        question: The problem question text
        ground_truth: Ground truth answer
        max_retries: Maximum number of retry attempts

    Returns:
        Formatted string with thinking blocks and <answer>...</answer> tags,
        or None if API call fails
    """
    # 强制规范输出格式的强力 Prompt
    system_prompt = (
        "你是一个顶级的几何数学老师。你会被提供一张几何图片、一道问题，以及这道题的【最终标准答案】。\n"
        "你的任务是：根据图片和已知答案，倒推并写出极其详细、逻辑严密的解题步骤（思维链）。\n"
        "【严格格式要求】\n"
        "1. 你必须把所有的推理、计算过程包裹在 ½ 和 ½ 标签之间。\n"
        "2. 在思考过程结束后，必须另起一行，用 <answer> 和 </answer> 标签包裹那个我提供给你的最终答案。\n"
        "3. 绝对不要在标签之外输出任何多余的废话。"
    )

    user_prompt = f"题目：{question}\n已知正确答案是：{ground_truth}\n请开始你的推演："

    messages = [
        {
            "role": "system",
            "content": [{"text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},  # DashScope 支持直接传本地文件绝对路径
                {"text": user_prompt}
            ]
        }
    ]

    for attempt in range(max_retries):
        try:
            response = MultiModalConversation.call(
                model='qwen-vl-max',
                messages=messages,
                result_format='message'
            )

            if response.status_code == 200:
                # 提取大模型的回复内容
                completion = response.output.choices[0].message.content[0]['text']
                return completion
            else:
                print(f"API 调用失败，状态码: {response.status_code}, 信息: {response.message}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None

        except Exception as e:
            print(f"请求发生异常 (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None

    return None


def generate_mock_cot(question: str, ground_truth: str) -> str:
    """
    Generate mock chain-of-thought for testing without API.
    """
    clean_answer = ground_truth.strip("$").strip() if ground_truth else "N/A"

    mock_response = f"""½
Looking at this geometry problem, I need to analyze the given information and solve step by step.

Question: {question}

[This is a mock response for testing purposes.]

Based on my analysis, the answer is {clean_answer}.
½

<answer>
{clean_answer}
</answer>"""

    return mock_response


def download_geometry3k():
    """Download the geometry3k dataset from Hugging Face."""
    print("Downloading hiyouga/geometry3k dataset...")
    dataset = load_dataset("hiyouga/geometry3k", split="train")
    print(f"Dataset loaded: {len(dataset)} samples")
    return dataset


def process_sft_split(
    dataset,
    num_samples: int,
    output_dir: Path,
    image_dir: Path,
    use_api: bool = True,
) -> list[dict]:
    """
    Process SFT split: first N samples with pseudo-CoT.

    Args:
        dataset: Hugging Face dataset
        num_samples: Number of samples for SFT
        output_dir: Output directory for JSONL file
        image_dir: Directory to save images
        use_api: Whether to use real API for CoT generation

    Returns:
        List of processed samples
    """
    samples = []
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing SFT split ({num_samples} samples)...")
    if use_api:
        print("Using Qwen-VL-Max API for pseudo-CoT generation...")
    else:
        print("Using mock CoT generation (for testing)...")

    for idx in tqdm(range(min(num_samples, len(dataset)))):
        item = dataset[idx]

        # Extract fields from dataset
        images = item.get("images", [])
        question = item.get("problem", "")
        answer = item.get("answer", "")

        # Get the first image from the list
        if not images:
            print(f"Warning: No images for sample {idx}")
            continue

        image = images[0]

        # Save image
        image_filename = f"sft_{idx:05d}.png"
        image_path = image_dir / image_filename

        try:
            if isinstance(image, Image.Image):
                # Convert to RGB if necessary (handle RGBA)
                if image.mode == "RGBA":
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                elif image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(image_path, format="PNG")
            else:
                print(f"Warning: Unexpected image type for sample {idx}: {type(image)}")
                continue
        except Exception as e:
            print(f"Error saving image for sample {idx}: {e}")
            continue

        # Generate pseudo-CoT
        if use_api:
            # Get absolute path for API
            abs_image_path = str(image_path.resolve())
            completion = generate_pseudo_cot_via_api(
                image_path=abs_image_path,
                question=question,
                ground_truth=answer,
            )

            # Fallback to mock if API fails
            if completion is None:
                print(f"  API failed for sample {idx}, using mock response")
                completion = generate_mock_cot(question, answer)
        else:
            completion = generate_mock_cot(question, answer)

        # Build sample - use relative path for portability
        sample = {
            "id": f"sft_{idx:05d}",
            "image": str(image_path),  # Will be resolved at training time
            "prompt": question,
            "completion": completion,
            "ground_truth": answer.strip("$").strip() if answer else "",
            "raw_answer": answer,
        }
        samples.append(sample)

        # Rate limiting: small delay between API calls
        if use_api:
            time.sleep(0.5)

    return samples


def process_rl_split(
    dataset,
    start_idx: int,
    output_dir: Path,
    image_dir: Path,
) -> list[dict]:
    """
    Process RL split: remaining samples without completion.

    Args:
        dataset: Hugging Face dataset
        start_idx: Starting index for RL split
        output_dir: Output directory for JSONL file
        image_dir: Directory to save images

    Returns:
        List of processed samples
    """
    samples = []
    image_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(dataset) - start_idx
    print(f"\nProcessing RL split ({total_samples} samples)...")

    for idx in tqdm(range(start_idx, len(dataset))):
        item = dataset[idx]

        # Extract fields from dataset
        images = item.get("images", [])
        question = item.get("problem", "")
        answer = item.get("answer", "")

        # Get the first image from the list
        if not images:
            print(f"Warning: No images for sample {idx}")
            continue

        image = images[0]

        # Save image
        rl_idx = idx - start_idx
        image_filename = f"rl_{rl_idx:05d}.png"
        image_path = image_dir / image_filename

        try:
            if isinstance(image, Image.Image):
                # Convert to RGB if necessary (handle RGBA)
                if image.mode == "RGBA":
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                elif image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(image_path, format="PNG")
            else:
                print(f"Warning: Unexpected image type for sample {idx}: {type(image)}")
                continue
        except Exception as e:
            print(f"Error saving image for sample {idx}: {e}")
            continue

        # Build sample for GRPO - no completion, only prompt and ground truth
        sample = {
            "id": f"rl_{rl_idx:05d}",
            "image": str(image_path),
            "prompt": question,
            "ground_truth": answer.strip("$").strip() if answer else "",
            "raw_answer": answer,
        }
        samples.append(sample)

    return samples


def save_jsonl(samples: list[dict], output_path: Path) -> None:
    """Save samples to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    """Main entry point for data preparation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare geometry3k dataset for Geometry-R1 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sft-samples",
        type=int,
        default=500,
        help="Number of samples for SFT split",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (overrides DATA_DIR env var, default: data/)",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        default=False,
        help="Use Qwen-VL-Max API for pseudo-CoT generation (requires DASHSCOPE_API_KEY)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DashScope API key (or set DASHSCOPE_API_KEY environment variable)",
    )

    args = parser.parse_args()

    # Setup API key
    if args.api_key:
        dashscope.api_key = args.api_key
    elif os.environ.get("DASHSCOPE_API_KEY"):
        dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
    else:
        if args.use_api:
            print("Warning: DASHSCOPE_API_KEY not found. Falling back to mock mode.")
            args.use_api = False

    # Setup paths using config module
    if args.data_dir:
        data_dir = Path(args.data_dir)
        # Also set env var for downstream use
        os.environ["DATA_DIR"] = str(data_dir)
    else:
        data_dir = get_data_dir()

    sft_image_dir = data_dir / "images" / "sft"
    rl_image_dir = data_dir / "images" / "rl"

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    sft_image_dir.mkdir(parents=True, exist_ok=True)
    rl_image_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    dataset = download_geometry3k()

    print("=" * 60)
    print("Geometry-R1 Data Preparation")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Total dataset size: {len(dataset)}")
    print(f"SFT samples: {args.sft_samples}")
    print(f"RL samples: {len(dataset) - args.sft_samples}")
    print(f"Use API: {args.use_api}")
    print("=" * 60)

    # Process SFT split
    sft_samples = process_sft_split(
        dataset=dataset,
        num_samples=args.sft_samples,
        output_dir=data_dir,
        image_dir=sft_image_dir,
        use_api=args.use_api,
    )
    save_jsonl(sft_samples, data_dir / "sft_geometry3k.jsonl")

    # Process RL split
    rl_samples = process_rl_split(
        dataset=dataset,
        start_idx=args.sft_samples,
        output_dir=data_dir,
        image_dir=rl_image_dir,
    )
    save_jsonl(rl_samples, data_dir / "rl_geometry3k.jsonl")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - SFT data: {data_dir / 'sft_geometry3k.jsonl'}")
    print(f"  - RL data: {data_dir / 'rl_geometry3k.jsonl'}")
    print(f"  - SFT images: {sft_image_dir} ({len(sft_samples)} images)")
    print(f"  - RL images: {rl_image_dir} ({len(rl_samples)} images)")


if __name__ == "__main__":
    main()
