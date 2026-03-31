"""
Evaluate a Geometry-R1 compatible vision-language model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reward import check_mathematical_equivalence, extract_answer, extract_answer_relaxed
from src.train_utils import (
    build_generation_prompt,
    extract_assistant_text,
    load_model,
    load_processor,
    prepare_grpo_dataset,
    resolve_base_model_name,
    save_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Geometry-R1 models")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, default="data/rl_geometry3k.jsonl")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference backend. vllm requires the package to be installed and a plain base model path.",
    )
    parser.add_argument(
        "--relaxed_extraction",
        action="store_true",
        help="Use relaxed answer extraction for diagnostics when <answer> tags are missing.",
    )
    return parser.parse_args()


def batched_indices(total: int, batch_size: int) -> list[range]:
    for start in range(0, total, batch_size):
        yield range(start, min(start + batch_size, total))


def evaluate_with_transformers(args, dataset, sample_count, processor, model, device):
    total = 0
    exact_matches = 0
    symbolic_matches = 0
    format_matches = 0
    relaxed_exact_matches = 0
    relaxed_symbolic_matches = 0
    rows = []

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    for batch_range in tqdm(list(batched_indices(sample_count, args.batch_size)), desc="Evaluating"):
        batch_examples = [dataset[idx] for idx in batch_range]
        prompts = [example["prompt"] for example in batch_examples]
        images_list = [example["images"] for example in batch_examples]
        answers = [example["answer"] for example in batch_examples]
        ids = [example["id"] for example in batch_examples]

        prompt_texts = [build_generation_prompt(processor, prompt, images) for prompt, images in zip(prompts, images_list)]
        inputs = processor(text=prompt_texts, images=images_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        completions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for sample_id, answer, completion in zip(ids, answers, completions):
            predicted_answer = extract_answer(completion)
            relaxed_answer = extract_answer_relaxed(completion) if args.relaxed_extraction else predicted_answer

            total += 1
            if predicted_answer is not None:
                format_matches += 1
                if predicted_answer.strip() == answer.strip():
                    exact_matches += 1
                if check_mathematical_equivalence(predicted_answer, answer):
                    symbolic_matches += 1

            if args.relaxed_extraction and relaxed_answer is not None:
                if relaxed_answer.strip() == answer.strip():
                    relaxed_exact_matches += 1
                if check_mathematical_equivalence(relaxed_answer, answer):
                    relaxed_symbolic_matches += 1

            rows.append(
                {
                    "id": sample_id,
                    "prediction": completion,
                    "predicted_answer": predicted_answer,
                    "relaxed_predicted_answer": relaxed_answer,
                    "ground_truth": answer,
                }
            )

    return {
        "num_samples": total,
        "format_rate": format_matches / total if total else 0.0,
        "exact_match": exact_matches / total if total else 0.0,
        "symbolic_accuracy": symbolic_matches / total if total else 0.0,
        "relaxed_exact_match": relaxed_exact_matches / total if total else 0.0,
        "relaxed_symbolic_accuracy": relaxed_symbolic_matches / total if total else 0.0,
        "rows": rows,
    }


def evaluate_with_vllm(args, dataset, sample_count, processor):
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("vLLM backend requested but vllm is not installed.") from exc

    base_model_name = resolve_base_model_name(args.model_name_or_path)
    llm = LLM(model=base_model_name, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    requests = []
    metadata = []
    for idx in range(sample_count):
        example = dataset[idx]
        prompt = build_generation_prompt(processor, example["prompt"], example["images"])
        image_payload = example["images"][0] if len(example["images"]) == 1 else example["images"]
        requests.append({"prompt": prompt, "multi_modal_data": {"image": image_payload}})
        metadata.append((example["id"], example["answer"]))

    outputs = llm.generate(requests, sampling_params)

    total = 0
    exact_matches = 0
    symbolic_matches = 0
    format_matches = 0
    relaxed_exact_matches = 0
    relaxed_symbolic_matches = 0
    rows = []

    for output, (sample_id, answer) in zip(outputs, metadata):
        completion = output.outputs[0].text.strip()
        predicted_answer = extract_answer(completion)
        relaxed_answer = extract_answer_relaxed(completion) if args.relaxed_extraction else predicted_answer

        total += 1
        if predicted_answer is not None:
            format_matches += 1
            if predicted_answer.strip() == answer.strip():
                exact_matches += 1
            if check_mathematical_equivalence(predicted_answer, answer):
                symbolic_matches += 1

        if args.relaxed_extraction and relaxed_answer is not None:
            if relaxed_answer.strip() == answer.strip():
                relaxed_exact_matches += 1
            if check_mathematical_equivalence(relaxed_answer, answer):
                relaxed_symbolic_matches += 1

        rows.append(
            {
                "id": sample_id,
                "prediction": completion,
                "predicted_answer": predicted_answer,
                "relaxed_predicted_answer": relaxed_answer,
                "ground_truth": answer,
            }
        )

    return {
        "num_samples": total,
        "format_rate": format_matches / total if total else 0.0,
        "exact_match": exact_matches / total if total else 0.0,
        "symbolic_accuracy": symbolic_matches / total if total else 0.0,
        "relaxed_exact_match": relaxed_exact_matches / total if total else 0.0,
        "relaxed_symbolic_accuracy": relaxed_symbolic_matches / total if total else 0.0,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    dataset = prepare_grpo_dataset(args.data_file)
    sample_count = min(args.max_samples, len(dataset)) if args.max_samples else len(dataset)

    processor = load_processor(resolve_base_model_name(args.model_name_or_path))
    if args.backend == "vllm":
        results = evaluate_with_vllm(args, dataset, sample_count, processor)
    else:
        model = load_model(args.model_name_or_path, bf16=True, fp16=False)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        results = evaluate_with_transformers(args, dataset, sample_count, processor, model, device)

    metrics = {
        "model_name_or_path": args.model_name_or_path,
        "backend": args.backend,
        "batch_size": args.batch_size,
        "num_samples": results["num_samples"],
        "format_rate": results["format_rate"],
        "exact_match": results["exact_match"],
        "symbolic_accuracy": results["symbolic_accuracy"],
        "relaxed_exact_match": results["relaxed_exact_match"],
        "relaxed_symbolic_accuracy": results["relaxed_symbolic_accuracy"],
        "relaxed_extraction": args.relaxed_extraction,
        "samples": results["rows"],
    }

    output_file = args.output_file
    if output_file is None:
        safe_name = args.model_name_or_path.rstrip("/").split("/")[-1]
        output_file = str(Path("output") / f"eval_{safe_name}.json")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save_metrics(output_file, metrics)
    print(f"Saved evaluation report to {output_file}")
    print(
        f"format_rate={metrics['format_rate']:.4f}, "
        f"exact_match={metrics['exact_match']:.4f}, "
        f"symbolic_accuracy={metrics['symbolic_accuracy']:.4f}, "
        f"relaxed_exact_match={metrics['relaxed_exact_match']:.4f}, "
        f"relaxed_symbolic_accuracy={metrics['relaxed_symbolic_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
