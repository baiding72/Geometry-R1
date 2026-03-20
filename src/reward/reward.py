"""
Reward Functions for Geometry-R1 GRPO Training

This module implements reward functions based on mathematical equivalence
using SymPy for symbolic computation.
"""

import re
from typing import Optional

from sympy import simplify
from sympy.parsing.latex import parse_latex


def extract_answer(response: str) -> Optional[str]:
    """
    Extract the answer string from model response.

    Looks for <answer>...</answer> tags and extracts the content.

    Args:
        response: Model's generated response text

    Returns:
        Extracted answer string, or None if not found
    """
    # Try to find <answer>...</answer> pattern
    answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(answer_pattern, response, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return None


def normalize_latex(latex_str: str) -> str:
    """
    Normalize LaTeX string for comparison.

    Removes common formatting variations.

    Args:
        latex_str: LaTeX string to normalize

    Returns:
        Normalized string
    """
    # Remove surrounding $ symbols
    result = latex_str.strip("$").strip()

    # Remove \displaystyle if present
    result = result.replace("\\displaystyle", "")

    # Remove extra whitespace
    result = " ".join(result.split())

    # Normalize double backslashes to single (for escaped LaTeX in JSON)
    result = result.replace("\\\\", "\\")

    return result


def check_mathematical_equivalence(pred: str, gt: str) -> bool:
    """
    Check if two mathematical expressions are equivalent using SymPy.

    This function:
    1. First tries direct string comparison (after normalization)
    2. If not equal, parses both as LaTeX and checks symbolic equivalence

    Args:
        pred: Predicted answer (may be LaTeX or plain text)
        gt: Ground truth answer (may be LaTeX or plain text)

    Returns:
        True if expressions are mathematically equivalent
    """
    # Normalize both strings
    pred_normalized = normalize_latex(pred)
    gt_normalized = normalize_latex(gt)

    # Try direct string comparison first
    if pred_normalized == gt_normalized:
        return True

    # Try symbolic equivalence
    try:
        # Parse LaTeX expressions
        pred_expr = parse_latex(pred_normalized)
        gt_expr = parse_latex(gt_normalized)

        # Check if difference simplifies to zero
        diff = simplify(pred_expr - gt_expr)

        # Check if the difference is exactly zero
        if diff == 0:
            return True

        # Additional check: evaluate numerically if possible
        try:
            pred_val = float(pred_expr.evalf())
            gt_val = float(gt_expr.evalf())
            # Use relative tolerance for floating point comparison
            if abs(pred_val - gt_val) < 1e-9 * max(abs(pred_val), abs(gt_val), 1):
                return True
        except (TypeError, ValueError):
            pass

        return False

    except Exception as e:
        # If LaTeX parsing fails, fall back to string comparison
        # This handles cases where the answer is text (e.g., "A", "B", "Yes", "No")
        return pred_normalized.lower() == gt_normalized.lower()


def accuracy_reward(
    prompts: list[str],
    completions: list[str],
    answer: list[str],
    **kwargs,
) -> list[float]:
    """
    Compute accuracy reward for a batch of completions.

    This function is designed to work with TRL's GRPOTrainer interface.

    Args:
        prompts: List of input prompts (not used in reward calculation)
        completions: List of model-generated completions
        answer: List of ground truth answers
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        List of reward values (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []

    for completion, gt in zip(completions, answer):
        extracted = extract_answer(completion)

        if extracted is None:
            # No valid answer format found
            rewards.append(0.0)
            continue

        # Check mathematical equivalence
        if check_mathematical_equivalence(extracted, gt):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def compute_reward(
    response: str,
    ground_truth: str,
    format_bonus: float = 0.1,
    correct_answer_bonus: float = 1.0,
) -> float:
    """
    Compute reward for a single response.

    This is a convenience function for single-sample evaluation.

    Args:
        response: Model's generated response text
        ground_truth: Ground truth answer
        format_bonus: Bonus for having valid <answer> tags
        correct_answer_bonus: Reward for correct mathematical answer

    Returns:
        Total reward value
    """
    total_reward = 0.0

    # Check for valid format
    extracted = extract_answer(response)
    if extracted is not None:
        total_reward += format_bonus

        # Check mathematical equivalence
        if check_mathematical_equivalence(extracted, ground_truth):
            total_reward += correct_answer_bonus

    return total_reward


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Compute format reward based on presence of thinking and answer blocks.

    Args:
        completions: List of model-generated completions
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        List of format reward values
    """
    rewards = []

    for completion in completions:
        reward = 0.0

        # Check for thinking block (emoji U+1F914)
        if "\U0001F914" in completion:
            reward += 0.05

        # Check for answer block
        if extract_answer(completion) is not None:
            reward += 0.05

        rewards.append(reward)

    return rewards


if __name__ == "__main__":
    # Test the reward function
    test_cases = [
        # (response, ground_truth, expected_equivalence)
        (
            "<answer>12</answer>",
            "12",
            True,
        ),
        (
            "<answer>\\frac{12}{5}</answer>",
            "\\frac{12}{5}",
            True,
        ),
        (
            "<answer>\\frac{24}{10}</answer>",
            "\\frac{12}{5}",
            True,
        ),  # Equivalent fractions
        (
            "<answer>\\sqrt{2}</answer>",
            "$\\sqrt{2}$",
            True,
        ),
        (
            "<answer>5</answer>",
            "6",
            False,
        ),
        (
            "No answer tag here",
            "5",
            False,
        ),
    ]

    print("Testing reward functions...")
    print("=" * 60)

    for response, gt, expected in test_cases:
        extracted = extract_answer(response)
        is_equiv = check_mathematical_equivalence(extracted or "", gt)
        status = "✓" if is_equiv == expected else "✗"
        print(f"{status} Response: {response[:50]}...")
        print(f"  Ground truth: {gt}")
        print(f"  Extracted: {extracted}")
        print(f"  Equivalent: {is_equiv} (expected: {expected})")
        print()

    # Test batch accuracy_reward
    print("\nTesting accuracy_reward (batch mode)...")
    prompts = ["Q1", "Q2", "Q3"]
    completions = [
        "½ Reasoning... ½ <answer>12</answer>",
        "<answer>\\frac{12}{5}</answer>",
        "No answer here",
    ]
    answers = ["12", "\\frac{24}{10}", "5"]

    rewards = accuracy_reward(prompts, completions, answers)
    print(f"Rewards: {rewards}")
