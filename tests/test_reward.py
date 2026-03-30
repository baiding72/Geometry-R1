import unittest

from src.reward.reward import accuracy_reward, check_mathematical_equivalence, extract_answer, format_reward


class RewardTestCase(unittest.TestCase):
    def test_extract_answer_from_tags(self):
        self.assertEqual(extract_answer("<answer>42</answer>"), "42")

    def test_symbolic_equivalence_fraction(self):
        self.assertTrue(check_mathematical_equivalence(r"\frac{24}{10}", r"\frac{12}{5}"))

    def test_accuracy_reward_accepts_ground_truth_alias(self):
        rewards = accuracy_reward(
            prompts=["q"],
            completions=["<answer>5</answer>"],
            ground_truth=["5"],
        )
        self.assertEqual(rewards, [1.0])

    def test_format_reward_uses_reasoning_and_answer_tags(self):
        rewards = format_reward(["½ reasoning ½\n<answer>5</answer>"])
        self.assertEqual(rewards, [0.1])


if __name__ == "__main__":
    unittest.main()
