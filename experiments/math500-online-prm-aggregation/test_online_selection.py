import unittest

from sampling import Prefix, select_adaptive


class OnlineSelectionTest(unittest.TestCase):
    def test_agreement_returns_before_reading_prm_scores(self):
        candidates = [{"prediction": "42"}] * 3
        prefix = Prefix(3, 4, "agreement")
        self.assertEqual(select_adaptive(candidates, prefix, "answer_prm_sum"), 0)

    def test_fallback_uses_only_the_candidates_inside_the_budget(self):
        candidates = [
            {"prediction": " 42", "mean_reward": 0.4},
            {"prediction": "42", "mean_reward": 0.45},
            {"prediction": "7", "mean_reward": 0.7},
            {"prediction": "7", "mean_reward": 1.0},
        ]
        prefix = Prefix(3, 12, "token_budget")
        self.assertEqual(select_adaptive(candidates, prefix, "single_prm"), 2)
        self.assertEqual(select_adaptive(candidates, prefix, "answer_prm_sum"), 1)

    def test_no_completed_candidate_has_no_winner(self):
        self.assertIsNone(select_adaptive([], Prefix(0, 1, "token_budget"), "answer_prm_sum"))


if __name__ == "__main__":
    unittest.main()
