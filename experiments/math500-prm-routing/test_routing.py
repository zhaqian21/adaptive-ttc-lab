import unittest
from types import SimpleNamespace

from routing import bounded_winner, run_policy


def sampler(answers):
    def sample(sequence, budget):
        budget.check(20, 1)
        budget.record("sample", 20, 1)
        prediction = answers[sequence] if sequence < len(answers) else str(sequence)
        return SimpleNamespace(prediction=prediction)
    return sample


class RoutingTest(unittest.TestCase):
    def test_strict_bound_preserves_tie_break(self):
        calls = []
        def score(index):
            calls.append(index)
            return [0.5, 0.5, 1.0, 0.1][index]
        self.assertEqual(bounded_winner(["same", "same", "tie", "other"], score), 2)
        self.assertEqual(calls, [0, 1, 2, 3])

    def test_group_bound_skips_losing_groups(self):
        calls = []
        def score(index):
            calls.append(index)
            return [0.95, 0.10, 0.8][index]
        self.assertEqual(bounded_winner(["same", "other", "same"], score), 0)
        self.assertEqual(calls, [0, 2])

    def test_zero_group_scores_recover_original_prm(self):
        def score(index):
            return [0.0, 0.7, 0.0][index]
        self.assertEqual(bounded_winner(["valid", None, "valid"], score), 1)

    def test_factorial_path_keeps_the_prefix_and_full_budget(self):
        sample = sampler(["same", "same", "same", "different"])
        _, _, _, rewards, _, arms = run_policy(sample, lambda root, i: 0.6, "factorial", 84)
        self.assertEqual(len(rewards), 4)
        self.assertEqual(arms["adaptive_sum"]["candidates"], 3)
        self.assertEqual(arms["adaptive_sum"]["prm_calls"], 0)
        self.assertEqual(arms["full_sum"]["candidates"], 4)

    def test_margin_stops_after_five_and_uses_each_score_once(self):
        calls = []
        def score(root, index):
            calls.append(index)
            return 0.8 if root.prediction == "same" else 0.4
        sample = sampler(["same", "same", "other", "same", "same", "extra"])
        _, _, _, _, checks, arms = run_policy(sample, score, "fg", 147)
        self.assertEqual(checks, [{"candidates": 5, "stop": True}])
        self.assertEqual(calls, list(range(5)))
        self.assertEqual(arms["fg"]["reason"], "prm_margin")
        self.assertEqual(arms["fg"]["generation_calls"], 5)


if __name__ == "__main__":
    unittest.main()
