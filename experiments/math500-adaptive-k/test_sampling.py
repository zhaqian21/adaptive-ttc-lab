"""Regression tests for real stopping and the shared token boundary."""

from types import SimpleNamespace
import unittest

from sampling import Prefix, agreed, collect, costs, select


def run_sample(sequence, budget, answers, charge=20_000):
    budget.check(charge, 1)
    budget.record("sample", charge, 1)
    prediction = answers[sequence] if sequence < len(answers) else str(sequence)
    return SimpleNamespace(prediction=prediction)


class SamplingTest(unittest.TestCase):
    def test_prm_uses_upstream_tie_break(self):
        candidates = [{"mean_reward": value} for value in (0.0, 0.0, 0.5, 0.5)]
        self.assertEqual(select(candidates, Prefix(2, 2, "fixed_count")), 1)
        self.assertEqual(select(candidates, Prefix(4, 4, "fixed_count")), 2)

    def test_agreement_requires_three_nonempty_answers(self):
        self.assertTrue(agreed([" 42", "42", "42 "]))
        for predictions in (["42", "42"], [None, None, None], ["", " ", ""], ["42", "42", "43"]):
            self.assertFalse(agreed(predictions))

    def test_standalone_agreement_never_requests_a_fourth_candidate(self):
        def sample(sequence, budget):
            self.assertLess(sequence, 3)
            return run_sample(sequence, budget, ["42", "42", "42"])

        roots, attempts, budget, prefixes = collect(sample, "adaptive")
        self.assertEqual(len(roots), 3)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(prefixes, {"adaptive": Prefix(3, 3, "agreement")})
        self.assertEqual(costs(budget.calls, 3)["generation_input_tokens"], 60_000)

    def test_exact_fixed_k_continues_beyond_full_budget(self):
        def sample(sequence, budget):
            return run_sample(sequence, budget, ["41", "42", "42"])

        roots, _, budget, prefixes = collect(sample, "paired")
        self.assertEqual(len(roots), 10)
        self.assertEqual(budget.boundary["attempt"], 6)
        self.assertEqual(prefixes["full"], Prefix(6, 6, "token_budget"))
        self.assertEqual(prefixes["adaptive"], prefixes["full"])
        self.assertEqual(prefixes["k5"], Prefix(5, 5, "fixed_count"))
        self.assertEqual(prefixes["k10"], Prefix(10, 10, "fixed_count"))

    def test_partial_root_is_charged_without_entering_the_capped_pool(self):
        def sample(sequence, budget):
            if sequence < 5:
                return run_sample(sequence, budget, [])
            run_sample(sequence, budget, [], charge=19_000)
            return run_sample(sequence, budget, [], charge=20_000)

        roots, _, budget, prefixes = collect(sample, "adaptive")
        self.assertEqual(len(roots), 5)
        self.assertEqual(prefixes["adaptive"], Prefix(5, 6, "token_budget"))
        self.assertEqual(budget.boundary["attempt"], 5)
        self.assertEqual(costs(budget.calls, 6)["generation_input_tokens"], 119_000)

    def test_original_empty_initial_batch_does_not_start_resampling(self):
        def sample(sequence, budget):
            root = run_sample(sequence, budget, [], charge=1_000)
            return None if sequence < 3 else root

        _, _, _, prefixes = collect(sample, "paired")
        self.assertEqual(prefixes["full"], Prefix(0, 3, "no_initial_candidates"))
        self.assertEqual(prefixes["adaptive"], prefixes["full"])
        self.assertEqual(prefixes["k10"].calls, 13)


if __name__ == "__main__":
    unittest.main()
