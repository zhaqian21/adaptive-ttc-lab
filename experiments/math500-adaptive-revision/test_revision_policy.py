import unittest

from revision_policy import BudgetExhausted, Candidate, TokenBudget, schedule


def candidate(value):
    return Candidate(str(value), (value,), {}, str(value))


class SchedulingTest(unittest.TestCase):
    def test_fixed_point_requeues_changed_revision_and_replaces_unchanged_root(self):
        revised = iter((candidate(2), candidate(2)))
        budget = TokenBudget(3)

        def revise(original, budget):
            budget.check(0, 1)
            budget.record("revise", 0, 1)
            return next(revised)

        def sample(sequence, budget):
            budget.check(0, 1)
            budget.record("sample", 0, 1)
            return candidate(5 + sequence)

        roots, history = schedule("fixed-point", [candidate(1)], 1, sample, revise, budget)
        self.assertEqual([row["outcome"] for row in history], ["revised", "fixed_point", "sampled", "budget"])
        self.assertEqual([root.prediction for root in roots], ["2", "6"])

    def test_revise_once_moves_to_resampling_even_after_change(self):
        budget = TokenBudget(2)

        def revise(original, budget):
            budget.check(0, 1)
            budget.record("revise", 0, 1)
            return candidate(2)

        def sample(sequence, budget):
            budget.check(0, 1)
            budget.record("sample", 0, 1)
            return candidate(sequence + 2)

        _, history = schedule("revise-once", [candidate(1)], 1, sample, revise, budget)
        self.assertEqual([row["action"] for row in history], ["revise", "sample", "revise"])
        self.assertEqual(history[-1]["outcome"], "budget")

    def test_invalid_sample_gets_a_new_seed_and_partial_samples_are_charged(self):
        seen = []
        budget = TokenBudget(5)

        def sample(sequence, budget):
            seen.append(sequence)
            budget.check(2, 1)
            budget.record("sample", 2, 1)
            if sequence == 1:
                return None
            self.fail("The second sample should not fit")

        def revise(*args):
            self.fail("The resampling policy should not revise")

        roots, history = schedule("resample", [candidate(1)], 1, sample, revise, budget)
        self.assertEqual(seen, [1, 2])
        self.assertEqual([row["outcome"] for row in history], ["invalid", "budget"])
        self.assertEqual(len(roots), 1)
        self.assertEqual(budget.used, 3)

    def test_fork_charges_the_same_initial_work_for_each_policy(self):
        original = TokenBudget(10)
        original.check(4, 4)
        original.record("sample", 4, 2)
        fork = TokenBudget(original.limit, original.calls)
        with self.assertRaises(BudgetExhausted):
            fork.check(3, 2)
        fork.record("revise", 2, 2)
        self.assertEqual((original.used, fork.used), (6, 10))


if __name__ == "__main__":
    unittest.main()
