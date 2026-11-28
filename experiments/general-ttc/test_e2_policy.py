"""Regression checks for the E2 scoring and stopping policies."""
import unittest

from inference import Task, solve


class E2PolicyTest(unittest.TestCase):
    task = Task("test-question", "math", "Which answer?", ("one", "two"))

    def run_policy(self, answers, aggregation="mean", agreement=None):
        generated = []
        operations = []

        def generate(prompt, seed, maximum, temperature):
            index = len(generated)
            generated.append(seed)
            operations.append("generate")
            return f"Step: Candidate {index}\nFINAL: {answers[index]}", {}

        def score(question, steps):
            operations.append("score")
            candidate = int(steps[0].split()[-1])
            if candidate == 0:
                return [1.0, 0.4], 4
            if candidate == 1:
                return [0.65, 0.65], 4
            return [0.15, 0.15] if len(steps) == 2 else [0.15], 4

        result = solve(self.task, "E2", 10, 0, 1024, 2, generate, score,
                       prm_aggregation=aggregation, agreement_stop=agreement)
        return result, generated, operations

    def test_rank_uses_the_configured_step_aggregation(self):
        answers = ["A", "B"] * 5
        mean, mean_seeds, operations = self.run_policy(answers)
        minimum, min_seeds, _ = self.run_policy(answers, aggregation="min")
        self.assertEqual(mean["prediction"], "A")
        self.assertEqual(minimum["prediction"], "B")
        self.assertEqual(mean_seeds, min_seeds)
        self.assertEqual(operations, ["generate", "score"] * 10)
        self.assertEqual(minimum["candidates"][0]["score"], 0.4)

    def test_unanimous_valid_answer_uses_five_generations_and_no_scoring(self):
        for aggregation in ("mean", "min"):
            with self.subTest(aggregation=aggregation):
                row, seeds, operations = self.run_policy(["B"] * 5, aggregation, agreement=5)
                self.assertTrue(row["early_stopped"])
                self.assertEqual(row["prediction"], "B")
                self.assertEqual(len(seeds), 5)
                self.assertEqual(operations, ["generate"] * 5)
                self.assertEqual([e["iteration"] for e in row["events"]], list(range(5)))

    def test_disagreement_refills_to_ten_and_preserves_candidate_seeds(self):
        answers = ["A", "B"] * 5
        base, base_seeds, _ = self.run_policy(answers)
        stopped, seeds, operations = self.run_policy(answers, agreement=5)
        self.assertFalse(stopped["early_stopped"])
        self.assertEqual(base["prediction"], stopped["prediction"])
        self.assertEqual(base_seeds, seeds)
        self.assertEqual(operations, ["generate"] * 10 + ["score"] * 10)
        self.assertEqual([e["iteration"] for e in stopped["events"][10:]], list(range(10)))

    def test_missing_answers_do_not_constitute_agreement(self):
        answers = ["?"] * 5 + ["A"] * 5
        row, seeds, operations = self.run_policy(answers, agreement=5)
        self.assertFalse(row["early_stopped"])
        self.assertEqual(row["prediction"], "A")
        self.assertEqual(len(seeds), 10)
        self.assertEqual(operations.count("score"), 10)


if __name__ == "__main__":
    unittest.main()
