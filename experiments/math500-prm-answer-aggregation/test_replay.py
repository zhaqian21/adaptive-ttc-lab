import unittest

from replay import group_answers, replay_row, winner


def candidate(answer, reward, correct=False):
    return {"prediction": answer, "mean_reward": reward, "correct": correct}


def row(candidates, selected, count, reason="token_budget"):
    answer = candidates[selected] if selected is not None else {"prediction": None, "correct": False}
    return {
        "problem_id": 0,
        "dataset_unique_id": "fixture",
        "candidates": candidates,
        "policies": {"adaptive": {
            "candidates": count, "reason": reason, "selected_index": selected,
            "prediction": answer["prediction"], "correct": answer["correct"],
            "prm_calls": 0 if reason == "agreement" else count,
            "generation_input_tokens": 100, "generation_output_tokens": 20, "generation_calls": 4,
        }},
    }


class ReplayTest(unittest.TestCase):
    def test_weighted_votes_can_override_single_highest_score(self):
        groups = group_answers([" a ", "a", "b", None, "  "], [0.4, 0.45, 0.7, 0.9, 0.9])
        self.assertEqual([group["members"] for group in groups], [[0, 1], [2]])
        self.assertEqual(winner(groups, "sum_prm", 2), 1)

    def test_ties_use_highest_member_then_first_candidate(self):
        groups = group_answers(["a", "b", "a", "c"], [0.25, 0.5, 0.25, 0.5])
        self.assertEqual(winner(groups, "sum_prm", 1), 1)
        self.assertEqual(winner(groups, "support", 1), 0)

    def test_replay_cannot_see_candidates_beyond_the_adaptive_prefix(self):
        candidates = [candidate("a", 0.4, True), candidate("a", 0.4, True), candidate("b", 0.7),
                      candidate("b", 1.0), candidate("b", 1.0)]
        result = replay_row(row(candidates, selected=2, count=3))
        self.assertEqual(result["policies"]["sum_prm"]["selected_index"], 0)
        self.assertEqual(result["cost"]["prm_calls"], 3)
        self.assertTrue(result["policies"]["sum_prm"]["correct"])

    def test_early_stop_bypasses_aggregation_without_scores(self):
        candidates = [candidate("42", None, True), candidate(" 42", None, True),
                      candidate("42 ", None, True), candidate("7", 1.0)]
        result = replay_row(row(candidates, selected=0, count=3, reason="agreement"))
        self.assertEqual(result["answer_groups"], [])
        self.assertEqual(result["cost"]["prm_calls"], 0)
        self.assertTrue(all(choice == result["policies"]["adaptive"] for choice in result["policies"].values()))

    def test_zero_scores_preserve_weighted_baseline_and_empty_pool_abstains(self):
        candidates = [candidate("a", 0), candidate("b", 0), candidate("b", 0)]
        weighted = replay_row(row(candidates, selected=2, count=3))["policies"]["sum_prm"]
        self.assertEqual(weighted["selected_index"], 2)
        empty = replay_row(row([], selected=None, count=0, reason="no_initial_candidates"))
        self.assertTrue(all(choice["selected_index"] is None for choice in empty["policies"].values()))


if __name__ == "__main__":
    unittest.main()
