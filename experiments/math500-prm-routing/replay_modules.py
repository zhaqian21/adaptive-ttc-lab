"""Replay decisions against a recorded generation and PRM trace."""

from types import SimpleNamespace

import routing
from sampling import TOKEN_LIMIT


def replay(row, method, token_limit=TOKEN_LIMIT):
    attempts, calls = row["attempts"], row["calls"]
    candidates, boundary = row["candidates"], row["budget_boundary"]

    def sample(sequence, budget):
        if sequence >= len(attempts):
            raise ValueError("Replay ran past its candidate trace")
        event = attempts[sequence]
        start = attempts[sequence - 1]["calls"] if sequence else 0
        for input_tokens, output_tokens, max_tokens in calls[start:event["calls"]]:
            budget.check(input_tokens, max_tokens)
            budget.record("sample", input_tokens, output_tokens)
        if boundary is not None and boundary["attempt"] == sequence:
            budget.check(boundary["input_tokens"], boundary["max_output_tokens"])
            raise ValueError("Original budget boundary was not reproduced")
        return SimpleNamespace(**candidates[event["root"]]) if event["root"] is not None else None

    def verify(_, index):
        value = candidates[index]["mean_reward"]
        if value is None:
            raise ValueError(f"Replay needed an unscored candidate at index {index}")
        return value

    return routing.run_policy(sample, verify, method, token_limit)


def labelled(arms, candidates):
    result = {}
    for name, arm in arms.items():
        index = arm["selected_index"]
        candidate = candidates[index] if index is not None else {"prediction": None, "correct": False}
        result[name] = {**arm, "prediction": candidate["prediction"], "correct": candidate["correct"]}
    return result
