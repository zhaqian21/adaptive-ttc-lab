"""Shared-budget scheduling for the three MATH-500 revision ablations."""

from collections import deque
from dataclasses import dataclass


POLICIES = ("resample", "revise-once", "fixed-point")


class BudgetExhausted(Exception):
    """The next full model request would exceed the generation budget."""


class TokenBudget:
    def __init__(self, limit, calls=()):
        self.limit = limit
        self.calls = list(calls)
        self.used = sum(call["input_tokens"] + call["output_tokens"] for call in self.calls)

    def check(self, input_tokens, max_output_tokens):
        if self.used + input_tokens + max_output_tokens > self.limit:
            raise BudgetExhausted

    def record(self, action, input_tokens, output_tokens):
        self.calls.append({"action": action, "input_tokens": input_tokens, "output_tokens": output_tokens})
        self.used += input_tokens + output_tokens
        if self.used > self.limit:
            raise ValueError("Generation exceeded the token budget")


@dataclass(frozen=True)
class Candidate:
    text: str
    token_ids: tuple
    scoring_entry: dict
    prediction: str


def schedule(policy, initial, initial_attempts, sample, revise, budget):
    """Return the last complete version of each root plus the action history."""
    if policy not in POLICIES or not initial:
        raise ValueError("Expected a supported policy and at least one initial candidate")

    roots = list(initial)
    queue = deque([None] if policy == "resample" else range(len(roots)))
    history = []
    next_sample = initial_attempts

    while queue:
        index = queue.popleft()
        action = "sample" if index is None else "revise"
        root = len(roots) if index is None else index
        original = None if index is None else roots[index]
        before = budget.used
        try:
            result = sample(next_sample, budget) if index is None else revise(original, budget)
        except BudgetExhausted:
            history.append({"action": action, "root": root, "outcome": "budget",
                            "generation_tokens": budget.used - before})
            break

        event = {"action": action, "root": root, "generation_tokens": budget.used - before,
                 "before": original.prediction if original else None,
                 "after": result.prediction if result else None}
        if index is None:
            next_sample += 1
            if result is None:
                queue.append(None)
                event["outcome"] = "invalid"
            else:
                roots.append(result)
                queue.append(None if policy == "resample" else root)
                event["outcome"] = "sampled"
        elif result is None or result.token_ids == original.token_ids:
            queue.append(None)
            event["outcome"] = "invalid" if result is None else "fixed_point"
        else:
            roots[index] = result
            queue.append(index if policy == "fixed-point" else None)
            event["outcome"] = "revised"
        history.append(event)

    return roots, history
