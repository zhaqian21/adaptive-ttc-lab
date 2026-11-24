"""Trace a shared sequence of E4 candidates with independent stopping rules."""

from dataclasses import dataclass

from revision_policy import BudgetExhausted


FIXED_COUNTS = {"k5": 5, "k10": 10}
POLICIES = (*FIXED_COUNTS, "full", "adaptive")
TOKEN_LIMIT = 125_000
MAX_ATTEMPTS = 64
SELECTORS = ("single_prm", "answer_prm_sum")


@dataclass(frozen=True)
class Prefix:
    candidates: int
    calls: int
    reason: str


class TraceBudget:
    """Record actual calls and where the original request-level cap stops."""

    def __init__(self, limit=TOKEN_LIMIT):
        self.limit = limit
        self.calls = []
        self.tokens = 0
        self.boundary = None
        self.allow_overflow = False
        self.reservation = 0

    def check(self, input_tokens, max_output_tokens):
        if self.boundary is None and self.tokens + input_tokens + max_output_tokens > self.limit:
            self.boundary = {
                "calls": len(self.calls),
                "input_tokens": input_tokens,
                "max_output_tokens": max_output_tokens,
            }
        if self.boundary is not None and not self.allow_overflow:
            raise BudgetExhausted
        self.reservation = max_output_tokens

    def record(self, action, input_tokens, output_tokens):
        if action != "sample" or not 0 <= output_tokens <= self.reservation:
            raise ValueError("Unexpected generation accounting")
        self.calls.append([input_tokens, output_tokens, self.reservation])
        self.tokens += input_tokens + output_tokens


def agreed(predictions):
    if len(predictions) < 3 or any(not isinstance(value, str) for value in predictions[:3]):
        return False
    first, second, third = (value.strip() for value in predictions[:3])
    return bool(first) and first == second == third


def collect(sample, mode, *, stop_check=None, token_limit=TOKEN_LIMIT):
    """Sample each root once. In standalone mode, return at the adaptive stop."""
    budget = TraceBudget(token_limit)
    roots, attempts, fixed = [], [], {}
    full, early = None, None

    for sequence in range(MAX_ATTEMPTS):
        budget.allow_overflow = mode == "paired" and len(roots) < max(FIXED_COUNTS.values())
        try:
            root = sample(sequence, budget)
        except BudgetExhausted:
            root = None
        if budget.boundary is not None and "attempt" not in budget.boundary:
            budget.boundary["attempt"] = sequence
            if full is None:
                full = Prefix(len(roots), budget.boundary["calls"], "token_budget")

        attempts.append({"root": len(roots) if root is not None else None, "calls": len(budget.calls)})
        if root is not None:
            roots.append(root)
            for name, count in FIXED_COUNTS.items():
                if len(roots) == count:
                    fixed[name] = Prefix(count, len(budget.calls), "fixed_count")
            if full is None and len(roots) == 3 and agreed([item.prediction for item in roots]):
                early = Prefix(3, len(budget.calls), "agreement")
            if mode == "adaptive" and full is None and early is None and stop_check is not None and stop_check(roots):
                early = Prefix(len(roots), len(budget.calls), "prm_margin")

        # Preserve the original runner's handling of three empty initial attempts.
        if full is None and sequence == 2 and not roots:
            full = Prefix(0, len(budget.calls), "no_initial_candidates")

        if mode == "adaptive" and (early is not None or full is not None):
            return roots, attempts, budget, {"adaptive": early if early is not None else full}
        if mode == "factorial" and full is not None:
            return roots, attempts, budget, {"full": full, "adaptive": early or full}
        if mode == "paired" and full is not None and len(fixed) == len(FIXED_COUNTS):
            return roots, attempts, budget, {**fixed, "full": full, "adaptive": early or full}

    raise RuntimeError(f"Sampling did not complete within {MAX_ATTEMPTS} attempts")


def costs(calls, count):
    used = calls[:count]
    return {
        "generation_calls": count,
        "generation_input_tokens": sum(entry[0] for entry in used),
        "generation_output_tokens": sum(entry[1] for entry in used),
    }


def select(candidates, prefix):
    if not prefix.candidates:
        return None
    if prefix.reason == "agreement":
        return 0
    rewards = [entry["mean_reward"] for entry in candidates[:prefix.candidates]]
    # Upstream selects the last candidate when every PRM score is zero.
    return max(range(len(rewards)), key=rewards.__getitem__) if any(rewards) else len(rewards) - 1


def select_adaptive(candidates, prefix, selector):
    original = select(candidates, prefix)
    if selector == "answer_prm_sum" and prefix.reason != "agreement" and original is not None:
        from answer_aggregation import group_answers, winner

        pool = candidates[:prefix.candidates]
        groups = group_answers([row["prediction"] for row in pool], [row["mean_reward"] for row in pool])
        return winner(groups, "sum_prm", original)
    return original
