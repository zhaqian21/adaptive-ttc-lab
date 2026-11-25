"""PRM routing and score bounds for the MATH-500 module ablations."""

from dataclasses import asdict
import math

from answer_aggregation import answer_key, group_answers, winner
from sampling import TOKEN_LIMIT, collect, costs, select


METHODS = ("factorial", "g", "f", "fg")
CHECKPOINTS = (5, 7)
MIN_MARGIN = 1.0


def bounded_winner(predictions, score):
    """Choose the same answer and representative as fully scored answer-level PRM."""
    groups = {}
    for index, raw in enumerate(predictions):
        key = answer_key(raw)
        if key is not None:
            groups.setdefault(key, []).append(index)
    best = None
    for members in sorted(groups.values(), key=len, reverse=True):
        # Every unevaluated candidate contributes at most one. A tie still
        # requires evaluation because the strongest member may break it.
        if best is not None and len(members) < best[0]:
            return -best[2]
        values = [score(index) for index in members]
        position = max(range(len(members)), key=values.__getitem__)
        entry = (math.fsum(values), values[position], -members[position])
        if best is None or entry > best:
            best = entry
    if best is not None and best[0] > 0:
        return -best[2]

    # Preserve the original PRM fallback, including any empty extracted answers.
    rewards = [score(index) for index in range(len(predictions))]
    if any(rewards):
        return max(range(len(rewards)), key=rewards.__getitem__)
    return len(rewards) - 1 if rewards else None


def run_policy(sample, verify, method, token_limit=TOKEN_LIMIT):
    rewards, checkpoints = {}, []

    def score(roots, index):
        if index not in rewards:
            value = verify(roots[index], index)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError("Expected a PRM score in [0, 1]")
            rewards[index] = value
        return rewards[index]

    def should_stop(roots):
        count = len(roots)
        if count not in CHECKPOINTS:
            return False
        answers = [root.prediction for root in roots]
        groups = group_answers(answers, [score(roots, i) for i in range(count)])
        ranked = sorted(groups, key=lambda g: (g["sum_prm"], g["best_reward"], -g["representative"]), reverse=True)
        decision = False
        if len(ranked) >= 2:
            leader, runner_up = ranked[:2]
            decision = (3 * leader["support"] >= 2 * count
                        and leader["sum_prm"] - runner_up["sum_prm"] >= MIN_MARGIN)
        checkpoints.append({"candidates": count, "stop": decision})
        return decision

    mode = "factorial" if method == "factorial" else "adaptive"
    stop_check = should_stop if method in ("f", "fg") else None
    roots, attempts, budget, prefixes = collect(sample, mode, stop_check=stop_check, token_limit=token_limit)

    def choose(prefix, rule):
        count = prefix.candidates
        predictions = [root.prediction for root in roots[:count]]
        if prefix.reason == "agreement":
            return 0
        if rule == "bounded":
            return bounded_winner(predictions, lambda i: score(roots, i))
        complete = [{"mean_reward": score(roots, i)} for i in range(count)]
        original = select(complete, prefix)
        if rule == "single":
            return original
        return winner(group_answers(predictions, [c["mean_reward"] for c in complete]), "sum_prm", original)

    arms = {}
    if method == "factorial":
        for name, prefix in prefixes.items():
            for rule in ("single", "sum"):
                selected = choose(prefix, rule)
                calls = 0 if prefix.reason == "agreement" else prefix.candidates
                arms[f"{name}_{rule}"] = {
                    **asdict(prefix), **costs(budget.calls, prefix.calls),
                    "prm_calls": calls, "selected_index": selected,
                }
    else:
        prefix = prefixes["adaptive"]
        rule = "bounded" if method in ("g", "fg") else "sum"
        selected = choose(prefix, rule)
        arms[method] = {
            **asdict(prefix), **costs(budget.calls, prefix.calls),
            "prm_calls": len(rewards), "selected_index": selected,
        }
    return roots, attempts, budget, rewards, checkpoints, arms
