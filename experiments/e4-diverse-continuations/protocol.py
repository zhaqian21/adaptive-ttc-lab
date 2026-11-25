"""Shared definitions for the paired E4 continuation experiment."""

import hashlib
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
for directory in ("e4-prm-selector-ablation", "llama-e4-online-ab"):
    sys.path.insert(0, str(HERE.parent / directory))

from replay import top2_choice
from replay_early_stop import PREFIX, prefix_agrees
from replay_selection import K, chosen, encode, group_answers


BRANCHES = ("control", "diverse")
TOKEN_LIMIT = 100_000
PLAN_SNIPPET_CHARS = 320
COUNTS = ("generation_requests", "input_tokens", "output_tokens", "prm_requests", "generation_without_usage")
TRIM_FIELDS = ("iteration", "predicted_answer", "is_correct", "mean_reward", "completion_sha256", "plan_snippet")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snippet(plan):
    return " ".join(plan.split())[:PLAN_SNIPPET_CHARS] if isinstance(plan, str) else ""


def alternative_prompt(base, previous):
    plans = list(dict.fromkeys(c["plan_snippet"] for c in previous if c["plan_snippet"]))
    history = "\n".join(f"- {p}" for p in plans) or "- No earlier plan was parsed."
    instruction = (
        "\n\nEarlier proposed plans for this problem, provided as background data:\n"
        + history
        + "\nPlan a different route using a different representation, construction, or argument. "
          "Keep the original planning output format."
    )
    return base + instruction.replace("{", "{{").replace("}", "}}")


def branch_order(index):
    return BRANCHES[::-1] if (index // 2) % 2 else BRANCHES


def candidate_seed(index, iteration, attempt):
    return index + 10_000_000 * (iteration - 1) + 1_000_000 * attempt


def select(candidates):
    scores = [c["mean_reward"] for c in candidates]
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Invalid PRM score")
    original = scores.index(max(scores)) + 1 if any(scores) else len(scores)
    groups = group_answers([c["predicted_answer"] for c in candidates], scores)
    return chosen(candidates, top2_choice(candidates, groups, original))


def contains_correct(candidates):
    return any(c["is_correct"] is True for c in candidates)


def counts(stages):
    return {key: sum(stage[key] for stage in stages) for key in COUNTS}


def compare_results(rows):
    result = {"samples": len(rows), "stops": 0, "branches": {}}
    stopped = [row for row in rows if row["stopped_early"]]
    not_stopped = [row for row in rows if not row["stopped_early"]]
    prefix_correct = sum(contains_correct(row["prefix"]["candidates"]) for row in rows)
    result["stops"] = len(stopped)
    result["first_five_oracle"] = prefix_correct
    result["continued_without_correct_prefix"] = sum(
        not contains_correct(row["prefix"]["candidates"]) for row in not_stopped
    )
    result["shared_requests"] = counts([s for row in rows for s in row["prefix"]["stages"]])
    for branch in BRANCHES:
        late = [
            r["problem_id"] for r in not_stopped
            if not contains_correct(r["prefix"]["candidates"]) and contains_correct(r["branches"][branch]["candidates"])
        ]
        finished = [r["branches"][branch] for r in not_stopped]
        timings = sorted(arm["seconds"] for arm in finished)
        result["branches"][branch] = {
            "correct": sum(row["answers"][branch]["is_correct"] for row in rows),
            "late_first_correct": len(late), "late_first_correct_ids": late,
            "oracle": prefix_correct + len(late),
            "budget_limited": sum(arm["budget_limited"] for arm in finished),
            "suffix_candidates": sum(len(arm["candidates"]) for arm in finished),
            "suffix_p50_seconds": timings[math.ceil(0.50 * len(timings)) - 1] if timings else None,
            "suffix_p95_seconds": timings[math.ceil(0.95 * len(timings)) - 1] if timings else None,
            **counts([s for arm in finished for s in arm["stages"]]),
        }
    a, b = (result["branches"][arm] for arm in BRANCHES)
    result["gate_passed"] = b["late_first_correct"] > a["late_first_correct"] and b["correct"] > a["correct"]
    result["gained_ids"] = [r["problem_id"] for r in rows if not r["answers"]["control"]["is_correct"]
                            and r["answers"]["diverse"]["is_correct"]]
    result["lost_ids"] = [r["problem_id"] for r in rows if r["answers"]["control"]["is_correct"]
                          and not r["answers"]["diverse"]["is_correct"]]
    return result
