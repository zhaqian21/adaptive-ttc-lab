"""Replay five-answer agreement independently of the historical E4 PRM selector."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "llama-math500-answer-prm"
sys.path.insert(0, str(PREVIOUS))
from replay_selection import (
    E4_BASELINE_SHA, E4_SHA, K, answer_key, chosen, contrast, encode, load_e4, selectors, verify_bytes,
)


LEDGER_SHA = "8c568dc6578d5ff209d85c07dd8c0a84c2050f7c2c35b801849f3180ca8ba344"
PREFIX = 5
COST_FIELDS = ("generation_calls", "input_tokens", "output_tokens")
COST_METRICS = ("candidates_generated", *COST_FIELDS, "tokens", "prm_requests")
POLICIES = {
    "original": ("original", False),
    "original_early5": ("original", True),
    "sum_prm": ("sum_prm", False),
    "sum_prm_early5": ("sum_prm", True),
}
LABELS = {
    "original": "历史 E4",
    "original_early5": "原选优 + 五条一致早停",
    "sum_prm": "答案级 PRM 聚合",
    "sum_prm_early5": "答案级聚合 + 五条一致早停",
}


def prefix_agrees(candidates):
    keys = [answer_key(row["predicted_answer"]) for row in candidates[:PREFIX]]
    return len(keys) == PREFIX and keys[0] is not None and all(key == keys[0] for key in keys)


def evaluate_policy(candidates, ledger, selected_iteration, *, early_stop):
    stop = early_stop and prefix_agrees(candidates)
    count, iterations = (PREFIX, 1) if stop else (K, selected_iteration)
    costs = {key: sum(row[key] for row in ledger[:count]) for key in COST_FIELDS}
    return {
        **chosen(candidates, iterations), "stopped_early": stop, "candidates_generated": count,
        **costs, "tokens": costs["input_tokens"] + costs["output_tokens"], "prm_requests": 0 if stop else K,
    }


def replay(pool, costs, verified_previous):
    rows = []
    for (identity, candidates, reference), ledger_row, previous in zip(pool, costs, verified_previous):
        index = identity["problem_id"]
        if any(identity != {key: row[key] for key in identity} for row in (ledger_row, previous)):
            raise ValueError(f"E4 ledger and answer selection must use the same raw result at question {index}")
        ledger = ledger_row["candidate_costs"]
        if ([row["iteration"] for row in ledger] != list(range(1, K + 1))
                or any(type(row[key]) is not int or row[key] <= 0 for row in ledger for key in COST_FIELDS)
                or sum(row["generation_calls"] for row in ledger) != reference["llm_calls"]
                or sum(row["output_tokens"] for row in ledger) != reference["output_tokens"]):
            raise ValueError(f"Token ledger differs from historical E4 at question {index}")
        selections, _ = selectors(candidates)
        for name in ("original", "sum_prm"):
            if chosen(candidates, selections[name]) != previous[name]:
                raise ValueError(f"Replayed {name} choice differs at question {index}")
        outcomes = {
            name: evaluate_policy(candidates, ledger, selections[selection], early_stop=early_stop)
            for name, (selection, early_stop) in POLICIES.items()
        }
        rows.append({**identity,
                     "first_five_answers": [row["predicted_answer"] for row in candidates[:PREFIX]],
                     "failed_attempts_without_recorded_tokens": ledger_row["failed_attempts_without_recorded_tokens"],
                     **outcomes})
    return rows


def summarize(rows):
    arms = {}
    for policy in POLICIES:
        correct = sum(row[policy]["is_correct"] for row in rows)
        arms[policy] = {
            "correct": correct, "accuracy": correct / len(rows),
            "early_stops": sum(row[policy]["stopped_early"] for row in rows),
            **{cost: sum(row[policy][cost] for row in rows) for cost in COST_METRICS},
        }
    if (arms["original"]["correct"], arms["original"]["generation_calls"],
            arms["original"]["output_tokens"], arms["original"]["prm_requests"]) != (321, 85184, 9305961, 5000):
        raise ValueError("Historical cost and accuracy anchors disagree")
    if any(arms["original"][key] != arms["sum_prm"][key] or
           arms["original_early5"][key] != arms["sum_prm_early5"][key] for key in COST_METRICS):
        raise ValueError("Changing the selector alone must not change the committed replay cost")
    return arms


def run(ledger_path, output):
    pool = load_e4()
    ledger = [json.loads(line) for line in verify_bytes(ledger_path, LEDGER_SHA).splitlines()]
    reference = json.loads((PREVIOUS / "results" / "summary.json").read_bytes())["cases"]["e4"]
    paired = [json.loads(line) for line in verify_bytes(
        PREVIOUS / "results" / "e4_paired.jsonl", reference["paired_sha256"]).splitlines()]
    if len(pool) != len(ledger) or len(pool) != len(paired):
        raise ValueError("Expected 500 rows across every source")
    rows = replay(pool, ledger, paired)
    arms = summarize(rows)
    for name in ("original", "sum_prm"):
        if arms[name]["correct"] != reference["correct"][name]:
            raise ValueError(f"{name} accuracy differs from the fixed answer-selection archive")

    contrasts = {
        f"{left}_to_{right}": contrast(rows, left, right)
        for left, right in (
            ("original", "original_early5"), ("sum_prm", "sum_prm_early5"),
            ("original", "sum_prm"), ("original", "sum_prm_early5"),
            ("original_early5", "sum_prm_early5"),
        )
    }
    baseline, stopped = arms["original"], arms["original_early5"]
    saved = {key: baseline[key] - stopped[key] for key in COST_METRICS}
    payload = b"".join(encode(row) for row in rows)
    report = {
        "samples": len(rows), "agreement_count": PREFIX, "policies": arms, "contrasts": contrasts,
        "early_stop_saved": saved,
        "early_stop_saved_fraction": {key: saved[key] / baseline[key] for key in COST_METRICS},
        "failed_attempts_without_recorded_tokens": sum(r["failed_attempts_without_recorded_tokens"] for r in rows),
        "source_sha256": {
            "cost_ledger": LEDGER_SHA, "e4_candidate_pool": E4_SHA, "e4_baseline": E4_BASELINE_SHA,
            "answer_selection_pairs": reference["paired_sha256"],
        },
        "paired_sha256": hashlib.sha256(payload).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired.jsonl").write_bytes(payload)
    (output / "summary.json").write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return report


def markdown_table(report):
    lines = ["| 方案 | 正确率 | 早停题数 | 生成 token | PRM 请求 |", "|---|---:|---:|---:|---:|"]
    for name, row in report["policies"].items():
        lines.append(
            f"| {LABELS[name]} | {row['correct']}/{report['samples']}（{row['accuracy']:.1%}） | "
            f"{row['early_stops']} | {row['tokens']:,} | {row['prm_requests']:,} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=HERE / "results" / "e4_cost_ledger.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(markdown_table(run(args.ledger, args.output)))
