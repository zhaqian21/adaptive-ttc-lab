"""Replay three PRM selectors on complete and stopped E4 candidate pools."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

from prepare_aime24 import REFERENCE_SHA


HERE = Path(__file__).resolve().parent
ARCHIVES = HERE.parent / "llama-e4-online-ab" / "results"
sys.path.insert(0, str(ARCHIVES.parent))
from compare_online import verify_row
from replay_early_stop import PREFIX, prefix_agrees
from replay_selection import K, chosen, contrast, encode, selectors, verify_bytes


METHODS = ("original", "sum_prm", "top2_sum")
MODES = ("fixed_pool", "five_on_fixed_pool", "online_b_pool")


def records(path, sha, samples):
    rows = [json.loads(line) for line in verify_bytes(path, sha).splitlines()]
    if ([r["problem_id"] for r in rows] != list(range(samples))
            or len({r["dataset_unique_id"] for r in rows}) != samples):
        raise ValueError(f"Expected {samples} unique ordered questions at {path}")
    return rows


def scoring_choices(candidates):
    published, groups = selectors(candidates)
    top2 = published["original"]
    if any(g["sum_prm"] > 0 for g in groups):
        def rank(group):
            scores = sorted((candidates[i - 1]["mean_reward"] for i in group["members"]), reverse=True)
            return math.fsum(scores[:2]), group["best_reward"], -group["representative"]

        top2 = max(groups, key=rank)["representative"]
    return {"original": published["original"], "sum_prm": published["sum_prm"], "top2_sum": top2}


def replay(candidates, stop, indices):
    visible = candidates[:PREFIX] if stop else candidates
    oracle = any(c["is_correct"] is True for c in visible)
    choices = {}
    for method in METHODS:
        answer = chosen(candidates, 1 if stop else indices[method])
        if answer["is_correct"]:
            outcome = "correct"
        elif stop:
            outcome = "unanimous_wrong"
        elif oracle:
            outcome = "selector_missed"
        else:
            outcome = "no_correct_candidate"
        choices[method] = {**answer, "outcome": outcome}
    return {
        "stopped_early": stop, "candidate_count": len(visible), "logical_prm_requests": 0 if stop else K,
        "candidate_oracle_correct": oracle, "choices": choices,
    }


def fixed_modes(row):
    candidates = row["candidates"]
    indices = scoring_choices(candidates)
    return {
        "fixed_pool": replay(candidates, False, indices),
        "five_on_fixed_pool": replay(candidates, prefix_agrees(candidates), indices),
    }


def from_online(a_rows, b_rows):
    for a, b in zip(a_rows, b_rows):
        verify_row(a, "fixed")
        verify_row(b, "early")
        if (a["problem_id"], a["dataset_unique_id"]) != (b["problem_id"], b["dataset_unique_id"]):
            raise ValueError("A and B must contain the same ordered problem identities")
        modes = fixed_modes(a)
        candidates = b["candidates"]
        stop = b["stopped_early"]
        modes["online_b_pool"] = replay(candidates, stop, {} if stop else scoring_choices(candidates))
        yield {"problem_id": a["problem_id"], "dataset_unique_id": a["dataset_unique_id"], **modes}


def from_aime(a_rows, refs):
    for row, ref in zip(a_rows, refs):
        modes = fixed_modes(row)
        baseline = modes["fixed_pool"]["choices"]["original"]
        if (row["dataset_unique_id"] != ref["dataset_unique_id"]
                or row["original_iteration"] != ref["selected_iteration"]
                or baseline["selected_iteration"] != ref["selected_iteration"]
                or baseline["predicted_answer"] != ref["predicted_answer"]
                or baseline["is_correct"] != (ref["is_correct"] is True)
                or modes["fixed_pool"]["candidate_oracle_correct"] != ref["candidate_oracle_correct"]):
            raise ValueError("AIME24 candidates do not reproduce the published E4 baseline")
        yield {"problem_id": row["problem_id"], "dataset_unique_id": row["dataset_unique_id"], **modes}


def summarize(rows):
    report = {"samples": len(rows), "modes": {}, "early_stop_changes_on_fixed_pool": {}}
    for mode in MODES:
        if mode not in rows[0]:
            continue
        cases = [row[mode] for row in rows]
        report["modes"][mode] = {
            "early_stops": sum(r["stopped_early"] for r in cases),
            "candidates": sum(r["candidate_count"] for r in cases),
            "logical_prm_requests": sum(r["logical_prm_requests"] for r in cases),
            "candidate_oracle_correct": sum(r["candidate_oracle_correct"] for r in cases),
            "methods": {
                method: {
                    "correct": sum(r["choices"][method]["is_correct"] for r in cases),
                    "outcomes": dict(sorted(Counter(r["choices"][method]["outcome"] for r in cases).items())),
                } for method in METHODS
            },
        }
    for method in METHODS:
        choices = [{
            "problem_id": row["problem_id"],
            "fixed": row["fixed_pool"]["choices"][method],
            "five": row["five_on_fixed_pool"]["choices"][method],
        } for row in rows]
        report["early_stop_changes_on_fixed_pool"][method] = contrast(choices, "fixed", "five")
    if "online_b_pool" in rows[0]:
        loses = [
            row["online_b_pool"]["choices"]["sum_prm"]["outcome"] for row in rows
            if row["fixed_pool"]["choices"]["sum_prm"]["is_correct"]
            and not row["online_b_pool"]["choices"]["sum_prm"]["is_correct"]
        ]
        report["online_b_losses_against_a_sum"] = dict(sorted(Counter(loses).items()))
    return report


def verify_lock(lock, results):
    if (lock["early_stop_candidates"], lock["candidate_limit"], lock["selector_comparisons"]) != (PREFIX, K, list(METHODS)):
        raise ValueError("Selector or candidate budget differs from the lock")
    calibration = {
        case: {method: score["correct"] for method, score in results[case]["modes"]["five_on_fixed_pool"]["methods"].items()}
        for case in lock["calibration_sets"]
    }
    if calibration != lock["calibration_correct"]:
        raise ValueError("The frozen calibration counts could not be reproduced")
    gains = {
        method: min(case[method] - case["sum_prm"] for case in calibration.values())
        for method in ("original", "top2_sum")
    }
    best = max(gains, key=gains.__getitem__)
    if best != lock["locked_selector"] or gains[best] < 0 or list(gains.values()).count(gains[best]) != 1:
        raise ValueError("The locked selector does not satisfy the calibration rule")


def save_case(output, study, rows):
    payload = b"".join(encode(row) for row in rows)
    (output / f"{study}.jsonl").write_bytes(payload)
    return {**summarize(rows), "predictions_sha256": hashlib.sha256(payload).hexdigest()}


def run(archives, llama_root, output):
    lock_bytes = (HERE / "selection_lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    sources = json.loads((HERE / "sources.json").read_bytes())
    output.mkdir(parents=True, exist_ok=True)
    report = {"lock_sha256": hashlib.sha256(lock_bytes).hexdigest(), "locked_selector": lock["locked_selector"],
              "sources": sources, "studies": {}}
    studies = report["studies"]
    for study, spec in sources["online"].items():
        root = llama_root if study == "llama_math500" else archives / study.replace("_", "-")
        a_rows = records(root / "fixed/predictions.jsonl", spec["fixed_sha256"], spec["samples"])
        b_rows = records(root / "early/predictions.jsonl", spec["early_sha256"], spec["samples"])
        studies[study] = save_case(output, study, list(from_online(a_rows, b_rows)))

    verify_lock(lock, studies)
    if list(sources["heldout"]) != lock["heldout_sets"]:
        raise ValueError("Held-out datasets differ from the lock")
    aime_ids = None
    for study, spec in sources["heldout"].items():
        model = study.split("_")[0]
        a_rows = records(HERE / spec["input"], spec["input_sha256"], 30)
        refs = records(HERE.parent / spec["reference"], REFERENCE_SHA[model], 30)
        ids = [row["dataset_unique_id"] for row in a_rows]
        if aime_ids is not None and ids != aime_ids:
            raise ValueError("The two AIME24 models must use the same questions")
        aime_ids = ids
        studies[study] = save_case(output, study, list(from_aime(a_rows, refs)))

    (output / "summary.json").write_bytes(
        (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=ARCHIVES)
    parser.add_argument("--llama-math500", type=Path, help="PR #26 results directory when using separate worktrees")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.archives, args.llama_math500 or args.archives, args.output)
    for study, result in report["studies"].items():
        print(study, {mode: [m["correct"] for m in scores["methods"].values()]
                      for mode, scores in result["modes"].items()})
