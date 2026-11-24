"""Verify both shards and export the paired MATH-500 records."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from audit_math500 import load, shard_files
from run_adaptive_k import baseline, runner
from sampling import FIXED_COUNTS, MAX_ATTEMPTS, POLICIES, SELECTORS, TOKEN_LIMIT, Prefix, agreed, costs, select_adaptive


def verify_row(row, question, evaluator, mode, selector="single_prm"):
    calls, candidates, attempts = row["calls"], row["candidates"], row["attempts"]
    if row["problem_id"] != question["problem_id"] or row["dataset_unique_id"] != question["dataset_unique_id"]:
        raise ValueError("Dataset row mismatch")
    if not 0 < len(attempts) <= MAX_ATTEMPTS:
        raise ValueError("Unexpected attempt count")
    if "inference_seconds" in row and (not math.isfinite(row["inference_seconds"]) or row["inference_seconds"] < 0):
        raise ValueError("Invalid inference time")

    ends, root_attempts = [], []
    previous = 0
    for index, attempt in enumerate(attempts):
        end, root = attempt["calls"], attempt["root"]
        if not previous <= end <= len(calls):
            raise ValueError("Generation call ranges are inconsistent")
        if root is not None:
            if root != len(ends) or end == previous:
                raise ValueError("Candidate indices are inconsistent")
            ends.append(end)
            root_attempts.append(index)
        previous = end
    if len(ends) != len(candidates) or previous != len(calls):
        raise ValueError("Candidate or call coverage is incomplete")

    boundary = row["budget_boundary"]
    charged, first_over = 0, None
    for index, (input_tokens, output_tokens, max_output_tokens) in enumerate(calls):
        if input_tokens <= 0 or not 0 <= output_tokens <= max_output_tokens:
            raise ValueError("Invalid request accounting")
        if first_over is None and charged + input_tokens + max_output_tokens > TOKEN_LIMIT:
            first_over = (index, input_tokens, max_output_tokens)
        charged += input_tokens + output_tokens
    if boundary is None:
        if first_over is not None:
            raise ValueError("Budget boundary was omitted")
    else:
        call, attempt = boundary["calls"], boundary["attempt"]
        if not 0 <= call <= len(calls) or not 0 <= attempt < len(attempts):
            raise ValueError("Budget boundary lies outside the trace")
        if not (attempts[attempt - 1]["calls"] if attempt else 0) <= call <= attempts[attempt]["calls"]:
            raise ValueError("Budget boundary belongs to the wrong candidate attempt")
        reserved = (call, boundary["input_tokens"], boundary["max_output_tokens"])
        new_request = sum(x[0] + x[1] for x in calls[:call]) + reserved[1] + reserved[2]
        if (boundary["input_tokens"] <= 0 or boundary["max_output_tokens"] <= 0 or new_request <= TOKEN_LIMIT
                or (call < len(calls) and first_over != reserved)
                or (call == len(calls) and first_over is not None)):
            raise ValueError("Invalid first over-budget request")

    predictions = [candidate["prediction"] for candidate in candidates]
    early = Prefix(3, ends[2], "agreement") if agreed(predictions) else None
    if len(attempts) >= 3 and all(a["root"] is None for a in attempts[:3]) and (
        boundary is None or boundary["attempt"] > 2
    ):
        full = Prefix(0, attempts[2]["calls"], "no_initial_candidates")
    elif boundary is not None:
        full = Prefix(sum(i < boundary["attempt"] for i in root_attempts), boundary["calls"], "token_budget")
    else:
        full = None

    if early and full and full.candidates < 3:
        early = None
    if mode == "paired":
        if full is None or len(candidates) < max(FIXED_COUNTS.values()):
            raise ValueError("Fixed or full arms did not finish")
        expected = {p: Prefix(k, ends[k - 1], "fixed_count") for p, k in FIXED_COUNTS.items()}
        expected.update(full=full, adaptive=early or full)
    else:
        if early is None and full is None:
            raise ValueError("The adaptive policy did not stop")
        expected = {"adaptive": early or full}

    max_scores = max((p.candidates for p in expected.values() if p.reason != "agreement"), default=0)
    if (len(candidates) != max(p.candidates for p in expected.values())
            or len(calls) != max(p.calls for p in expected.values())
            or set(row["policies"]) != set(expected)):
        raise ValueError("Physical work did not match the policy checkpoints")
    for index, candidate in enumerate(candidates):
        actual = bool(evaluator._check_answers_equiv(candidate["prediction"], question["gold_answer_extracted"]))
        reward, seconds = candidate["mean_reward"], candidate["prm_seconds"]
        if candidate["correct"] != actual:
            raise ValueError("Candidate grading mismatch")
        if index < max_scores:
            if not math.isfinite(reward) or not 0 <= reward <= 1 or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid PRM score or request time")
        elif reward is not None or seconds is not None:
            raise ValueError("Unneeded PRM request")

    for name, prefix in expected.items():
        arm = row["policies"][name]
        accounting = {**asdict(prefix), **costs(calls, prefix.calls)}
        index = select_adaptive(candidates, prefix, selector)
        prm_calls = 0 if prefix.reason == "agreement" else prefix.candidates
        selected = candidates[index] if index is not None else {"prediction": None, "correct": False}
        if (any(arm[k] != v for k, v in accounting.items()) or arm["selected_index"] != index
                or arm["prediction"] != selected["prediction"] or arm["correct"] != selected["correct"]
                or arm["prm_calls"] != prm_calls
                or not math.isclose(arm["prm_request_seconds"], sum(c["prm_seconds"] for c in candidates[:prm_calls]))):
            raise ValueError(f"Result mismatch for {name}")
    return max_scores


def audit(root, wall_seconds):
    configs, executions, rows = [], [], []
    physical_prm_calls, retries = 0, []
    evaluator = runner.upstream.get_evaluator("MATH")
    for lane in (0, 1):
        shard = root / f"shard-{lane}"
        config, execution, files = shard_files(shard, lane)
        selector = config.get("selector", "single_prm")
        if selector not in SELECTORS or (config["mode"] != "adaptive" and selector != "single_prm"):
            raise ValueError(f"Wrong selection rule on shard {lane}")
        if execution["failures"] or config["dataset_sha256"] != baseline.DATASET_SHA256:
            raise ValueError(f"Wrong data or unresolved failures on shard {lane}")
        args = SimpleNamespace(shard=lane, shards=2, sample=config["sample"], case="E4", priority_file=None)
        expected = {q["problem_id"]: q for q in baseline.dataset_for_run(args)}
        if set(expected) != set(files):
            raise ValueError(f"Wrong sample on shard {lane}")
        for index, path in sorted(files.items()):
            row = load(path)
            try:
                physical_prm_calls += verify_row(row, expected[index], evaluator, config["mode"], selector)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {index}: {error}") from error
            rows.append(row)
        configs.append(config)
        executions.append(execution)
        first_attempt = shard / "first-attempt.json"
        if first_attempt.exists():
            previous = load(first_attempt)
            retries.append({"shard": lane, "failed_rows": sorted(map(int, previous["failures"])),
                            "first_attempt_seconds": previous["seconds"]})

    def comparable(config):
        return {k: v for k, v in config.items() if k not in ("shard", "question_ids")}

    if comparable(configs[0]) != comparable(configs[1]):
        raise ValueError("Shard configurations differ")
    mode = configs[0]["mode"]
    sample = configs[0]["sample"]
    rows.sort(key=lambda row: row["problem_id"])
    if len(rows) != (sample or 500) or len({row["dataset_unique_id"] for row in rows}) != len(rows):
        raise ValueError("Wrong total or duplicate dataset IDs")
    summary = {}
    for name in (POLICIES if mode == "paired" else ("adaptive",)):
        arms = [row["policies"][name] for row in rows]
        totals = {key: sum(arm[key] for arm in arms) for key in (
            "correct", "candidates", "generation_calls", "generation_input_tokens",
            "generation_output_tokens", "prm_calls", "prm_request_seconds",
        )}
        totals["accuracy"] = totals["correct"] / len(rows)
        totals["stop_reasons"] = dict(Counter(arm["reason"] for arm in arms))
        if mode == "paired":
            pairs = Counter((row["policies"]["full"]["correct"], row["policies"][name]["correct"]) for row in rows)
            totals["paired_vs_full"] = {"gained": pairs[(False, True)], "lost": pairs[(True, False)]}
        summary[name] = totals
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")
    destination = root / "export"
    destination.mkdir(exist_ok=True)
    (destination / "predictions.jsonl").write_bytes(payload)
    result = {
        "mode": mode, "samples": len(rows), "formal": sample is None, "wall_seconds": wall_seconds,
        "physical_generation_calls": sum(len(row["calls"]) for row in rows),
        "physical_prm_calls": physical_prm_calls, "configs": configs, "executions": executions,
        "retries": retries, "policies": summary, "predictions_sha256": hashlib.sha256(payload).hexdigest(),
    }
    (destination / "summary.json").write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"samples": len(rows), "policies": summary}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    audit(args.root, args.wall_seconds)
