"""Rebuild all paired PRM choices and separate marker placement from aggregation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from score_control import (
    CASES, HERE, INPUT_SHA256, K, PARAGRAPH_BREAK, PRM_PROMPT, PRM_REVISION,
    REFERENCE, SAMPLES, STEP_MARKER, digest, encode, read_pool,
)
from compare_replay import BASELINE, PAPER_CORRECT, across_cases, load_baseline, selected_record


ARMS = {
    "historical_whole": "历史整段",
    "same_text_whole": "同文本整段",
    "step_last": "同文本逐步，取末步",
    "step_mean": "同文本逐步，取均值",
}
CONTRASTS = (
    ("historical_whole", "same_text_whole"),
    ("same_text_whole", "step_last"),
    ("step_last", "step_mean"),
    ("same_text_whole", "step_mean"),
)


def read(path):
    return json.loads(path.read_bytes())


def load_shards(root):
    results, manifests = {}, []
    configs = []
    for lane in (0, 1):
        folder = root / f"gpu-{lane}"
        config, runtime = read(folder / "config.json"), read(folder / "runtime.json")
        path = folder / "paired_scores.jsonl"
        if ((config["lane"], config["input_sha256"], config["prm_revision"], config["system_prompt"],
             config["step_marker"], config["paragraph_break"], config["dtype"]) !=
                (lane, INPUT_SHA256, PRM_REVISION, PRM_PROMPT, STEP_MARKER, PARAGRAPH_BREAK, "torch.bfloat16")
                or runtime["scores_sha256"] != digest(path)):
            raise ValueError(f"GPU {lane}: scoring setup or result checksum")
        for name, source in (("score_control.py", HERE), ("prepare_pool.py", REFERENCE), ("score_steps.py", REFERENCE)):
            if config["source_sha256"][name] != digest(source / name):
                raise ValueError(f"GPU {lane}: scorer code fingerprint for {name}")
        shard = [json.loads(line) for line in path.read_bytes().splitlines()]
        keys = [(row["case"], row["problem_id"], row["iteration"]) for row in shard]
        expected = [(case, index, i) for case in CASES for index in range(lane, SAMPLES, 2)
                    for i in range(1, K + 1)]
        if keys != expected or runtime["candidate_pairs"] != len(expected) or set(results).intersection(keys):
            raise ValueError(f"GPU {lane}: missing, unexpected, or duplicate candidate pairs")
        results.update(zip(keys, shard))
        configs.append({key: value for key, value in config.items() if key not in ("lane", "batch_tokens")})
        manifests.append({"config": config, "runtime": runtime})
    if configs[0] != configs[1]:
        raise ValueError("The two GPU shards scored under different model configurations")
    return results, manifests


def verify_checkpoint(scores_root, scores):
    path = scores_root.parent / "checkpoint_resume.json"
    if not path.is_file():
        return None
    record = read(path)
    source = (HERE / "score_control.py").read_bytes()
    fixed = b"done.update((case, index, iteration) for case, index, iteration, _, _ in scored)"
    initial = b"done.update(key for key, _, _, _, _ in scored)"
    if (source.count(fixed) != 1
            or hashlib.sha256(source.replace(fixed, initial)).hexdigest() != record["initial_source_sha256"]
            or hashlib.sha256(source).hexdigest() != record["resume_source_sha256"]):
        raise ValueError("The checkpoint did not resume after the recorded counting fix")
    verified = []
    for lane in (0, 1):
        saved = record["lanes"][str(lane)]
        keys = [tuple(key) for key in saved["saved_keys"]]
        if (keys != sorted(set(keys)) or len(keys) != saved["saved_candidate_pairs"]
                or any(key not in scores or key[1] % 2 != lane for key in keys)):
            raise ValueError(f"GPU {lane}: invalid checkpoint candidate IDs")
        payload = b"".join(encode(scores[key]) for key in keys)
        if hashlib.sha256(payload).hexdigest() != saved["checkpoint_scores_sha256"]:
            raise ValueError(f"GPU {lane}: resumed scores differ from the saved checkpoint")
        verified.append(len(keys))
    return {"time": record["time"], "saved_pairs_per_gpu": verified, "sha256": digest(path)}


def evaluate_case(case, inputs, baseline, scores):
    path = inputs / f"{case.lower()}_pool.jsonl.gz"
    if digest(path) != INPUT_SHA256[case]:
        raise ValueError(f"{case}: candidate pool checksum")
    pool = read_pool(path)
    refs, baseline_sha = load_baseline(baseline / case, case)
    rows = []
    for ref, archived in zip(refs, pool):
        index = ref["problem_id"]
        if (archived["case"], archived["problem_id"], archived["dataset_unique_id"], archived["baseline_sha256"]) != (
                case, index, ref["dataset_unique_id"], baseline_sha):
            raise ValueError(f"{case}/{index}: candidate pool and historical verdicts differ")
        candidates = archived["candidates"]
        original, whole, last, average = [], [], [], []
        for item in candidates:
            record = scores[case, index, item["iteration"]]
            rewards = record["step_rewards"]
            scalar = record["whole_score"]
            if (len(rewards) != len(item["steps"]) or not rewards
                    or any(not math.isfinite(value) or not 0 <= value <= 1 for value in [scalar, *rewards])):
                raise ValueError(f"{case}/{index}/{item['iteration']}: invalid score counts or values")
            original.append(item["legacy_score"])
            whole.append(scalar)
            last.append(rewards[-1])
            average.append(mean(rewards))
        choices = {arm: selected_record(candidates, values)
                   for arm, values in zip(ARMS, (original, whole, last, average))}
        winner = choices["historical_whole"]
        if (winner["selected_iteration"], winner["answer"], winner["is_correct"]) != (
                ref["selected_iteration"], ref["predicted_answer"], ref["is_correct"]):
            raise ValueError(f"{case}/{index}: historical choice did not match PR #20")
        rows.append({
            "case": case, "problem_id": index, "dataset_unique_id": ref["dataset_unique_id"],
            "historical_result_sha256": archived["historical_result_sha256"], **choices,
        })
    return rows


def compare_pair(rows, a, b):
    changed = [r for r in rows if r[a]["selected_iteration"] != r[b]["selected_iteration"]]
    gains = [r["problem_id"] for r in changed if r[a]["is_correct"] is not True and r[b]["is_correct"] is True]
    losses = [r["problem_id"] for r in changed if r[a]["is_correct"] is True and r[b]["is_correct"] is not True]
    return {
        "from": a, "to": b, "changed_iteration": len(changed), "wrong_to_correct": len(gains),
        "correct_to_wrong": len(losses), "wrong_to_correct_ids": gains, "correct_to_wrong_ids": losses,
        "change_pp": 100 * (len(gains) - len(losses)) / SAMPLES,
    }


def run(inputs, baseline, scores_root, output):
    scores, manifests = load_shards(scores_root)
    checkpoint = verify_checkpoint(scores_root, scores)
    rows = {case: evaluate_case(case, inputs, baseline, scores) for case in CASES}
    paired = {arm: across_cases(rows["E2"], rows["E4"], arm) for arm in ARMS}
    output.mkdir(parents=True, exist_ok=True)
    cases = {}
    for case, entries in rows.items():
        output_file = output / f"{case.lower()}_paired.jsonl"
        output_file.write_bytes(b"".join(encode(row) for row in entries))
        totals = {arm: sum(r[arm]["is_correct"] is True for r in entries) for arm in ARMS}
        cases[case] = {
            "samples": SAMPLES, "n_candidates": K * SAMPLES,
            "correct": totals, "accuracy": {arm: n / SAMPLES for arm, n in totals.items()},
            "gap_vs_paper_pp": {arm: 100 * (n - PAPER_CORRECT[case]) / SAMPLES for arm, n in totals.items()},
            "paired_sha256": digest(output_file),
            "contrasts": {f"{a}_to_{b}": compare_pair(entries, a, b) for a, b in CONTRASTS},
        }
    summary = {"cases": cases, "paired_across_cases": paired, "shards": manifests}
    if checkpoint is not None:
        summary["verified_resume"] = checkpoint
    (output / "summary.json").write_bytes((json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return summary


def markdown_table(summary):
    lines = [
        "| PRM 评分方式 | E2 | E4 | E2 − E4（百分点） |",
        "|---|---:|---:|---:|",
    ]
    for arm, label in ARMS.items():
        a, b = [summary["cases"][case]["correct"][arm] for case in CASES]
        gap = summary["paired_across_cases"][arm]["e2_minus_e4_pp"]
        lines.append(f"| {label} | {a}/500（{a / SAMPLES:.1%}） | {b}/500（{b / SAMPLES:.1%}） | {gap:+.1f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=REFERENCE / "inputs")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--scores", type=Path, default=HERE / "results" / "scores")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(markdown_table(run(args.inputs, args.baseline, args.scores, args.output)))
