"""Validate the online request ledgers and compare E4 accuracy, cost, and latency."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
for name in ("llama-math500-answer-prm", "llama-math500-five-answer-stop"):
    sys.path.insert(0, str(HERE.parent / name))
from replay_selection import K, chosen, contrast, encode, selectors
from replay_early_stop import prefix_agrees


COST = ("generation_requests", "input_tokens", "output_tokens", "prm_requests", "generation_without_usage")
LABELS = {"fixed": "固定 K=10 + 聚合", "early": "五条一致早停 + 聚合"}


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)]


def verify_row(row, name):
    index = row["problem_id"]
    label = f"{name}/{index}"
    stopped = row["stopped_early"]
    candidates = row["candidates"]
    attempts = row["attempts"]
    require(row["policy"] == name and row["lane"] == index % 2, f"{label}: wrong arm or lane")
    require(stopped == (name != "fixed" and prefix_agrees(candidates)), f"{label}: wrong early-stop decision")
    require([c["iteration"] for c in candidates] == list(range(1, 6 if stopped else 11)),
            f"{label}: wrong completed candidate count")
    require(1 <= len(attempts) <= 3 and attempts[-1]["error"] is None
            and all(attempt["error"] for attempt in attempts[:-1]), f"{label}: retry history")
    for retry, attempt in enumerate(attempts):
        require(attempt["seed"] == index + 1_000_000 * retry, f"{label}: seed schedule")
        for stage in attempt["stages"]:
            require(all(type(stage[k]) is int and stage[k] >= 0 for k in COST)
                    and stage["generation_without_usage"] <= stage["generation_requests"],
                    f"{label}: malformed request ledger")
    final = attempts[-1]["stages"]
    require(row["seed"] == attempts[-1]["seed"] and
            [stage["stage"] for stage in final] == (list(range(1, 6)) if stopped else [*range(1, 11), "prm"]),
            f"{label}: wrong successful generation stages")
    require(sum(stage["prm_requests"] for stage in final) == (0 if stopped else K)
            and all(stage["generation_requests"] >= 1 and not stage["prm_requests"]
                    for stage in final if stage["stage"] != "prm")
            and all(not stage["generation_requests"] for stage in final if stage["stage"] == "prm"),
            f"{label}: unexpected HTTP requests")
    require(all(row[k] == sum(stage[k] for attempt in attempts for stage in attempt["stages"]) for k in COST),
            f"{label}: request totals do not reconcile")
    require(math.isfinite(row["seconds"]) and row["seconds"] > 0, f"{label}: missing latency")
    if stopped:
        require(row["answer"] == chosen(candidates, 1)
                and row["original_choice"] is None and all(c["mean_reward"] is None for c in candidates),
                f"{label}: early choices cannot depend on PRM")
    else:
        selection, _ = selectors(candidates)
        require(row["answer"] == chosen(candidates, selection["sum_prm"])
                and row["original_choice"] == chosen(candidates, selection["original"]),
                f"{label}: wrong PRM selector")


def load_arm(root, name, plan, verify_source):
    path = root / name
    manifest = read(path / "summary.json")
    payload = (path / "predictions.jsonl").read_bytes()
    require(hashlib.sha256(payload).hexdigest() == manifest["predictions_sha256"], f"{name}: wrong file checksum")
    rows = [json.loads(line) for line in payload.splitlines()]
    indices = plan["smoke" if name == "smoke" else "formal"]
    require([row["problem_id"] for row in rows] == sorted(indices), f"{name}: wrong or duplicate question IDs")
    require(manifest["samples"] == len(rows) and len({r["dataset_unique_id"] for r in rows}) == len(rows),
            f"{name}: sample count or duplicate dataset identities")
    for row in rows:
        verify_row(row, name)
    require(manifest["correct"] == sum(row["answer"]["is_correct"] for row in rows)
            and manifest["early_stops"] == sum(row["stopped_early"] for row in rows)
            and all(manifest[k] == sum(row[k] for row in rows) for k in COST),
            f"{name}: summary values")
    configs = [read(path / f"gpu-{lane}_config.json") for lane in (0, 1)]
    for lane, cfg in enumerate(configs):
        require(cfg["phase"] == name and cfg["lane"] == lane
                and cfg["submission_order"] == [i for i in indices if i % 2 == lane],
                f"{name}: unexpected worker order/config")
        if verify_source:
            require(cfg["source_sha256"] == verify_source, f"{name}: online source fingerprint")
        require(cfg["plan_sha256"] == hashlib.sha256((HERE / "question_plan.json").read_bytes()).hexdigest(),
                f"{name}: question plan checksum")
    return rows, manifest, configs


def summarize(root, output, source=None):
    plan = read(HERE / "question_plan.json")
    summary = {"arms": {}}
    arms = {}
    for name in ("smoke", "fixed", "early"):
        rows, manifest, configs = load_arm(root, name, plan, source)
        arms[name] = rows, configs
        seconds = [row["seconds"] for row in rows]
        summary["arms"][name] = {
            **manifest, "accuracy": manifest["correct"] / len(rows),
            "generated_candidates": sum(len(row["candidates"]) for row in rows),
            "attempt_retries": sum(len(row["attempts"]) - 1 for row in rows),
            "output_plus_input_tokens": manifest["input_tokens"] + manifest["output_tokens"],
            "p50_seconds": percentile(seconds, .50), "p95_seconds": percentile(seconds, .95),
        }
    base, cfg_a = arms["fixed"]
    trial, cfg_b = arms["early"]
    for a, b in zip(cfg_a, cfg_b):
        require({k: v for k, v in a.items() if k != "phase"} == {k: v for k, v in b.items() if k != "phase"},
                "Formal arms used different settings")

    pairs = []
    for a, b in zip(base, trial):
        require((a["problem_id"], a["dataset_unique_id"]) == (b["problem_id"], b["dataset_unique_id"]),
                "Formal arms used different question IDs")
        shadow_stops = prefix_agrees(a["candidates"])
        pairs.append({
            "problem_id": a["problem_id"], "dataset_unique_id": a["dataset_unique_id"],
            "same_first_five_completions": [c["completion_sha256"] for c in a["candidates"][:5]]
                == [c["completion_sha256"] for c in b["candidates"][:5]],
            "same_first_five_extracted_answers": [c["predicted_answer"] for c in a["candidates"][:5]]
                == [c["predicted_answer"] for c in b["candidates"][:5]],
            "shadow_stopped_early": shadow_stops, "online_stopped_early": b["stopped_early"],
            "original": a["original_choice"],
            "fixed": {**a["answer"], "seconds": a["seconds"], **{k: a[k] for k in COST}},
            "early": {**b["answer"], "seconds": b["seconds"], **{k: b[k] for k in COST}},
            "early_on_fixed_pool": chosen(a["candidates"], 1) if shadow_stops else a["answer"],
        })
    a, b = [summary["arms"][name] for name in ("fixed", "early")]
    metrics = ("input_tokens", "output_tokens", "output_plus_input_tokens", "prm_requests",
               "generation_requests", "wall_seconds", "p95_seconds")
    summary.update({
        "paired_online": contrast(pairs, "fixed", "early"),
        "paired_fixed_pool": contrast(pairs, "fixed", "early_on_fixed_pool"),
        "original_selector_on_fixed_pool_correct": sum(row["original"]["is_correct"] for row in pairs),
        "shadow_early_correct": sum(row["early_on_fixed_pool"]["is_correct"] for row in pairs),
        "shadow_early_stops": sum(row["shadow_stopped_early"] for row in pairs),
        "identical_first_five_completions": sum(row["same_first_five_completions"] for row in pairs),
        "identical_first_five_extracted_answers": sum(row["same_first_five_extracted_answers"] for row in pairs),
        "saved": {m: a[m] - b[m] for m in metrics},
        "saved_fraction": {m: (a[m] - b[m]) / a[m] for m in metrics},
        "timing_includes_resume": any(a["completed_before_start"] + b["completed_before_start"]),
    })
    output.mkdir(parents=True, exist_ok=True)
    payload = b"".join(encode(row) for row in pairs)
    (output / "paired.jsonl").write_bytes(payload)
    summary["paired_sha256"] = hashlib.sha256(payload).hexdigest()
    (output / "comparison.json").write_bytes((json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    return summary


def table(summary):
    lines = ["| 在线方案 | 正确率 | 早停题数 | 输入 token | 输出 token | 外层 PRM 请求 | 双卡墙钟 | P95 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, label in LABELS.items():
        row = summary["arms"][name]
        lines.append(
            f"| {label} | {row['correct']}/{row['samples']}（{row['accuracy']:.1%}） | {row['early_stops']} | "
            f"{row['input_tokens']:,} | {row['output_tokens']:,} | {row['prm_requests']:,} | "
            f"{row['wall_seconds']/60:.1f} 分钟 | {row['p95_seconds']:.1f} 秒 |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(table(summarize(args.root, args.output, hashlib.sha256((HERE / "online_ab.py").read_bytes()).hexdigest())))
