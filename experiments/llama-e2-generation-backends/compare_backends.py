"""Rebuild and compare the matched E2 candidate pools question by question."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from prepare_sample import MODEL_CHOICES, digest, encode, load_manifest, read, write_json
from score_candidates import BACKENDS, PRM_MODEL, PRM_PROMPT, PRM_REVISION, load_generation


HERE = Path(__file__).resolve().parent
HISTORICAL = HERE.parent / "llama-table1-24-2026-11-25/results/math/E2/predictions.jsonl"
METRICS = {"first_correct": "首条正确", "oracle_correct": "10 条中有正确解", "selected_correct": "PRM 选优正确"}
EXTENSION_MIN_PP = 5.0
EXTENSION_MAX_P = 0.05


def verify_resume(phase, folder, generation):
    config = read(folder / "config.json")
    if "native_resume_plan_sha256" not in config:
        return
    plan_path = phase / "native_plan.json"
    if (digest(plan_path) != config["native_resume_plan_sha256"]
            or digest(HERE / "resume_native.py") != config["native_resume_source_sha256"]):
        raise ValueError("Native resume fingerprint")
    plan = read(plan_path)
    keyed = {(r["problem_id"], r["iteration"]): encode(r) for r in generation}
    seen = set()

    def verify(keys, expected):
        keys = [tuple(k) for k in keys]
        if len(set(keys)) != len(keys) or seen.intersection(keys):
            raise ValueError("Native checkpoint or shard reused a candidate")
        seen.update(keys)
        if hashlib.sha256(b"".join(keyed[key] for key in keys)).hexdigest() != expected:
            raise ValueError("Native checkpoint or shard candidates changed")

    verify(plan["retained_keys"], plan["retained_sha256"])
    runtime = read(folder / "runtime.json")
    for lane, result in enumerate(runtime["resume_workers"]):
        verify(plan["lanes"][str(lane)]["keys"], result["candidate_sha256"])
    if seen != set(keyed):
        raise ValueError("Native worker coverage mismatch")


def load_scored(phase, backend, manifest_path, manifest):
    folder = phase / backend
    generation = load_generation(folder, backend, manifest_path, manifest)
    if backend == "transformers":
        verify_resume(phase, folder, generation)
    path = folder / "scored.jsonl"
    runtime, config = read(folder / "scorer_runtime.json"), read(folder / "scorer_config.json")
    if ((config["manifest_sha256"], config["generation_sha256"], config["prm_model"], config["prm_revision"],
         config["prm_prompt"], config["source_sha256"]) !=
            (digest(manifest_path), digest(folder / "candidates.jsonl"), PRM_MODEL, PRM_REVISION,
             PRM_PROMPT, digest(HERE / "score_candidates.py"))
            or runtime["candidates"] != len(generation) or runtime["scored_sha256"] != digest(path)):
        raise ValueError(f"{backend}: PRM settings or checksum mismatch")
    scores = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len(scores) != len(generation):
        raise ValueError(f"{backend}: incomplete PRM scores")
    for gen, score in zip(generation, scores):
        if ((gen["problem_id"], gen["iteration"]) != (score["problem_id"], score["iteration"])
                or not math.isfinite(score["mean_reward"]) or not 0 <= score["mean_reward"] <= 1
                or score["is_correct"] not in (True, False, None)):
            raise ValueError(f"{backend}: candidate and PRM score disagree")
    return generation, scores


def question_result(gen, scores):
    rewards = [r["mean_reward"] for r in scores]
    index = rewards.index(max(rewards)) if any(rewards) else MODEL_CHOICES - 1
    chosen = scores[index]
    return {
        "first_correct": scores[0]["is_correct"] is True,
        "oracle_correct": any(r["is_correct"] is True for r in scores),
        "selected_correct": chosen["is_correct"] is True,
        "selected_iteration": chosen["iteration"], "predicted_answer": chosen["predicted_answer"],
        "mean_reward": rewards[index],
        "correct_candidates": sum(r["is_correct"] is True for r in scores),
        "missing_box_candidates": sum(r["predicted_answer"] is None for r in scores),
        "output_tokens": sum(r["completion_tokens"] for r in gen),
        "length_stops": sum(r["finish_reason"] == "length" for r in gen),
    }


def compare_phase(path, manifest_path):
    manifest = load_manifest(manifest_path)
    pools = {backend: load_scored(path, backend, manifest_path, manifest) for backend in BACKENDS}
    rows = []
    for offset, question in enumerate(manifest["questions"]):
        position = slice(offset * MODEL_CHOICES, (offset + 1) * MODEL_CHOICES)
        choices = {backend: question_result(gen[position], scores[position]) for backend, (gen, scores) in pools.items()}
        a, b = [pools[name][0][position] for name in BACKENDS]
        rows.append({
            "problem_id": question["problem_id"], "dataset_unique_id": question["dataset_unique_id"],
            "level": question["level"], "prompt_sha256": question["prompt_sha256"],
            "identical_candidate_tokens": sum(x["token_ids_sha256"] == y["token_ids_sha256"] for x, y in zip(a, b)),
            **choices,
        })
    return rows


def mcnemar(a, b):
    total = a + b
    return min(1.0, 2 * sum(math.comb(total, k) for k in range(min(a, b) + 1)) / (2 ** total))


def paired(rows, metric):
    diffs = defaultdict(list)
    counts = Counter()
    for row in rows:
        a, b = [row[name][metric] for name in BACKENDS]
        counts[a, b] += 1
        diffs[row["level"]].append(int(b) - int(a))
    af, bf = counts[True, False], counts[False, True]
    rng = random.Random(0)
    bootstrap = sorted(sum(sum(rng.choices(values, k=len(values))) for values in diffs.values()) for _ in range(10000))
    return {
        "transformers_only": af, "vllm_only": bf, "both": counts[True, True], "neither": counts[False, False],
        "vllm_minus_transformers_pp": 100 * (bf - af) / len(rows),
        "bootstrap_95_pp": [100 * bootstrap[i] / len(rows) for i in (250, 9749)],
        "exact_mcnemar_p": mcnemar(af, bf),
    }


def compare(args):
    rows = [row for phase in args.phases for row in
            compare_phase(args.root / phase, args.manifests / f"{phase}_manifest.json")]
    rows.sort(key=lambda row: row["problem_id"])
    n = len(rows)
    if n not in (100, 500) or len({row["dataset_unique_id"] for row in rows}) != n:
        raise ValueError("Expected 100 pilot questions or the full 500-question comparison")

    history = [json.loads(line) for line in args.historical.read_bytes().splitlines()]
    if (len(history) != 500 or [r["problem_id"] for r in history] != list(range(500))
            or sum(r["is_correct"] is True for r in history) != 322):
        raise ValueError("Expected the 500-question E2 result from PR #20")
    for row in rows:
        ref = history[row["problem_id"]]
        if ref["dataset_unique_id"] != row["dataset_unique_id"]:
            raise ValueError("Historical question identity mismatch")
        row["historical_vllm_correct"] = ref["is_correct"] is True

    fields = (*METRICS, "correct_candidates", "missing_box_candidates", "output_tokens", "length_stops")
    metrics = {name: {key: sum(r[name][key] for r in rows) for key in fields} for name in BACKENDS}
    contrasts = {metric: paired(rows, metric) for metric in METRICS}
    selected = contrasts["selected_correct"]
    expand = (n == 100 and selected["vllm_minus_transformers_pp"] >= EXTENSION_MIN_PP
              and selected["exact_mcnemar_p"] < EXTENSION_MAX_P)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "paired.jsonl"
    output.write_bytes(b"".join(encode(row) for row in rows))
    summary = {
        "samples": n, "phases": args.phases, "levels": dict(sorted(Counter(r["level"] for r in rows).items())),
        "metrics": metrics, "paired": contrasts,
        "historical_vllm_correct": sum(r["historical_vllm_correct"] for r in rows),
        "identical_candidate_tokens": sum(r["identical_candidate_tokens"] for r in rows),
        "extension_rule": {"minimum_vllm_advantage_pp": EXTENSION_MIN_PP, "exact_mcnemar_p_below": EXTENSION_MAX_P},
        "expand_to_full": expand, "paired_sha256": digest(output),
    }
    write_json(args.output / "summary.json", summary)
    return summary


def markdown_table(summary):
    n = summary["samples"]
    lines = [
        "| 指标 | Transformers | vLLM | vLLM − Transformers（百分点） |",
        "|---|---:|---:|---:|",
    ]
    for metric, label in METRICS.items():
        a, b = (summary["metrics"][name][metric] for name in BACKENDS)
        gap = summary["paired"][metric]["vllm_minus_transformers_pp"]
        lines.append(f"| {label} | {a}/{n}（{a/n:.1%}） | {b}/{n}（{b/n:.1%}） | {gap:+.1f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--manifests", type=Path, default=HERE / "results")
    parser.add_argument("--historical", type=Path, default=HISTORICAL)
    parser.add_argument("--phases", nargs="+", choices=("pilot", "extension"), default=["pilot"])
    parser.add_argument("--output", type=Path, required=True)
    print(markdown_table(compare(parser.parse_args())))
