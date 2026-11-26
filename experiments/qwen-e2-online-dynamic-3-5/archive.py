"""Audit, compare, and archive the three online arms on identical questions."""
import argparse
from collections import Counter
import gzip
from pathlib import Path
from statistics import median

from online_e2 import LANES, LIMITS, MANIFEST, WORKERS, questions, released, sha, validate
from archive_pools import encode, save


def archive(data, results, output, *, lanes=LANES, limits=LIMITS, reference="e2_fixed10"):
    pair_key = f"paired_vs_{reference}"
    cohort = questions(data)["math"]
    methods = [method for method, _, _ in lanes.values()]
    output.mkdir(parents=True, exist_ok=True)
    pools = output / "pools"
    pools.mkdir(exist_ok=True)
    settings = {}
    outcomes = {}
    totals = {}
    for lane, (method, _, _) in lanes.items():
        root = results / "formal" / method
        config = released.read(root / "config.json")
        expected_ids = [q["problem_id"] for q in cohort]
        if (config["ids"] != expected_ids or config["method"] != method or config["lane"] != lane
                or config["workers"] != WORKERS or config["decode"]["max_tokens"] != 2048
                or config["question_manifest_sha256"] != sha(MANIFEST.read_bytes())):
            raise ValueError(f"{method}: configuration or cohort mismatch")
        settings[method] = config
        timing = released.read(root / "timing.json")
        if timing["wall_seconds"] <= 0 or not 0 <= timing["resumed_questions"] <= len(cohort):
            raise ValueError(f"{method}: invalid elapsed time")
        if {f.name for f in (root / "problems").glob("*.json")} != {f"{i:03d}.json" for i in expected_ids}:
            raise ValueError(f"{method}: missing or extra question checkpoints")
        counters = Counter()
        latencies = []
        rows = []
        outcomes[method] = []
        for q in cohort:
            row = released.read(root / "problems" / f"{q['problem_id']:03d}.json")
            validate(row, q, lane, lanes=lanes, limits=limits)
            answer = row["answer"]
            correct = answer.get("is_correct", answer.get("correct")) is True
            count = len(row["candidates"] if "candidates" in row else row["outputs"])
            stopped = row.get("stopped_early", False)
            counters.update(correct=correct, candidates=count, early_stops=stopped, input_tokens=row["input_tokens"],
                            output_tokens=row["output_tokens"], prm_requests=row["prm_requests"],
                            generation_requests=row["generation_requests"], retries=len(row.get("attempts", [None])) - 1)
            latencies.append(row["seconds"])
            outcomes[method].append({
                "correct": correct, "answer": answer.get("predicted_answer", answer.get("answer")),
                "candidates": count, "stopped": stopped, "seconds": row["seconds"],
                **{k: row[k] for k in ("input_tokens", "output_tokens", "prm_requests", "generation_requests")},
            })
            rows.append(row)
        pools_sha256 = {}
        for start in range(0, len(rows), 25):
            path = pools / f"{method}_{start // 25:03d}.jsonl.gz"
            payload = gzip.compress(b"".join(encode(row) for row in rows[start:start + 25]), mtime=0)
            path.write_bytes(payload)
            pools_sha256[path.name] = sha(payload)
        totals[method] = {
            **counters, "accuracy": counters["correct"] / len(cohort), "samples": len(cohort),
            "mean_candidates": counters["candidates"] / len(cohort),
            "tokens": counters["input_tokens"] + counters["output_tokens"], "latency_median_seconds": median(latencies),
            "wall_seconds": None if timing["resumed_questions"] else timing["wall_seconds"],
            "timing_last_attempt": timing, "pools_sha256": pools_sha256,
        }

    for field in ("model", "workers", "decode", "engine", "upstream_commit", "question_manifest_sha256"):
        if any(settings[method][field] != settings[methods[0]][field] for method in methods[1:]):
            raise ValueError(f"Arms differ on shared configuration {field}")
    e2_configs = [settings[method] for method, _, prm in lanes.values() if prm is not None]
    for field in ("agent", "prm"):
        if any(config[field] != e2_configs[0][field] for config in e2_configs[1:]):
            raise ValueError(f"E2 arms have different {field} settings")
    for method, config in settings.items():
        save(output / f"{method}_config.json", config)
        pairs = zip(outcomes[reference], outcomes[method], strict=True)
        changes = Counter()
        for baseline, other in pairs:
            changes.update(lost=baseline["correct"] and not other["correct"],
                           gained=other["correct"] and not baseline["correct"])
        totals[method][pair_key] = changes
    paired = b"".join(encode({
        "problem_id": q["problem_id"], "dataset_unique_id": q["dataset_unique_id"], "gold": q["gold_answer_extracted"],
        **{method: outcomes[method][i] for method in methods},
    }) for i, q in enumerate(cohort))
    (output / "paired.jsonl").write_bytes(paired)
    save(output / "summary.json", {"methods": totals, "paired_sha256": sha(paired)})
    table = [
        "| Method | Correct | Avg candidates | Input + output tokens | PRM requests | Wall time (s) | Lost / gained |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        m = totals[method]
        wall = "resumed" if m["wall_seconds"] is None else f"{m['wall_seconds']:.1f}"
        delta = m[pair_key]
        table.append(f"| {method} | {m['correct']}/{m['samples']} ({m['accuracy']:.1%}) | {m['mean_candidates']:.2f} | "
                     f"{m['tokens']:,} | {m['prm_requests']:,} | {wall} | {delta['lost']} / {delta['gained']} |")
    (output / "table.md").write_bytes(("\n".join(table) + "\n").encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "results", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    archive(args.data, args.results, args.output)
