"""Merge completed E1 records with the missing-only run and audit the totals."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
base = root.parent
archive = base.parent / "github/experiments/aime24-amo-bench-published-code-2026-11-24"
first = root / "results-e1-other-datasets-20261128"
output = Path(sys.argv[1])
counts = {"aime24": 30, "amo": 50, "amc23": 40, "aime26": 30, "gsm8k": 1319}

def read(path):
    return json.loads(path.read_text())

reference = read(archive / "config/formal-runs.json")
specs = read(base / "datasets/manifest.json")

def completed_part(folder):
    config = read(folder / "config.json")
    part = read(folder / "summary.json")
    rows = read(folder / "predictions.json")
    assert config["sample_ids"] == [row["problem_id"] for row in rows]
    assert not part["failed"] and part["samples"] == part["requested"] == len(rows)
    assert part["correct"] == sum(row["is_correct"] is True for row in rows)
    assert part["output_tokens"] == sum(row["output_tokens"] for row in rows)
    assert part["generation_calls"] == 2 * len(rows) and not part["prm_calls"]
    assert len(list((folder / "problems").glob("*/result.json"))) == len(rows)
    return config, rows, part["seconds"], part["question_retries"]

def stopped_amo():
    folder = first / "amo/shard-0"
    config_file = folder / "config.json"
    config = read(config_file)
    marker = base / "logs/e1-other-datasets.exit"
    assert marker.read_text().strip() == "143"
    elapsed = (marker.stat().st_mtime_ns - config_file.stat().st_mtime_ns) / 1e9
    assert elapsed > 0
    records = []
    retries = 0
    for path in sorted((folder / "problems").glob("*/result.json")):
        result = read(path)
        obs = result["observation"]
        calls = obs["llm_calls"]
        assert result["problem_id"] in config["sample_ids"]
        assert len(calls) == 2 and not obs["prm_calls"]
        records.append({
            "problem_id": result["problem_id"], "dataset_unique_id": result["dataset_unique_id"],
            "predicted_answer": result["final_predicted_answer"], "is_correct": result["final_is_correct"],
            "selected_iteration": result["selected_iteration_idx"], "seed": obs["seed"],
            "llm_calls": len(calls), "output_tokens": sum(c["output_tokens"] for c in calls),
        })
        retries += len(obs["failed_attempts"])
    assert len(records) == 6
    return config, records, elapsed, retries

def aggregate(rows, times, retries):
    good = sum(row["is_correct"] is True for row in rows)
    tokens = sum(row["output_tokens"] for row in rows)
    seconds = sum(times)
    return {
        "questions":len(rows), "correct":good, "accuracy":good/len(rows), "elapsed_seconds":seconds,
        "output_tokens":tokens, "output_tokens_per_gpu_second":tokens/seconds,
        "generation_calls":sum(row["llm_calls"] for row in rows), "question_retries":retries,
        "phase_seconds":times,
    }

def compare_config(config, ref):
    for field in ("case", "dataset", "backend", "concurrency", "package_version", "upstream_commit",
                  "dataset_sha256", "source_sha256", "decode", "seed", "strict_errors", "agent",
                  "eval", "datasets_and_models"):
        assert config[field] == ref[field], (config["dataset"], field)
    for field in ("sha256","repo","revision","source_file","split","samples"):
        assert config["dataset_spec"][field] == ref["dataset_spec"][field]
    now = config["client"].copy()
    before = ref["client"].copy()
    assert Path(now.pop("model_id")).name == Path(before.pop("model_id")).name
    assert now == before

results = {}
for name, count in counts.items():
    if name == "aime24":
        phases = [completed_part(first / f"aime24/shard-{i}") for i in (0, 1)]
        labels = ["original even-ID shard", "original odd-ID shard"]
    elif name == "amo":
        phases = [stopped_amo(), completed_part(output / "amo/full")]
        labels = ["initial phase (estimated wall time)", "remaining 44 questions"]
    else:
        phases = [completed_part(output / name / "full")]
        labels = ["single unsharded run"]
    rows = sorted((row for _, items, _, _ in phases for row in items), key=lambda row: row["problem_id"])
    assert [r["problem_id"] for r in rows] == list(range(count)), name
    assert len({r["dataset_unique_id"] for r in rows}) == count and all(r["llm_calls"] == 2 for r in rows)
    assert all((row["seed"] - row["problem_id"]) in {0,1_000_000,2_000_000} for row in rows)
    for config, _, _, _ in phases:
        assert config["dataset_sha256"] == specs[name]["sha256"]
    stats = {
        "autodl":aggregate(rows, [p[2] for p in phases], sum(p[3] for p in phases)),
        "phase_labels":labels, "elapsed_is_estimated": name=="amo",
    }
    if name in reference:
        past = [reference[name]["E1"][f"gpu-{i}"] for i in (0,1)]
        for index, (config, _, _, _) in enumerate(phases):
            compare_config(config, past[index if name=="aime24" else 0]["config"])
        ref_path = archive / "results" / name / "E1"
        payload = (ref_path / "predictions.jsonl").read_bytes()
        audit = read(ref_path / "audit.json")
        assert hashlib.sha256(payload).hexdigest() == audit["predictions_sha256"]
        h20 = [json.loads(line) for line in payload.splitlines()]
        assert len(h20) == count
        assert [(x["problem_id"],x["dataset_unique_id"]) for x in rows] == [
            (y["problem_id"],y["dataset_unique_id"]) for y in h20
        ]
        stats["h20"] = aggregate(h20, [p["summary"]["seconds"] for p in past],
                                 sum(p["summary"]["question_retries"] for p in past))
        assert stats["h20"]["correct"] == audit["correct"]
        assert stats["h20"]["output_tokens"] == sum(p["summary"]["output_tokens"] for p in past)
        stats["time_comparison"] = {
            "h20_basis":"sum of historical H20 shard times",
            "autodl_to_h20_elapsed":stats["autodl"]["elapsed_seconds"]/stats["h20"]["elapsed_seconds"],
        }
        stats["paired"] = dict(Counter(
            ("autodl_correct" if x["is_correct"] is True else "autodl_incorrect") +
            ("_h20_correct" if y["is_correct"] is True else "_h20_incorrect")
            for x,y in zip(rows,h20,strict=True)
        ))
    else:
        plan = read(output / name / "question-manifest.json")
        assert [(r["problem_id"], r["dataset_unique_id"]) for r in rows] == [
            (q["problem_id"], q["dataset_unique_id"]) for q in plan
        ]
    if name == "gsm8k":
        good = sum(row["is_correct"] is True for row in rows[:500])
        stats["historical_500_subset"] = {"questions":500,"correct":good,"accuracy":good/500}
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    (folder / "predictions.jsonl").write_bytes(payload)
    stats["predictions_sha256"] = hashlib.sha256(payload).hexdigest()
    results[name] = stats
    print(name, stats["autodl"]["correct"], count, round(stats["autodl"]["elapsed_seconds"], 3), flush=True)

pending = output / "summary.json.tmp"
pending.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
pending.replace(output / "summary.json")
