"""Audit AutoDL E3 results and compare with the matched H20 archive."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
base = root.parent
archive = base.parent / "github/experiments/aime24-amo-bench-published-code-2026-11-24"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--ready-only", action="store_true")
args = parser.parse_args()
output = args.output
counts = {"math500":500, "aime24":30, "amo":50, "amc23":40, "aime26":30, "gsm8k":1319}
reference_fields = (
    "case", "backend", "concurrency", "package_version", "upstream_commit", "dataset_sha256",
    "source_sha256", "decode", "seed", "strict_errors", "agent", "eval", "datasets_and_models",
)

def read(path):
    return json.loads(path.read_text())

math_refs = read(root / "config/formal-runs.json")
other_refs = read(archive / "config/formal-runs.json")
specs = read(base / "datasets/manifest.json")

def stats(rows, slices):
    seconds = sum(s["seconds"] for s in slices)
    correct = sum(r["is_correct"] is True for r in rows)
    tokens = sum(r["output_tokens"] for r in rows)
    calls = sum(r["llm_calls"] for r in rows)
    assert correct == sum(s["correct"] for s in slices)
    assert len(rows) == sum(s["samples"] for s in slices)
    assert tokens == sum(s["output_tokens"] for s in slices)
    assert calls == sum(s["generation_calls"] for s in slices)
    return {
        "questions":len(rows), "correct":correct, "accuracy":correct / len(rows),
        "elapsed_seconds":seconds, "per_part_seconds":[s["seconds"] for s in slices],
        "output_tokens":tokens, "output_tokens_per_gpu_second":tokens / seconds,
        "generation_calls":calls,
        "question_retries":sum(s["question_retries"] for s in slices),
    }

result = {}
for name, count in counts.items():
    directory = output / name
    full = directory / "full"
    status = full / "summary.json"
    if args.ready_only and (not status.is_file() or read(status)["failed"]):
        continue
    summary = read(status)
    config = read(full / "config.json")
    rows = read(full / "predictions.json")
    ids = list(range(count))
    assert summary["case"] == "E3" and not summary["prm_calls"] and not summary["failed"]
    assert summary["samples"] == summary["requested"] == count
    assert sorted(config["sample_ids"]) == sorted(config["submission_order"]) == ids
    assert [r["problem_id"] for r in rows] == ids
    assert len({r["dataset_unique_id"] for r in rows}) == count
    assert all(r["llm_calls"] >= 2 and r["seed"] - r["problem_id"] in (0,1000000,2000000) for r in rows)
    assert len(list((full / "problems").glob("*/result.json"))) == count
    assert config["dataset_sha256"] == specs[name]["sha256"]
    row = {"autodl":stats(rows,[summary]), "concurrency":config["concurrency"]}
    if name == "math500" or name in other_refs:
        if name == "math500":
            refs = [math_refs[f"gpu-{n}"]["E3"] for n in (0,1)]
            prior_file = output / "h20-math500-e3-predictions.jsonl"
            expected_hash = "5f023d8fc05ee1216d2d182b905997f4ca6f39c15b8656f542d917742317b276"
        else:
            refs = [other_refs[name]["E3"][f"gpu-{n}"] for n in (0,1)]
            archive_dir = archive / "results" / name / "E3"
            prior_file = archive_dir / "predictions.jsonl"
            expected_hash = read(archive_dir / "audit.json")["predictions_sha256"]
        for past in refs:
            for key in reference_fields:
                assert config[key] == past["config"][key], (name,key)
            found_client = config["client"].copy()
            expected_client = past["config"]["client"].copy()
            assert Path(found_client.pop("model_id")).name == Path(expected_client.pop("model_id")).name
            assert found_client == expected_client
        prior = prior_file.read_bytes()
        assert hashlib.sha256(prior).hexdigest() == expected_hash
        old_rows = [json.loads(line) for line in prior.splitlines()]
        assert [(x["problem_id"],x["dataset_unique_id"]) for x in rows] == [
            (x["problem_id"],x["dataset_unique_id"]) for x in old_rows
        ]
        row["h20"] = stats(old_rows,[p["summary"] for p in refs])
        row["h20"]["timing_basis"] = "sum of the archived H20 shard times"
        row["paired"] = dict(Counter(
            ("autodl_correct" if x["is_correct"] is True else "autodl_incorrect") +
            ("_h20_correct" if y["is_correct"] is True else "_h20_incorrect")
            for x,y in zip(rows,old_rows,strict=True)
        ))
    else:
        manifest = read(directory / "question-manifest.json")
        assert [(r["problem_id"],r["dataset_unique_id"]) for r in rows] == [
            (x["problem_id"],x["dataset_unique_id"]) for x in manifest
        ]
    if name == "gsm8k":
        correct = sum(r["is_correct"] is True for r in rows[:500])
        row["historical_500_subset"] = {"questions":500, "correct":correct, "accuracy":correct/500}
    predictions = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    (directory / "predictions.jsonl").write_bytes(predictions)
    row["predictions_sha256"] = hashlib.sha256(predictions).hexdigest()
    result[name] = row
    print(f"E3 {name} audit: {row['autodl']['correct']}/{count}, {summary['seconds']:.2f}s, "
          f"{row['autodl']['generation_calls']} calls", flush=True)

if not args.ready_only:
    assert set(result) == set(counts) and sum(row["autodl"]["questions"] for row in result.values()) == 1969
final = output / "summary.json"
pending = final.with_suffix(".json.tmp")
pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
pending.replace(final)

from compare_e3_e1 import compare
compare(output, result)
