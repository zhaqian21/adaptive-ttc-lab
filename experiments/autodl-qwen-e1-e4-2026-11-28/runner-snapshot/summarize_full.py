import argparse
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
archive = json.loads((root / "performance/trials.json").read_text())
results = root / "results-autodl"
levels = {"E1": (16, 32, 64, 96), "E2": (24, 48), "E3": (32, 64, 96), "E4": (24, 48)}
parser = argparse.ArgumentParser()
parser.add_argument("--require-all", action="store_true")
args = parser.parse_args()

rows = []
for case, workers in levels.items():
    for concurrency in workers:
        key = f"vllm-{case}-c{concurrency}"
        trial = results / f"{case.lower()}-c{concurrency}"
        status = trial / "summary.json"
        if not status.is_file():
            continue
        prior = archive[key]
        old = prior["summary"]
        now = json.loads(status.read_text())
        config = json.loads((trial / "config.json").read_text())
        for field in ("case", "backend", "concurrency", "sample_ids", "decode", "seed",
                      "strict_errors", "dataset_sha256", "upstream_commit", "source_sha256",
                      "package_version", "agent", "eval", "datasets_and_models"):
            assert config[field] == prior["config"][field], (key, field)
        for field in ("samples", "requested", "prm_calls"):
            assert now[field] == old[field], (key, field, now[field], old[field])
        assert not now["failed"] and not now["question_retries"], key
        if case in {"E1", "E2"}:
            assert now["generation_calls"] == old["generation_calls"], key
        assert len(list((trial / "problems").glob("*/result.json"))) == now["samples"], key
        rows.append({
            "case": case, "concurrency": concurrency, "questions": now["samples"],
            "autodl_seconds": now["seconds"], "h20_seconds": old["seconds"],
            "autodl_questions_per_minute": now["questions_per_minute"],
            "h20_questions_per_minute": old["questions_per_minute"],
            "autodl_output_tokens_per_second": now["output_tokens_per_second"],
            "h20_output_tokens_per_second": old["output_tokens_per_second"],
            "throughput_pct_of_h20": 100 * now["output_tokens_per_second"] / old["output_tokens_per_second"],
            "autodl_output_tokens": now["output_tokens"], "h20_output_tokens": old["output_tokens"],
            "autodl_correct": now["correct"], "h20_correct": old["correct"],
            "autodl_generation_calls": now["generation_calls"], "h20_generation_calls": old["generation_calls"],
            "prm_calls": now["prm_calls"], "autodl_prm_seconds": now["prm_compute_seconds"],
            "h20_prm_seconds": old["prm_compute_seconds"],
            "autodl_prm_queue_s": now["prm_queue_seconds"], "h20_prm_queue_s": old["prm_queue_seconds"],
            "autodl_client_runtime": config["runtime"], "h20_client_runtime": prior["config"]["runtime"],
        })

expected = sum(len(workers) for workers in levels.values())
if args.require_all:
    assert len(rows) == expected, f"Only {len(rows)} of {expected} trials verified"

def publish(name, contents):
    path = results / name
    staged = path.with_suffix(".tmp")
    staged.write_text(contents, encoding="utf-8")
    staged.replace(path)

publish("h20-comparison.json", json.dumps(rows, indent=2) + "\n")
if rows:
    fields = [key for key in rows[0] if not key.endswith("_runtime")]
    temporary = results / "h20-comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(results / "h20-comparison.csv")
print(f"Audited {len(rows)}/{expected} matched trials")
for row in rows:
    print(f"{row['case']} c{row['concurrency']}: "
          f"4090 {row['autodl_seconds']:.2f}s, H20 {row['h20_seconds']:.2f}s, "
          f"{row['throughput_pct_of_h20']:.1f}% output tok/s")
