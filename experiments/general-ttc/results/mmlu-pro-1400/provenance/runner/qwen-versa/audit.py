import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean

from prepare import write_json


ROOT = Path.cwd()
FROZEN = ROOT / "mmlu-pro-1400.json"
TRAIN = ROOT / "queue-e3-e2-e4"
EXPECTED = "9246bdbb6f285f89babb74cbefa6cd3f8e443f79bf798130c867ab2d4b509dd0"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory, manifest):
    spec = read(manifest)
    config = read(directory / "config.json")
    summary = read(directory / "summary.json")
    assert config["manifest_sha256"] == summary["manifest_sha256"] == sha(manifest)
    targets = {row["id"]: row for row in spec["questions"]}
    assert set(config["questions"]) == set(targets)
    rows = {}
    for path in (directory / "questions").glob("*.json"):
        row = read(path)
        task_id = row["task_id"]
        truth = targets[task_id]
        assert task_id not in rows
        assert (row["category"], row["gold"]) == (truth["category"], truth["answer"])
        assert row["correct"] == (row["prediction"] == truth["answer"])
        assert row["prediction"] == row["candidates"][row["selected"]]["answer"]
        for event in row["events"]:
            if event["kind"] == "prm":
                assert len(event["steps"]) == len(event["scores"])
                assert all(0 <= score <= 1 for score in event["scores"])
        rows[task_id] = row
    assert len(rows) == len(targets) == summary["questions"]
    assert sum(row["correct"] for row in rows.values()) == summary["correct"]
    for key in ("generator_calls", "input_tokens", "output_tokens", "prm_calls", "prm_input_tokens"):
        assert sum(row[key] for row in rows.values()) == summary[key], key
    assert not (directory / "errors.json").exists()
    return summary, rows


def prepare():
    assert sha(FROZEN) == EXPECTED, "Frozen 1400-question manifest changed"
    reference = read(FROZEN)
    categories = Counter(row["category"] for row in reference["questions"])
    assert len(categories) == 14 and set(categories.values()) == {100}
    audit(ROOT / "full" / "E1-1400", FROZEN)
    groups = {category: [] for category in categories}
    for row in reference["questions"]:
        groups[row["category"]].append(row)
    sample = []
    for category, rows in groups.items():
        sample.extend(sorted(
            rows, key=lambda row: hashlib.sha256(("e3-stress-v1:" + row["id"]).encode()).digest())[:10])
    assert len(sample) == 140
    write_json(TRAIN / "benchmark-manifest.json", {
        **reference, "questions": sample, "per_category": 10, "derived_from_sha256": EXPECTED,
    })


def choose():
    completed = []
    failed = []
    for path in (TRAIN / "benchmark").glob("c*"):
        if not path.is_dir():
            continue
        workers = int(path.name[1:].split("-")[0])
        try:
            summary, rows = audit(path, TRAIN / "benchmark-manifest.json")
        except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
            failed.append({"case": path.name, "reason": str(exc) or type(exc).__name__})
            continue
        metrics = {
            "case": path.name, "workers": workers, "questions": summary["questions"],
            "seconds": summary["elapsed_seconds"], "questions_per_minute": 60 * summary["questions"] / summary["elapsed_seconds"],
            "correct": summary["correct"], "invalid_answers": summary["invalid_answers"],
            "generation_calls": summary["generator_calls"], "prm_calls": summary["prm_calls"],
            "strategies": dict(Counter(row["candidates"][0]["route"]["strategy"] for row in rows.values())),
        }
        completed.append(metrics)
    assert completed, "No stable E3 concurrency trial"
    groups = {}
    for metrics in completed:
        groups.setdefault(metrics["workers"], []).append(metrics["questions_per_minute"])
    rates = {workers: fmean(runs) for workers, runs in groups.items()}
    winner = max(rates, key=rates.get)
    report = {
        "selected_workers": winner, "selection": "highest mean measured questions/minute among complete audited trials",
        "rates_by_workers": rates, "completed": sorted(completed, key=lambda run: run["case"]), "failed": failed,
    }
    write_json(TRAIN / "e3-benchmark.json", report)
    print(winner)


def compare(method):
    reference, baseline = audit(ROOT / "full" / "E1-1400", FROZEN)
    current, rows = audit(ROOT / "full" / f"{method}-1400", FROZEN)
    assert set(baseline) == set(rows)
    categories = {name: [task_id for task_id, row in rows.items() if row["category"] == name] for name in current["by_category"]}
    categories["All"] = list(rows)
    result = []
    for name, ids in categories.items():
        base_correct = sum(baseline[i]["correct"] for i in ids)
        method_correct = sum(rows[i]["correct"] for i in ids)
        wins = sum(rows[i]["correct"] and not baseline[i]["correct"] for i in ids)
        losses = sum(baseline[i]["correct"] and not rows[i]["correct"] for i in ids)
        assert method_correct - base_correct == wins - losses
        result.append({
            "category": name, "questions": len(ids), "e1_correct": base_correct, "method_correct": method_correct,
            "delta_pp": round(100 * (method_correct - base_correct) / len(ids), 4), "improved": wins, "regressed": losses,
            "e1_output_tokens": sum(baseline[i]["output_tokens"] for i in ids),
            "method_output_tokens": sum(rows[i]["output_tokens"] for i in ids),
            "method_prm_calls": sum(rows[i]["prm_calls"] for i in ids),
        })
    paths = TRAIN / f"{method.lower()}-vs-e1"
    write_json(paths.with_suffix(".json"), {
        "method": method, "manifest_sha256": EXPECTED, "paired": result,
        "e1_elapsed_seconds": reference["elapsed_seconds"], "method_elapsed_seconds": current["elapsed_seconds"],
        "method_timing_basis": current["timing_basis"], "e1_timing_basis": reference["timing_basis"],
    })
    with paths.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    lines = [f"# {method} against E1 on the same frozen MMLU-Pro 1,400 questions", "",
             f"| Category | Questions | E1 correct | {method} correct | Difference (pp) | Improved | Regressed |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result:
        lines.append("| " + " | ".join(str(row[key]) for key in (
            "category", "questions", "e1_correct", "method_correct", "delta_pp", "improved", "regressed")) + " |")
    paths.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{method} {current['correct']}/{current['questions']} vs E1 {reference['correct']}/{reference['questions']}; paired report saved", flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("action", choices=("prepare", "choose", "compare"))
parser.add_argument("--method", choices=("E2", "E3", "E4"))
args = parser.parse_args()
if args.action == "prepare":
    prepare()
elif args.action == "choose":
    choose()
else:
    compare(args.method)
