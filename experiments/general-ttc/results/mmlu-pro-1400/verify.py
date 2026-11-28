"""Recalculate published MMLU-Pro accuracy and token totals from JSONL."""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAIRS = ("qwen-versa", "llama-versa", "qwen-dprm", "llama-dprm")
METHODS = ("E1", "E2", "E3", "E4")
COUNTS = ("questions", "correct", "invalid_answers", "generator_calls", "input_tokens",
          "output_tokens", "prm_calls", "prm_input_tokens")
TOKEN_FIELDS = ("input_tokens", "output_tokens", "prm_input_tokens")


def read_csv(name, fields):
    with (ROOT / name).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {tuple(row[k] for k in fields): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{name} has duplicate rows")
    return indexed


def check_counts(observed, stored):
    for key in COUNTS:
        assert int(stored[key]) == observed[key], (key, stored[key], observed[key])
    assert math.isclose(float(stored["accuracy"]), observed["correct"] / observed["questions"],
                        rel_tol=0, abs_tol=1e-12)
    if "total_tokens" in stored:
        assert int(stored["total_tokens"]) == sum(observed[key] for key in TOKEN_FIELDS)


def main():
    data = gzip.decompress((ROOT / "mmlu-pro-1400.json.gz").read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    manifest = json.loads(data)
    questions = {item["id"]: (item["category"], item["answer"]) for item in manifest["questions"]}
    assert len(questions) == len(manifest["questions"]) == 1400
    categories = Counter(category for category, _ in questions.values())
    assert len(categories) == 14 and set(categories.values()) == {100}

    totals = read_csv("summary.csv", ("combination", "method"))
    by_category = read_csv("by-category.csv", ("combination", "method", "category"))
    assert len(totals) == len(PAIRS) * len(METHODS)
    assert len(by_category) == len(totals) * len(categories)

    for pair in PAIRS:
        for method in METHODS:
            folder = ROOT / "runs" / pair / method
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            assert summary["manifest_sha256"] == config["manifest_sha256"] == digest
            assert config["source_sha256"] == manifest["source_sha256"]
            assert config["method"] == method == summary["method"]
            assert config["seed"] == 0 and config["max_tokens"] == 1024 and config["search_steps"] == 2
            assert config["outer_k"] == (10 if method in ("E2", "E4") else 1)

            ordered_ids = []
            counts = Counter()
            category_counts = {category: Counter() for category in categories}
            with gzip.open(folder / "predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    task = row["task_id"]
                    assert (row["category"], row["gold"]) == questions[task]
                    assert row["correct"] == (row["prediction"] == row["gold"])
                    assert len(row["candidates"]) == config["outer_k"]
                    ordered_ids.append(task)
                    fields = {key: row[key] for key in COUNTS if key in row}
                    fields.update(questions=1, invalid_answers=int(row["prediction"] is None))
                    counts.update(fields)
                    category_counts[row["category"]].update(fields)

            assert ordered_ids == config["questions"]
            assert len(set(ordered_ids)) == len(ordered_ids) == len(questions)
            check_counts(counts, summary)
            entry = totals[(pair, method)]
            check_counts(counts, entry)
            assert float(entry["elapsed_seconds"]) == summary["elapsed_seconds"]
            assert entry["timing_basis"] == summary["timing_basis"]
            for category, counter in category_counts.items():
                check_counts(counter, summary["by_category"][category])
                check_counts(counter, by_category[(pair, method, category)])
            print(pair, method, f"{counts['correct']}/{counts['questions']}",
                  sum(counts[key] for key in TOKEN_FIELDS), flush=True)

    print("Verified 16 runs, 22,400 question records, and 224 category summaries.")


if __name__ == "__main__":
    main()
