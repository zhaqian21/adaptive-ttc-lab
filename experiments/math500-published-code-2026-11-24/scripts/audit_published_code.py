"""Verify native outputs and merge the two published-code MATH-500 shards."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_results import check_problem, read

K = {"E1": 1, "E2": 10, "E3": 1, "E4": 10}
PAPER = {"E1": 0.712, "E2": 0.720, "E3": 0.742, "E4": 0.814}
LAB = {"E1": 0.684, "E2": 0.754, "E3": 0.672, "E4": 0.764}
COMMIT = "d9ca5f71f5c7d155153e8f7decd8b1c524968775"
DATASET = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, text):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def merge_case(root, case, questions):
    shards = [root / f"gpu-{gpu}" / case for gpu in (0, 1)]
    rows, profiles = [], []
    for gpu, shard in enumerate(shards):
        config = read(shard / "config.json")
        summary = read(shard / "summary.json")
        predictions = {row["problem_id"]: row for row in read(shard / "predictions.json")}
        ids = list(range(gpu, 500, 2))
        assert config["sample_ids"] == ids and set(predictions) == set(ids)
        assert config["case"] == case and config["upstream_commit"] == COMMIT and config["dataset_sha256"] == DATASET
        assert config["eval"]["num_iterations"] == K[case]
        assert config["backend"] == "vllm" and config["source_sha256"] == digest(root / "run_published_code.py")
        assert summary["samples"] == summary["requested"] == 250 and not summary["failed"]
        assert summary["correct"] == sum(row["is_correct"] is True for row in predictions.values())
        for index in ids:
            folder = shard / "problems" / f"{index:03d}"
            native = read(folder / "result.json")
            check_problem(folder, native, index, K[case], "published_code")
            source = questions[index]
            row = predictions[index]
            assert source["unique_id"] == native["dataset_unique_id"] == row["dataset_unique_id"]
            assert source["problem"] == native["problem"] and source["answer"] == native["gold_answer_raw"]
            assert row["predicted_answer"] == native["final_predicted_answer"]
            assert row["is_correct"] == native["final_is_correct"]
            assert row["selected_iteration"] == native["selected_iteration_idx"]
            assert row["llm_calls"] == len(native["observation"]["llm_calls"])
            if case in ("E1", "E2"):
                assert [call["method"] for call in native["observation"]["llm_calls"]] == [
                    "generate_with_structured", "generate"
                ] * K[case]
            row.update(gpu=gpu, candidate_oracle=any(c["is_correct"] is True for c in native["iterations"]))
            rows.append(row)
        profiles.append(summary)
    rows.sort(key=lambda row: row["problem_id"])
    assert [row["problem_id"] for row in rows] == list(range(500))
    target = root / case
    target.mkdir(exist_ok=True)
    save(target / "predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    verdicts = Counter(str(row["is_correct"]) for row in rows)
    correct = verdicts["True"]
    audit = {
        "case": case, "classification": "published_code", "samples": len(rows), "correct": correct,
        "accuracy": correct / 500, "paper_accuracy": PAPER[case], "paper_description_accuracy": LAB[case],
        "answer_verdicts": dict(verdicts), "pinned_dataset_checked": True, "native_files_verified": True,
        "candidate_oracle_correct": sum(row["candidate_oracle"] for row in rows),
        "workers_per_gpu": [row["workers"] for row in profiles],
        "generation_calls": sum(row["generation_calls"] for row in profiles),
        "prm_calls": sum(row["prm_calls"] for row in profiles),
        "question_retries": sum(row["question_retries"] for row in profiles),
        "predictions_sha256": digest(target / "predictions.jsonl"),
    }
    save(target / "audit.json", json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(f"{case}: {correct}/500 ({correct / 5:.1f}%), both shards verified", flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--ready-only", action="store_true")
    args = parser.parse_args()
    assert digest(args.dataset) == DATASET
    questions = [json.loads(row) for row in args.dataset.read_text(encoding="utf-8").splitlines()]
    assert len(questions) == len({row["unique_id"] for row in questions}) == 500
    audits = {}
    for case in K:
        if args.ready_only and not all((args.root / f"gpu-{gpu}" / case / "summary.json").is_file() for gpu in range(2)):
            continue
        audits[case] = merge_case(args.root, case, questions)
    save(args.root / "audit.json", json.dumps(audits, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
