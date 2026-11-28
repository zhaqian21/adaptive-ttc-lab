"""Verify the four online E2 studies using only the archived JSONL."""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

ARMS = {
    "k10-mean": ("mean", None),
    "k10-min": ("min", None),
    "stop5-mean": ("mean", 5),
    "stop5-min": ("min", 5),
}
TOKENS = ("input_tokens", "output_tokens", "prm_input_tokens")
FIELDS = (*TOKENS, "generator_calls", "prm_calls", "early_stopped", "correct")
FROZEN_SHA = "9246bdbb6f285f89babb74cbefa6cd3f8e443f79bf798130c867ab2d4b509dd0"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(folder, arm, rows, expected):
    rule, agreement = ARMS[arm]
    config = load(folder / "config.json")
    summary = load(folder / "summary.json")
    assert config["manifest_sha256"] == summary["manifest_sha256"] == FROZEN_SHA
    assert config["generator"] == "published-qwen25"
    assert config["scorer"]["assets"]["adapter"]["repo"] == "dongboklee/dPRM-qwen"
    assert config["seed"] == 0 and config["outer_k"] == 10 and config["max_tokens"] == 1024
    assert config["method"] == summary["method"] == "E2"
    assert config["workers"] == 96
    assert (config["prm_aggregation"], config["agreement_stop"]) == (rule, agreement)
    assert (summary["prm_aggregation"], summary["agreement_stop"]) == (rule, agreement)
    correct = {}
    durations = []
    counts = Counter()

    for row in rows:
        qid = row["task_id"]
        assert qid not in correct and (row["category"], row["gold"]) == expected[qid]
        candidates = row["candidates"]
        lead = candidates[0]["answer"]
        stop = bool(agreement and lead and all(c["answer"] == lead for c in candidates[:agreement]))
        assert row["early_stopped"] == stop
        count = agreement if stop else 10
        assert len(candidates) == row["generator_calls"] == count

        expected_calls = [] if stop else [
            index for index, c in enumerate(candidates) if c["steps"] or c["answer"]
        ]
        gen = [e for e in row["events"] if e["kind"] == "generation"]
        prm = [e for e in row["events"] if e["kind"] == "prm"]
        assert [e["iteration"] for e in gen] == list(range(count))
        assert [e["iteration"] for e in prm] == expected_calls
        assert len(prm) == row["prm_calls"]
        for index, event in enumerate(gen):
            seed = hashlib.sha256(f"0:{qid}:direct:{index}".encode()).digest()
            assert event["seed"] == int.from_bytes(seed[:4], "big") % (2**31)
        assert row["input_tokens"] == sum(e["usage"]["prompt_tokens"] for e in gen)
        assert row["output_tokens"] == sum(e["usage"]["completion_tokens"] for e in gen)
        assert row["prm_input_tokens"] == sum(e["input_tokens"] for e in prm)
        for e in prm:
            assert e["scores"] == candidates[e["iteration"]]["step_scores"]
        reduce_scores = fmean if rule == "mean" else min
        for candidate in candidates:
            scores = candidate["step_scores"]
            assert candidate["score"] == (reduce_scores(scores) if scores else None)

        def rank_key(index):
            candidate = candidates[index]
            score = candidate["score"]
            return bool(candidate["answer"]), score if score is not None else -1.0

        selected = 0 if stop else max(range(count), key=rank_key)
        assert row["selected"] == selected
        assert row["prediction"] == candidates[selected]["answer"]
        assert row["correct"] == (row["prediction"] == row["gold"])
        correct[qid] = row["correct"]
        durations.append(row["seconds"])
        counts.update({field: row[field] for field in FIELDS})

    assert len(correct) == summary["questions"] == len(expected) == 1400
    assert list(correct) == config["questions"]
    assert all(summary[field] == counts[field] for field in FIELDS)
    durations.sort()
    result = {
        "arm": arm, "aggregation": rule, "agreement_stop": agreement,
        "questions": len(correct), **{k: counts[k] for k in FIELDS},
        "accuracy": counts["correct"] / len(correct),
        "total_tokens": sum(counts[k] for k in TOKENS),
        "elapsed_seconds": summary["elapsed_seconds"],
        "timing_basis": summary["timing_basis"],
        "p95_question_seconds": durations[math.ceil(0.95 * len(durations)) - 1],
    }
    return result, correct


def main():
    root = Path(__file__).resolve().parent
    frozen = root.parent / "mmlu-pro-1400/mmlu-pro-1400.json.gz"
    data = gzip.decompress(frozen.read_bytes())
    assert hashlib.sha256(data).hexdigest() == FROZEN_SHA
    expected = {row["id"]: (row["category"], row["answer"]) for row in json.loads(data)["questions"]}
    with (root / "summary.csv").open(encoding="utf-8", newline="") as handle:
        table = {row["arm"]: row for row in csv.DictReader(handle)}
    assert set(table) == set(ARMS)
    for arm in ARMS:
        folder = root / "runs" / arm
        with gzip.open(folder / "predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
            result, _ = audit(folder, arm, map(json.loads, handle), expected)
        assert table[arm] == {k: "" if v is None else str(v) for k, v in result.items()}
        print(arm, result["correct"], result["total_tokens"], result["prm_calls"], result["elapsed_seconds"])
    print("Verified 4 conditions and 5,600 question records.")


if __name__ == "__main__":
    main()
