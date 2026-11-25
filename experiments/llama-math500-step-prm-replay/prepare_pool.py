"""Extract fixed paper-style step inputs and verify the historical PRM choices."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean


CASES = ("E2", "E4")
SAMPLES = 500
K = 10
EXPECTED_CORRECT = {"E2": 322, "E4": 321}
PRM_REVISION = "0610740060112df12585d00a1c5f4624d2f59051"
DATASET_SHA = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
UPSTREAM = "d9ca5f71f5c7d155153e8f7decd8b1c524968775"
PRM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."
STEP_MARKER = "<extra_0>"
VERDICT_NOTES = ("Numeric verification:", "General verification:")
TOOL_LABEL = re.compile(r"^(?:Reframed|Summary):\s*", re.IGNORECASE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(row):
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def split_steps(text):
    """Reuse the paragraph and line rule from scripts/paper_protocol.py."""
    text = text.replace("\r\n", "\n").strip()
    if STEP_MARKER in text:
        parts = text.split(STEP_MARKER)
    else:
        parts = re.split(r"\n[ \t]*\n+", text)
        if len(parts) == 1:
            parts = text.splitlines()
    return [part.strip() for part in parts if part.strip()]


def steps_for(case, iteration):
    completion = iteration["answer_unstructured"]
    if case == "E2":
        return split_steps(completion)
    steps = [
        step for entry in iteration["reasoning_steps"]
        if not entry.startswith(VERDICT_NOTES)
        for step in split_steps(TOOL_LABEL.sub("", entry))
    ]
    if completion != iteration["reasoning"] or not steps:
        steps.extend(split_steps(completion))
    return steps


def winner(rewards):
    highest = max(rewards)
    return rewards.index(highest) + 1 if highest else len(rewards)


def load_baseline(path, case):
    audit = json.loads((path / "audit.json").read_bytes())
    predictions = path / "predictions.jsonl"
    if digest(predictions) != audit["predictions_sha256"]:
        raise ValueError(f"Historical predictions hash differs for {case}")
    rows = [json.loads(line) for line in predictions.read_bytes().splitlines()]
    if ((audit["case"], audit["samples"], audit["correct"], audit["upstream_commit"])
            != (case, SAMPLES, EXPECTED_CORRECT[case], UPSTREAM)
            or [r["problem_id"] for r in rows] != list(range(SAMPLES))
            or len({r["dataset_unique_id"] for r in rows}) != SAMPLES
            or sum(r["is_correct"] is True for r in rows) != EXPECTED_CORRECT[case]
            or any(c["dataset_sha256"] != DATASET_SHA or c["eval"]["num_iterations"] != K
                   or c["eval"]["prm_selection_metric"] != "mean_reward" for c in audit["configs"])):
        raise ValueError(f"Expected the full PR #20 MATH-500 {case} baseline")
    return rows, audit["predictions_sha256"]


def prepare_question(case, history, ref, baseline_sha):
    index, lane = ref["problem_id"], ref["lane"]
    file = history / f"formal/math/gpu-{lane}/{case}/problems/{index:03d}/result.json"
    raw = file.read_bytes()
    native = json.loads(raw)
    iterations = native["iterations"]
    scores = native["observation"]["prm_calls"]
    if ((native["problem_id"], native["dataset_unique_id"], native["selected_iteration_idx"],
         native["final_predicted_answer"], native["final_is_correct"]) !=
            (index, ref["dataset_unique_id"], ref["selected_iteration"], ref["predicted_answer"], ref["is_correct"])
            or [i["iteration"] for i in iterations] != list(range(1, K + 1))
            or len(scores) != K):
        raise ValueError(f"{case}/{index}: historical trajectory list differs from PR #20")
    candidates = []
    for item, score in zip(iterations, scores):
        values = score["step_rewards"]
        legacy = score["mean_reward"]
        steps = steps_for(case, item)
        if (len(values) != 1 or not 0 <= legacy <= 1 or not math.isclose(legacy, mean(values))
                or item["prm_scores"]["mean_reward"] != legacy or not steps):
            raise ValueError(f"{case}/{index}/{item['iteration']}: invalid historical reward or step input")
        candidates.append({
            "iteration": item["iteration"], "answer": item["predicted_answer"],
            "correct": item["is_correct"], "legacy_score": legacy, "steps": steps,
        })

    selected = winner([c["legacy_score"] for c in candidates])
    chosen = candidates[selected - 1]
    if (selected, chosen["answer"], chosen["correct"]) != (
            ref["selected_iteration"], ref["predicted_answer"], ref["is_correct"]):
        raise ValueError(f"{case}/{index}: historical selector does not reproduce PR #20")
    return {
        "case": case, "problem_id": index, "dataset_unique_id": ref["dataset_unique_id"],
        "historical_result_sha256": hashlib.sha256(raw).hexdigest(), "baseline_sha256": baseline_sha,
        "problem": native["problem"], "candidates": candidates,
    }


def prepare(history, baseline, output):
    output.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        refs, sha = load_baseline(baseline / case, case)
        path = output / f"{case.lower()}_pool.jsonl.gz"
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as file, gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as stream:
            for ref in refs:
                stream.write(encode(prepare_question(case, history, ref, sha)))
        temp.replace(path)
        print(path.name, digest(path), flush=True)


def read_pool(path):
    with gzip.open(path, "rb") as source:
        rows = [json.loads(line) for line in source]
    if ([row["problem_id"] for row in rows] != list(range(SAMPLES))
            or len({row["dataset_unique_id"] for row in rows}) != SAMPLES
            or any([c["iteration"] for c in r["candidates"]] != list(range(1, K + 1)) for r in rows)
            or len({row["case"] for row in rows}) != 1 or rows[0]["case"] not in CASES
            or len({row["baseline_sha256"] for row in rows}) != 1):
        raise ValueError("Expected a complete ordered MATH-500 candidate pool")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.history, args.baseline, args.output)
