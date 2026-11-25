"""Recover the ordered E2 candidate pool from the native MATH-500 records."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_five import load_verified
from prepare_pool import CANDIDATES, DATASET, FINAL_METHODS, REASONER, SAMPLES, UPSTREAM


HERE = Path(__file__).resolve().parent
E2_BASELINE = HERE.parent / "math500-published-code-2026-11-24" / "results" / "E2" / "predictions.jsonl"
E2_BASELINE_SHA = "d3ece8d47d7e9b4f96f3b7d2b3b00f0979c9e23bf5e68f1ce31857dc7fce1dac"


def extract_question(shards, ref):
    index, lane = ref["problem_id"], ref["gpu"]
    raw = (shards[lane] / "problems" / f"{index:03d}" / "result.json").read_bytes()
    row = json.loads(raw)
    obs = row["observation"]
    calls, scores = obs["llm_calls"], obs["prm_calls"]
    iterations = row["iterations"]
    if ((row["problem_id"], row["dataset_unique_id"], row["selected_iteration_idx"],
         row["final_predicted_answer"], row["final_is_correct"], obs["seed"],
         len(calls), sum(call["output_tokens"] for call in calls)) !=
            (index, ref["dataset_unique_id"], ref["selected_iteration"],
             ref["predicted_answer"], ref["is_correct"], ref["seed"],
             ref["llm_calls"], ref["output_tokens"])
            or obs["classification"] != "published_code" or obs["backend"] != "vllm"
            or obs["failed_attempts"] or obs["strategy_calls"] or obs["intra_iteration_scoring"]
            or [item["iteration"] for item in iterations] != list(range(1, CANDIDATES + 1))
            or tuple(call["method"] for call in calls) != FINAL_METHODS * CANDIDATES
            or len(scores) != CANDIDATES):
        raise ValueError(f"Historical E2 record differs from PR #2 on question {index}")

    candidates = []
    for item, score, structured, freeform in zip(iterations, scores, calls[::2], calls[1::2]):
        candidate_score = dict(item["prm_scores"])
        if (candidate_score.pop("iteration") != item["iteration"]
                or candidate_score != {key: value for key, value in score.items() if key != "seconds"}):
            raise ValueError(f"Historical E2 PRM calls do not match candidate order on question {index}")
        candidates.append({
            "answer": item["predicted_answer"],
            "correct": item["is_correct"],
            "mean_reward": score["mean_reward"],
            "generation_calls": len(FINAL_METHODS),
            "input_tokens": structured["input_tokens"] + freeform["input_tokens"],
            "output_tokens": structured["output_tokens"] + freeform["output_tokens"],
        })
    return {
        "problem_id": index,
        "dataset_unique_id": row["dataset_unique_id"],
        "selected_iteration": row["selected_iteration_idx"],
        "candidates": candidates,
        "historical_result_sha256": hashlib.sha256(raw).hexdigest(),
    }


def build_pool(history, baseline, output):
    shards = [history / f"gpu-{gpu}" / "E2" for gpu in (0, 1)]
    configs = [json.loads((shard / "config.json").read_bytes()) for shard in shards]
    for gpu, config in enumerate(configs):
        expected_eval = config["eval"]
        if ((config["case"], config["backend"], config["dataset_sha256"],
             config["upstream_commit"], config["seed"], expected_eval["num_iterations"],
             expected_eval["use_prm_selection"], expected_eval["prm_selection_metric"]) !=
                ("E2", "vllm", DATASET, UPSTREAM, 0, CANDIDATES, True, "mean_reward")
                or config["sample_ids"] != list(range(gpu, SAMPLES, 2))
                or Path(config["client"]["model_id"]).name != REASONER):
            raise ValueError(f"Unexpected original E2 settings in shard {gpu}")
    for key in ("client", "decode"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Inconsistent historical E2 shard settings for {key}")

    refs = load_verified(baseline, E2_BASELINE_SHA)
    if [(row["problem_id"], row["gpu"]) for row in refs] != [(i, i % 2) for i in range(SAMPLES)]:
        raise ValueError("Expected all 500 original E2 questions in shard order")
    rows = [extract_question(shards, ref) for ref in refs]
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                      for row in rows).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(f"Archived {len(rows)} questions and {len(rows) * CANDIDATES} E2 candidates")
    print(f"Pool SHA-256 {hashlib.sha256(payload).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=E2_BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_pool(args.history, args.baseline, args.output)
