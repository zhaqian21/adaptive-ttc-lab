"""Export score-and-answer-only AIME24 E4 pools from the original run directories."""

import argparse
import hashlib
import json
from pathlib import Path


REFERENCE_SHA = {
    "llama": "6d031242d56049e2c9b3b73cf58e25b58406b071220dcb41e1428b7921c605ec",
    "qwen": "fe0851e67e7fafb1baae73e7c17681b312c8ba19f7b1e5b4cefa34413ca806e0",
}


def export(raw, reference, model):
    reference_bytes = reference.read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != REFERENCE_SHA[model]:
        raise ValueError("The reference must be the archived AIME24 E4 predictions")
    refs = [json.loads(line) for line in reference_bytes.splitlines()]
    if [r["problem_id"] for r in refs] != list(range(30)) or len({r["dataset_unique_id"] for r in refs}) != 30:
        raise ValueError("Expected the full ordered AIME24 reference")

    records = {}
    for path in raw.glob("gpu-*/E4/problems/*/result.json"):
        payload = path.read_bytes()
        row = json.loads(payload)
        i = row["problem_id"]
        if i in records or not 0 <= i < len(refs):
            raise ValueError(f"Duplicate or out-of-range problem {i}")
        ref = refs[i]
        obs = row["observation"]
        lane = int(path.parents[3].name.split("-")[1])
        iterations = row["iterations"]
        selected = row["selected_iteration_idx"]
        pathway_bytes = path.with_name("pathway.json").read_bytes()
        pathway, = json.loads(pathway_bytes)
        if (ref["dataset_unique_id"] != row["dataset_unique_id"] or lane != i % 2
                or ref.get("lane", ref.get("gpu")) != lane
                or obs["seed"] != ref["seed"] or len(obs["llm_calls"]) != ref["llm_calls"]
                or sum(call["output_tokens"] for call in obs["llm_calls"]) != ref["output_tokens"]
                or obs["classification"] != "published_code" or obs["backend"] != "vllm"
                or [c["iteration"] for c in iterations] != list(range(1, 11))
                or len(obs["prm_calls"]) != 10 or len(pathway["prm_scores"]) != 10
                or pathway["problem_id"] != i or pathway["selected_iteration_idx"] != selected
                or pathway["selection_metric"] != "mean_reward"):
            raise ValueError(f"Raw execution does not match the archived run for problem {i}")

        candidates = []
        for c, call, stage in zip(iterations, obs["prm_calls"], pathway["prm_scores"]):
            result = c["prm_scores"]
            if (result["iteration"] != c["iteration"] or stage != result
                    or any(call[k] != v for k, v in result.items() if k != "iteration")
                    or c["selected_by_prm"] != (c["iteration"] == selected)):
                raise ValueError(f"Inconsistent PRM provenance for problem {i}, candidate {c['iteration']}")
            candidates.append({
                "iteration": c["iteration"], "predicted_answer": c["predicted_answer"],
                "is_correct": c["is_correct"], "mean_reward": result["mean_reward"],
            })
        c = candidates[selected - 1]
        if (selected != ref["selected_iteration"]
                or c["predicted_answer"] != ref["predicted_answer"] or c["is_correct"] is not ref["is_correct"]
                or c["predicted_answer"] != row["final_predicted_answer"] or c["is_correct"] is not row["final_is_correct"]
                or any(x["is_correct"] is True for x in candidates) != ref["candidate_oracle_correct"]):
            raise ValueError(f"Final answer does not match the archived run for problem {i}")

        records[i] = {
            "problem_id": i, "dataset_unique_id": row["dataset_unique_id"],
            "original_iteration": selected, "raw_result_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_pathway_sha256": hashlib.sha256(pathway_bytes).hexdigest(), "candidates": candidates,
        }

    if sorted(records) != list(range(len(refs))):
        raise ValueError("Both original shards must contain all 30 AIME24 questions")
    return b"".join(
        (json.dumps(records[i], ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        for i in range(len(refs))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=REFERENCE_SHA, required=True)
    parser.add_argument("--raw", type=Path, required=True, help="Directory containing the formal GPU shards")
    parser.add_argument("--reference", type=Path, required=True, help="Published E4 predictions.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = export(args.raw, args.reference, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(hashlib.sha256(payload).hexdigest())
