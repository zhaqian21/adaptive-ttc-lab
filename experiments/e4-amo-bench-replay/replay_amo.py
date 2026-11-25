"""Compare historical AMO-Bench E4 with answer grouping and five-answer early stopping."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e4-prm-selector-ablation"))

from replay import METHODS, from_archive, records
from replay_selection import contrast, encode
from prepare_pool import AMO_REFERENCE_SHA, AMO_SHA, COST_FIELDS


REFERENCES = {
    "llama": "llama-table1-24-2026-11-25/results/amo/E4/predictions.jsonl",
    "qwen": "aime24-amo-bench-published-code-2026-11-24/results/amo/E4/predictions.jsonl",
}
MODES = ("fixed_pool", "five_on_fixed_pool")


def replay_model(inputs, refs):
    for source, ref, row in zip(inputs, refs, from_archive(inputs, refs)):
        full = source["generation_full"]
        prefix = source["generation_prefix"]
        if (full["generation_calls"], full["output_tokens"]) != (ref["llm_calls"], ref["output_tokens"]):
            raise ValueError("Full generation cost differs from the published reference")
        if any(type(full[k]) is not int or full[k] <= 0 for k in COST_FIELDS):
            raise ValueError("Invalid archived generation cost")
        if bool(prefix) != row["five_on_fixed_pool"]["stopped_early"]:
            raise ValueError("Missing or unexpected retained-prefix ledger")
        if prefix and any(type(prefix[k]) is not int or not 0 < prefix[k] < full[k] for k in COST_FIELDS):
            raise ValueError("Invalid retained-prefix cost")
        failure_count = source["failed_attempts_without_usage"]
        if type(failure_count) is not int or failure_count < 0:
            raise ValueError("Invalid failed-attempt count")
        row["fixed_pool"]["generation"] = full
        row["five_on_fixed_pool"]["generation"] = prefix if prefix else full
        row["failed_attempts_without_usage"] = failure_count
        yield row


def summarize(rows):
    report = {"samples": len(rows), "failed_attempts_without_usage": sum(r["failed_attempts_without_usage"] for r in rows)}
    for mode in MODES:
        cases = [r[mode] for r in rows]
        cost = {name: sum(r["generation"][name] for r in cases) for name in COST_FIELDS}
        report[mode] = {
            "correct": {method: sum(r["choices"][method]["is_correct"] for r in cases) for method in METHODS},
            "early_stops": sum(r["stopped_early"] for r in cases),
            "candidate_pool_contains_correct": sum(r["candidate_oracle_correct"] for r in cases),
            "prm_requests": sum(r["logical_prm_requests"] for r in cases),
            "total_tokens": cost["input_tokens"] + cost["output_tokens"],
            **cost,
        }
    report["paired_combined_vs_e4"] = contrast([{
        "problem_id": r["problem_id"], "e4": r["fixed_pool"]["choices"]["original"],
        "combined": r["five_on_fixed_pool"]["choices"]["sum_prm"],
    } for r in rows], "e4", "combined")
    return report


def run(output):
    checksums = json.loads((HERE / "inputs/checksums.json").read_bytes())
    ordered_ids, report = None, {"dataset_sha256": AMO_SHA, "models": {}}
    output.mkdir(parents=True, exist_ok=True)
    for model, reference in REFERENCES.items():
        inputs = records(HERE / f"inputs/{model}.jsonl", checksums[model], 50)
        refs = records(HERE.parent / reference, AMO_REFERENCE_SHA[model], 50)
        ids = [r["dataset_unique_id"] for r in inputs]
        if ordered_ids is not None and ordered_ids != ids:
            raise ValueError("The two models must use the same ordered AMO-Bench problem IDs")
        ordered_ids = ids
        rows = list(replay_model(inputs, refs))
        payload = b"".join(encode(r) for r in rows)
        (output / f"{model}.jsonl").write_bytes(payload)
        report["models"][model] = {
            **summarize(rows), "candidate_input_sha256": checksums[model],
            "baseline_reference_sha256": AMO_REFERENCE_SHA[model],
            "predictions_sha256": hashlib.sha256(payload).hexdigest(),
        }
    (output / "summary.json").write_bytes((json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), separators=(",", ":")))
