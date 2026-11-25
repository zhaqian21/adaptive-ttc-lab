"""Recover E4 candidates and the cost of their retained prefixes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for folder in ("e4-prm-selector-ablation", "llama-math500-five-answer-stop"):
    sys.path.insert(0, str(HERE.parent / folder))

from prepare_aime24 import AMO_REFERENCE_SHA, export
from prepare_costs import MODEL, UPSTREAM, capture_signature, partition
from replay_early_stop import PREFIX, prefix_agrees
from replay_selection import K, encode

AMO_SHA = "eb08f5239c0cc092e13552c31e505a7fdb8e8140aa8e227beedd62fc442cde58"
DATASET_SHA = {
    "amo": AMO_SHA,
    "aime24": "3cb34696a75837f848ea16f15a42f4130be655d5995e8cd24efcc54f4e6e1524",
}
MODELS = {
    "llama": MODEL,
    "qwen": "Qwen--Qwen2.5-7B-Instruct--a09a35458c702b33eeacc393d103063234e8bc28",
}
COST_FIELDS = ("generation_calls", "input_tokens", "output_tokens")


def measure(calls):
    return {"generation_calls": len(calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls)}


def prepare(raw, reference, source, model, *, dataset="amo", all_prefixes=False):
    shards = [raw / f"gpu-{lane}/E4" for lane in (0, 1)]
    cfgs = [json.loads((p / "config.json").read_bytes()) for p in shards]
    for cfg in cfgs:
        if ((cfg["dataset"], cfg["case"], cfg["backend"], cfg["upstream_commit"],
             cfg["dataset_sha256"], cfg["eval"]["num_iterations"])
                != (dataset, "E4", "vllm", UPSTREAM, DATASET_SHA[dataset], K)
                or Path(cfg["client"]["model_id"]).name != MODELS[model]):
            raise ValueError(f"Expected the pinned {dataset} E4 configuration")
    if any(cfgs[0][key] != cfgs[1][key] for key in ("client", "decode", "dataset_spec", "seed")):
        raise ValueError(f"{dataset} shards use different inference settings")

    signature = None
    pool = export(raw, reference, model, dataset=dataset)
    for line in pool.splitlines():
        item = json.loads(line)
        i = item["problem_id"]
        raw_path = shards[i % 2] / f"problems/{i:03d}/result.json"
        contents = raw_path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != item["raw_result_sha256"]:
            raise ValueError(f"Raw result changed during export at question {i}")
        row = json.loads(contents)
        observed = row["observation"]
        calls = observed["llm_calls"]
        prefix_cost = None
        if all_prefixes or prefix_agrees(item["candidates"]):
            if signature is None:
                signature = capture_signature(source, cfgs[0]["client"], model_folder=MODELS[model])
            steps = row["iterations"]
            signatures = [signature(row["problem"], entry) for entry in steps]
            try:
                boundaries = partition(calls, signatures, [entry["time_seconds"] for entry in steps])
            except ValueError as error:
                raise ValueError(f"{dataset} question {i}: {error}") from error
            prefix_cost = measure(calls[:boundaries[PREFIX - 1]])
        record = {
            **item, "generation_full": measure(calls), "generation_prefix": prefix_cost,
            "failed_attempts_without_usage": len(observed["failed_attempts"]),
        }
        if all_prefixes:
            record["generation_first_three"] = measure(calls[:boundaries[2]])
        yield record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=AMO_REFERENCE_SHA, required=True)
    parser.add_argument("--dataset", choices=DATASET_SHA, default="amo")
    parser.add_argument("--all-prefixes", action="store_true")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = b"".join(encode(r) for r in prepare(
        args.raw, args.reference, args.source, args.model, dataset=args.dataset, all_prefixes=args.all_prefixes
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(hashlib.sha256(payload).hexdigest())
