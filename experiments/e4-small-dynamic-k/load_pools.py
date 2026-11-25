"""Load the audited E4 candidates and exact three-, five-, and ten-candidate costs."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for name in ("llama-e4-online-ab", "llama-math500-answer-prm", "llama-math500-five-answer-stop",
             "e4-prm-selector-ablation", "e4-amo-bench-replay"):
    sys.path.insert(0, str(EXPERIMENTS / name))

from compare_online import verify_row
from prepare_aime24 import AMO_REFERENCE_SHA, REFERENCE_SHA
from replay import records
from replay_early_stop import LEDGER_SHA, PREFIX, load_e4, prefix_agrees
from replay_selection import K, answer_key, chosen, selectors, verify_bytes
from replay_amo import REFERENCES as AMO_REFERENCES


QWEN_MATH_POOL_SHA = "c1ece04f0e8d727a00bba995ab1320a849af25a7368608c0e3a64befca517373"
SMALL_K = 3
AIME_REFERENCES = {
    "llama": "llama-table1-24-2026-11-25/results/aime24/E4/predictions.jsonl",
    "qwen": "aime24-amo-bench-published-code-2026-11-24/results/aime24/E4/predictions.jsonl",
}


def json_rows(payload):
    return [json.loads(line) for line in payload.splitlines()]


def tokens(entries):
    return sum(e["input_tokens"] + e["output_tokens"] for e in entries)


def make_row(row, candidates, original_iteration, three_tokens, five_tokens, ten_tokens):
    if not 0 < three_tokens < five_tokens < ten_tokens:
        raise ValueError("Unexpected token ledger")
    selection, _ = selectors(candidates)
    if selection["original"] != original_iteration:
        raise ValueError("The complete pool does not reproduce the raw PRM selector")
    return {"problem_id": row["problem_id"], "dataset_unique_id": row["dataset_unique_id"],
            "candidates": candidates, "original_iteration": original_iteration,
            "tokens3": three_tokens, "tokens5": five_tokens, "tokens10": ten_tokens}


def llama_math():
    ledger = EXPERIMENTS / "llama-math500-five-answer-stop/results/e4_cost_ledger.jsonl"
    costs = json_rows(verify_bytes(ledger, LEDGER_SHA))
    pool = load_e4()
    if len(costs) != len(pool):
        raise ValueError("Inconsistent archived sample count")
    for (identity, candidates, ref), row in zip(pool, costs):
        calls = row["candidate_costs"]
        if (any(row[k] != identity[k] for k in identity) or [r["iteration"] for r in calls] != list(range(1, K + 1))
                or sum(r["output_tokens"] for r in calls) != ref["output_tokens"]):
            raise ValueError("Wrong Llama generation ledger")
        yield make_row(identity, candidates, ref["selected_iteration"],
                       tokens(calls[:SMALL_K]), tokens(calls[:PREFIX]), tokens(calls))


def qwen_math(pool):
    for row in records(pool, QWEN_MATH_POOL_SHA, 500):
        calls = row["candidates"]
        candidates = [
            {"iteration": i, "predicted_answer": c["answer"], "mean_reward": c["mean_reward"], "is_correct": c["correct"]}
            for i, c in enumerate(calls, 1)
        ]
        yield make_row(row, candidates, row["selected_iteration"],
                       tokens(calls[:SMALL_K]), tokens(calls[:PREFIX]), tokens(calls))


def amc(model):
    spec = json.loads((EXPERIMENTS / "e4-prm-selector-ablation/sources.json").read_bytes())
    expected = spec["online"][model + "_amc23"]["fixed_sha256"]
    path = EXPERIMENTS / "llama-e4-online-ab/results" / f"{model}-amc23/fixed/predictions.jsonl"
    for row in records(path, expected, 40):
        verify_row(row, "fixed")
        if len(row["attempts"]) != 1 or row["generation_without_usage"]:
            raise ValueError("Unexpected AMC23 interruptions or missing usage")
        calls = row["attempts"][0]["stages"]
        yield make_row(row, row["candidates"], row["original_choice"]["selected_iteration"],
                       tokens(calls[:SMALL_K]), tokens(calls[:PREFIX]), tokens(calls))


def historical(model, dataset, checksums):
    name = f"{model}_{dataset}"
    references, digests, samples = ((AIME_REFERENCES, REFERENCE_SHA, 30) if dataset == "aime24"
                                     else (AMO_REFERENCES, AMO_REFERENCE_SHA, 50))
    rows = records(HERE / "inputs" / f"{name}.jsonl", checksums[name], samples)
    refs = records(EXPERIMENTS / references[model], digests[model], samples)
    for row, ref in zip(rows, refs):
        full, five = row["generation_full"], row["generation_prefix"]
        three = row["generation_first_three"]
        selection = row["original_iteration"]
        if ((row["dataset_unique_id"], selection, full["generation_calls"], full["output_tokens"])
                != (ref["dataset_unique_id"], ref["selected_iteration"], ref["llm_calls"], ref["output_tokens"])
                or chosen(row["candidates"], selection)["is_correct"] != (ref["is_correct"] is True)
                or row["failed_attempts_without_usage"]):
            raise ValueError("Historical E4 archive mismatch")
        yield make_row(row, row["candidates"], selection, three["input_tokens"] + three["output_tokens"],
                       five["input_tokens"] + five["output_tokens"],
                       full["input_tokens"] + full["output_tokens"])


def load(qwen_pool):
    checksums = json.loads((HERE / "inputs/checksums.json").read_bytes())
    return {
        "llama_math": list(llama_math()), "qwen_math": list(qwen_math(qwen_pool)),
        "llama_amc23": list(amc("llama")), "qwen_amc23": list(amc("qwen")),
        **{f"{model}_{dataset}": list(historical(model, dataset, checksums))
           for dataset in ("aime24", "amo") for model in ("llama", "qwen")},
    }
