"""Check close opposite-verdict choices by scoring without batch padding."""

import argparse
import json
from pathlib import Path
from statistics import mean

from prepare_pool import PRM_REVISION, SAMPLES, STEP_MARKER, digest, read_pool, winner
from score_steps import PRMSelector, chat_tokens, score_batch, write_json


def audit(args):
    pool = read_pool(args.pool)
    paired = [json.loads(line) for line in args.paired.read_bytes().splitlines()]
    if not 0 <= args.margin <= 1 or len(paired) != SAMPLES:
        raise ValueError("Expected 500 paired records and a valid score margin")
    if args.model.name != f"Qwen--Qwen2.5-Math-PRM-7B--{PRM_REVISION}":
        raise ValueError("Expected the pinned scoring model")
    scorer = PRMSelector(str(args.model), device="cuda:0")
    step_id, = scorer.tokenizer.encode(STEP_MARKER, add_special_tokens=False)
    records = []
    original_correct = sum(row["step_mean"]["is_correct"] is True for row in paired)
    for row, saved in zip(pool, paired):
        index = row["problem_id"]
        if (index, row["dataset_unique_id"], row["historical_result_sha256"]) != (
                saved["problem_id"], saved["dataset_unique_id"], saved["historical_result_sha256"]):
            raise ValueError(f"Different trajectory set at question {index}")
        selected = saved["step_mean"]
        opponents = [r["step_mean_score"] for r in saved["candidates"]
                     if (r["is_correct"] is True) != (selected["is_correct"] is True)]
        if not opponents:
            continue
        gap = selected["score"] - max(opponents)
        if gap > args.margin:
            continue
        values = []
        for c in row["candidates"]:
            tokens = chat_tokens(scorer.tokenizer, row["problem"], c["steps"])
            rewards, = score_batch(scorer, step_id, [((index, c["iteration"]), tokens)])
            if len(rewards) != len(c["steps"]) or any(not 0 <= s <= 1 for s in rewards):
                raise ValueError(f"Invalid singleton rewards at {index}/{c['iteration']}")
            values.append(mean(rewards))
        new = winner(values)
        records.append({
            "problem_id": index,
            "opposite_verdict_margin": gap,
            "batch_selected_iteration": selected["selected_iteration"],
            "batch_correct": selected["is_correct"],
            "singleton_selected_iteration": new,
            "singleton_correct": row["candidates"][new - 1]["correct"],
            "singleton_mean_scores": values,
        })
    gain = sum(r["batch_correct"] is not True and r["singleton_correct"] is True for r in records)
    loss = sum(r["batch_correct"] is True and r["singleton_correct"] is not True for r in records)
    summary = {
        "case": pool[0]["case"], "opposite_verdict_margin": args.margin,
        "pool_sha256": digest(args.pool), "paired_sha256": digest(args.paired),
        "checked_questions": len(records), "checked_candidates": 10 * len(records),
        "changed_to_correct": gain, "changed_to_wrong": loss,
        "batch_correct": original_correct,
        "correct_after_replacing_checked_choices": original_correct + gain - loss,
        "questions": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summary)
    print({key: value for key, value in summary.items() if key != "questions"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.01)
    audit(parser.parse_args())
