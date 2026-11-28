"""Compare audited E3 and E1 on matched AutoDL questions."""
from collections import Counter
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
e1_math = root / "results-e1-formal-autodl-20261128"
e1_others = root / "results-e1-other-datasets-resume-20261128"

def read(path):
    return json.loads(path.read_text())

def predictions(path, expected_hash):
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected_hash, path
    return [json.loads(line) for line in payload.splitlines()]

def compare(root_e3, e3_results):
    if not e3_results:
        return
    previous_math = read(e1_math / "comparison.json")
    previous_others = read(e1_others / "summary.json")
    results = {}
    for name, result in e3_results.items():
        e3 = predictions(root_e3 / name / "predictions.jsonl", result["predictions_sha256"])
        after = result["autodl"]
        if name == "math500":
            before = previous_math["aggregate"]["autodl"]
            e1 = predictions(e1_math / "predictions.jsonl", previous_math["predictions_sha256"])
            before_time, estimated = before["sum_shard_seconds"], False
            assert before["total_output_tokens"] == sum(row["output_tokens"] for row in e1)
        else:
            previous = previous_others[name]
            e1 = predictions(e1_others / name / "predictions.jsonl", previous["predictions_sha256"])
            before = previous["autodl"]
            before_time, estimated = before["elapsed_seconds"], previous["elapsed_is_estimated"]
            assert before["output_tokens"] == sum(row["output_tokens"] for row in e1)
        keys = ("problem_id", "dataset_unique_id")
        assert [tuple(r[k] for k in keys) for r in e1] == [tuple(r[k] for k in keys) for r in e3], name
        assert before["correct"] == sum(row["is_correct"] is True for row in e1)
        paired = Counter((a["is_correct"] is True, b["is_correct"] is True) for a,b in zip(e1,e3,strict=True))
        e1_tokens = sum(row["output_tokens"] for row in e1)
        result = {
            "questions":len(e1),
            "e1":{"correct":before["correct"], "elapsed_seconds":before_time, "elapsed_is_estimated":estimated,
                  "output_tokens":e1_tokens, "generation_calls":before["generation_calls"]},
            "e3":{"correct":after["correct"], "elapsed_seconds":after["elapsed_seconds"],
                  "output_tokens":after["output_tokens"], "generation_calls":after["generation_calls"]},
            "accuracy_delta_percentage_points":100*(after["correct"]-before["correct"])/len(e1),
            "paired":{"both_correct":paired[True,True], "e3_only":paired[False,True],
                      "e1_only":paired[True,False], "both_incorrect":paired[False,False]},
            "wall_time_ratio":after["elapsed_seconds"]/before_time,
            "output_token_ratio":after["output_tokens"]/e1_tokens,
            "generation_call_ratio":after["generation_calls"]/before["generation_calls"],
        }
        if name == "gsm8k":
            result["historical_500_subset"] = {
                "questions":500, "e1_correct":sum(row["is_correct"] is True for row in e1[:500]),
                "e3_correct":sum(row["is_correct"] is True for row in e3[:500]),
            }
        results[name] = result

    target = root_e3 / "e3-vs-e1-4090.json"
    staged = target.with_suffix(".tmp")
    staged.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staged.replace(target)
