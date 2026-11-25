"""Compare the routing modules and matched controls on one fixed candidate trace."""

import argparse
import hashlib
import json
from pathlib import Path

from answer_aggregation import group_answers, winner
from replay_modules import labelled, replay
from sampling import Prefix, TOKEN_LIMIT, select


COSTS = ("generation_input_tokens", "generation_output_tokens", "generation_calls", "prm_calls")


def load(root, method):
    folder = root / method / "export"
    manifest = json.loads((folder / "summary.json").read_bytes())
    raw = (folder / "predictions.jsonl").read_bytes()
    values = [json.loads(line) for line in raw.splitlines()]
    if (manifest["method"] != method or hashlib.sha256(raw).hexdigest() != manifest["predictions_sha256"]
            or len(values) != manifest["samples"]):
        raise ValueError(f"Wrong archived input for {method}")
    return values, manifest


def token_total(values):
    return sum(row["generation_input_tokens"] + row["generation_output_tokens"] for row in values)


def from_trace(row, method, limit=TOKEN_LIMIT):
    _, _, _, _, _, arms = replay(row, method, limit)
    return labelled(arms, row["candidates"])


def chronological_prm(row, g):
    if g["reason"] == "agreement":
        return g
    count = g["prm_calls"]
    candidates = row["candidates"][:count]
    baseline = select(candidates, Prefix(count, 0, "fixed_count"))
    groups = group_answers([r["prediction"] for r in candidates], [r["mean_reward"] for r in candidates])
    choice = winner(groups, "sum_prm", baseline)
    return labelled({"control": {**g, "selected_index": choice}}, row["candidates"])["control"]


def match_generation_budget(rows, target):
    def control(cap):
        return [from_trace(row, "factorial", cap)["adaptive_sum"] for row in rows]

    lo, hi = 1, TOKEN_LIMIT
    while lo < hi:
        middle = (lo + hi) // 2
        if token_total(control(middle)) < target:
            lo = middle + 1
        else:
            hi = middle
    possible = [(value, control(value)) for value in sorted({lo, max(1, lo - 1)})]
    return min(possible, key=lambda option: abs(token_total(option[1]) - target))


def run(root):
    source, original = load(root, "factorial")
    reports = {"factorial": original}
    for method in ("g", "f", "fg"):
        observed, manifest = load(root, method)
        if [(row["problem_id"], row["dataset_unique_id"]) for row in observed] != [
            (row["problem_id"], row["dataset_unique_id"]) for row in source
        ]:
            raise ValueError("Online stages evaluated a different set of questions")
        for actual, expected in zip(manifest["configs"], original["configs"]):
            if {k: v for k, v in actual.items() if k != "extension"} != {
                k: v for k, v in expected.items() if k != "extension"
            }:
                raise ValueError("Online stages used different settings outside their modules")
        reports[method] = manifest
    rows = []
    for row in source:
        policies = {"adaptive_sum": row["policies"]["adaptive_sum"]}
        for name in ("g", "f", "fg"):
            policies.update(from_trace(row, name))
        if policies["g"]["selected_index"] != policies["adaptive_sum"]["selected_index"]:
            raise ValueError("Pruned score groups changed the full-aggregation answer")
        if any(policies["f"][key] != policies["fg"][key] for key in ("selected_index", "calls", "reason")):
            raise ValueError("Exact score pruning altered the sampling policy")
        policies["same_prm_chronological"] = chronological_prm(row, policies["g"])
        rows.append({"problem_id": row["problem_id"], "dataset_unique_id": row["dataset_unique_id"],
                     "policies": policies})

    target = token_total([row["policies"]["f"] for row in rows])
    cap, baseline = match_generation_budget(source, target)
    for row, control in zip(rows, baseline):
        row["policies"]["same_generation_static_cap"] = control
    policies = {}
    for name in rows[0]["policies"]:
        arms = [row["policies"][name] for row in rows]
        policies[name] = {"correct": sum(arm["correct"] for arm in arms),
                          **{key: sum(arm[key] for arm in arms) for key in COSTS}}
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                      for row in rows).encode("utf-8")
    output = root / "matched"
    output.mkdir(exist_ok=True)
    report = {
        "samples": len(rows), "formal": original["formal"],
        "source_predictions_sha256": original["predictions_sha256"],
        "observed": {method: {"wall_seconds": actual["wall_seconds"], "policies": actual["policies"]}
                     for method, actual in reports.items()},
        "paired_on_factorial_trace": policies,
        "static_token_control": {"per_question_limit": cap, "target_total_tokens": target,
                                 "observed_total_tokens": token_total(baseline)},
        "predictions_sha256": hashlib.sha256(payload).hexdigest(),
    }
    (output / "predictions.jsonl").write_bytes(payload)
    (output / "summary.json").write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
