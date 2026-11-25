"""Audit candidate sharing, token admission, PRM selection, and transfer results."""

import argparse
import json
import math
import re
from pathlib import Path

from protocol import (
    BRANCHES, COUNTS, HERE, K, PLAN_SNIPPET_CHARS, PREFIX, TOKEN_LIMIT, alternative_prompt,
    branch_order, candidate_seed, chosen, compare_results, digest, prefix_agrees, select, sha,
)


def require(pred, label):
    if not pred:
        raise ValueError(label)


def audit_track(track, index, branch, prior, stopped):
    candidates, stages = track["candidates"], track["stages"]
    start = len(prior) + 1
    cap = None if branch == "prefix" else TOKEN_LIMIT
    require(math.isfinite(track["seconds"]) and track["seconds"] > 0, "Invalid latency")
    require([c["iteration"] for c in candidates] == list(range(start, start + len(candidates))), "Candidate order")
    require(all(type(c["is_correct"]) in (bool, type(None)) and len(c["plan_snippet"]) <= PLAN_SNIPPET_CHARS
                and re.fullmatch("[0-9a-f]{64}", c["completion_sha256"]) for c in candidates), "Candidate fields")
    if stopped:
        require(all(c["mean_reward"] is None for c in candidates), "Early stop was scored")
    else:
        require(all(math.isfinite(c["mean_reward"]) and 0 <= c["mean_reward"] <= 1 for c in candidates),
                "Missing or invalid PRM score")
    names = [c["iteration"] for c in candidates]
    if cap is not None and len(candidates) < K - PREFIX:
        names.append(start + len(candidates))
    if not stopped:
        names += [f"prm-{c['iteration']}" for c in candidates]
    sequences = [(name, [s for s in stages if s["stage"] == name]) for name in names]
    require([s for _, group in sequences for s in group] == stages, "Unexpected or out-of-order stages")
    running_charge = 0
    budget_limited = False
    for name, group in sequences:
        require(1 <= len(group) <= 3 and [s["attempt"] for s in group] == list(range(len(group))),
                "Unexpected retry order")
        is_generation = type(name) is int
        stopped_by_budget = is_generation and not any(c["iteration"] == name for c in candidates)
        require(group[-1]["status"] == ("budget" if stopped_by_budget else "complete")
                and all(s["status"] not in ("complete", "budget") for s in group[:-1]), "Unexpected stage outcome")
        for s in group:
            for key in (*COUNTS, "charged_tokens", "denied_tokens"):
                require(type(s[key]) is int and s[key] >= 0, f"Invalid counter {key}")
            if is_generation:
                context = alternative_prompt("", prior + candidates[:name - start]) if branch == "diverse" else ""
                require(s["seed"] == candidate_seed(index, name, s["attempt"])
                        and s["planner_context_sha256"] == digest(context), "Planner or seed mismatch")
                require(not s["prm_requests"] and s["generation_without_usage"] <= s["generation_requests"],
                        "Invalid generation counters")
                require(bool(s["first_prompt_sha256"]) == bool(s["generation_requests"]), "Missing first prompt")
                reported = s["input_tokens"] + s["output_tokens"]
                require(s["charged_tokens"] >= reported
                        and (s["generation_without_usage"] or s["charged_tokens"] == reported), "Invalid budget charge")
                running_charge += s["charged_tokens"]
                require(cap is None or running_charge <= cap, "Suffix budget exceeded")
                if s["status"] == "budget":
                    require(cap is not None and s["denied_tokens"] > 0
                            and running_charge + s["denied_tokens"] > cap, "Invalid budget rejection")
                    budget_limited = True
                else:
                    require(not s["denied_tokens"], "Generation admitted a denied request")
                if s["status"] == "complete":
                    require(s["generation_requests"] > 0, "Candidate lacked a generation request")
            else:
                require(s["seed"] is None and s["planner_context_sha256"] is None and s["first_prompt_sha256"] is None
                        and not any(s[key] for key in COUNTS if key != "prm_requests")
                        and not s["charged_tokens"] and not s["denied_tokens"] and s["prm_requests"] <= 1
                        and (s["status"] != "complete" or s["prm_requests"] == 1), "Invalid PRM stage")
    if cap is not None:
        require(track["budget_charge"] == running_charge and track["budget_limited"] == budget_limited,
                "Suffix budget summary")
    return candidates


def validate_row(row):
    index = row["problem_id"]
    stop = row["stopped_early"]
    branches = row["branches"]
    prefix = row["prefix"]["candidates"]
    require(index % 2 == row["lane"] and len(prefix) == PREFIX and stop == prefix_agrees(prefix), "Shared prefix or lane")
    require(list(row["branch_order"]) == ([] if stop else list(branch_order(index)))
            and set(row["answers"]) == set(BRANCHES) and set(branches) == (set() if stop else set(BRANCHES)),
            "Mismatched continuation branches")
    try:
        audit_track(row["prefix"], index, "prefix", [], stop)
        for branch in BRANCHES:
            if stop:
                expected = chosen(prefix, 1)
            else:
                arm = branches[branch]
                require(len(arm["candidates"]) <= K - PREFIX, "Too many suffix candidates")
                candidates = audit_track(arm, index, branch, prefix, False)
                expected = select(prefix + candidates)
            require(row["answers"][branch] == expected, "Wrong Top-2 choice")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Question {index}: {error}") from error


def audit_phase(root, name, indices):
    folder = root / name
    result = folder / "predictions.jsonl"
    summary = json.loads((folder / "summary.json").read_bytes())
    require(summary["predictions_sha256"] == sha(result), f"{name} checksum")
    rows = [json.loads(line) for line in result.read_bytes().splitlines()]
    require([r["problem_id"] for r in rows] == sorted(indices)
            and len({r["dataset_unique_id"] for r in rows}) == len(rows), f"{name} problem identities")
    for row in rows:
        validate_row(row)
    recomputed = compare_results(rows)
    require(all(summary[key] == value for key, value in recomputed.items()), f"{name} summary mismatch")
    configs = []
    for lane in (0, 1):
        archived = folder / f"gpu-{lane}_config.json"
        path = archived if archived.exists() else folder / f"gpu-{lane}/config.json"
        cfg = json.loads(path.read_bytes())
        require(cfg["phase"] == name and cfg["lane"] == lane and cfg["indices"] == [i for i in indices if i % 2 == lane],
                f"{name} GPU {lane} configuration")
        require(cfg["sources"] == {p: sha(HERE / p) for p in ("run.py", "protocol.py", "audit.py")}
                and cfg["suffix_token_limit"] == TOKEN_LIMIT, f"{name} GPU {lane} source verification")
        configs.append({k: v for k, v in cfg.items() if k not in ("lane", "indices")})
    require(configs[0] == configs[1], f"{name} differs across GPUs")
    return summary


def audit(root, smoke_only=False):
    plan = json.loads((HERE.parent / "llama-e4-online-ab/question_plan.json").read_bytes())
    summary = {"smoke": audit_phase(root, "smoke", plan["smoke"])}
    if not smoke_only:
        summary["math"] = audit_phase(root, "math", plan["formal"])
        if summary["math"]["gate_passed"]:
            summary["aime24"] = audit_phase(root, "aime24", list(range(30)))
        else:
            require(not (root / "aime24/summary.json").exists(), "AIME24 was run despite the transfer gate")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.smoke_only), ensure_ascii=False, separators=(",", ":")))
