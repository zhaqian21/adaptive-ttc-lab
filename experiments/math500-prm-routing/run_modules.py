"""Run one of the MATH-500 PRM ablations against the resident services."""

import argparse
from pathlib import Path
import time

import routing
import run_adaptive_k as base
from sampling import TOKEN_LIMIT


def assemble(question, evaluator, roots, attempts, budget, rewards, checkpoints, arms):
    cases = []
    for i, root in enumerate(roots):
        cases.append({
            "prediction": root.prediction, "mean_reward": rewards.get(i),
            "correct": bool(evaluator._check_answers_equiv(root.prediction, question["gold_answer_extracted"])),
        })
    for arm in arms.values():
        selected = arm["selected_index"]
        candidate = cases[selected] if selected is not None else {"prediction": None, "correct": False}
        arm.update(prediction=candidate["prediction"], correct=candidate["correct"])
    return {
        "problem_id": question["problem_id"], "dataset_unique_id": question["dataset_unique_id"],
        "attempts": attempts, "calls": budget.calls, "budget_boundary": budget.boundary,
        "candidates": cases, "prm_indices": list(rewards), "route_checks": checkpoints, "policies": arms,
    }


def solve(question, args, config, tokenizer, decode):
    index = question["problem_id"]
    path = args.output / "problems" / f"{index:03d}.json"
    if path.exists():
        if base.baseline.read(path)["dataset_unique_id"] != question["dataset_unique_id"]:
            raise ValueError(f"Wrong checkpoint dataset row at {index}")
        return

    problem = question["problem"]
    evaluator = base.runner.upstream.get_evaluator("MATH")
    start = time.perf_counter()
    with base.baseline.requests.Session() as session:
        session.trust_env = False
        sample, _ = base.revision.make_sampler(problem, index, evaluator, args, config, tokenizer, decode, session)
        prm = base.revision.ResidentSelector(args.prm_port, session)

        def verify(root, _):
            scores = prm.score_reasoning(problem, prm.combine_fields(root.scoring_entry), base.revision.SYSTEM_PROMPT)
            return scores["mean_reward"]

        trace = routing.run_policy(sample, verify, args.method)
        seconds = time.perf_counter() - start
    if len(trace[3]) != len(prm.calls):
        raise ValueError("Unlogged PRM request")
    result = assemble(question, evaluator, *trace)
    result.update(inference_seconds=seconds, prm_total_seconds=sum(request["seconds"] for request in prm.calls))
    base.baseline.write(path, result)


def run(args):
    args.mode, args.selector = "adaptive", "answer_prm_sum"
    extension = {
        "method": args.method, "token_limit": TOKEN_LIMIT, "checkpoints": list(routing.CHECKPOINTS),
        "minimum_lead_in_prm_sum": routing.MIN_MARGIN, "min_support_fraction": "2/3",
    }
    return base.run(args, solve_question=solve, extension=extension,
                    extension_sources=(__file__, routing.__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=routing.METHODS, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--sample", type=int, choices=range(2, 501))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Worker count must be positive")
    args.prm_port, args.seed, args.context = 18422, 0, 32768
    args.model, args.vllm_version = "published-qwen25", "0.33.0"
    raise SystemExit(run(args))
