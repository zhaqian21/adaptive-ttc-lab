"""Run paired revision/resampling policies on the pinned MATH-500 test set."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from revision_policy import BudgetExhausted, Candidate, POLICIES, TokenBudget, schedule

# The baseline scripts are supplied on PYTHONPATH by the reproduction command.
import run_published_code as baseline


runner = baseline.runner
SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."
REVISION_PROMPT = (
    "Review this solution to the problem. Correct any errors and their consequences, "
    "keeping correct reasoning. If no correction is needed, reproduce the solution "
    r"verbatim. Output the complete solution and put its final answer in \boxed{}. "
    "Do not add commentary about the review."
)


class StopSampleAtBudget(BaseException):
    """Unwind the upstream client, which retries every Exception."""


class BudgetedGeneration(baseline.GenerationProxy):
    def __init__(self, args, decode, seed, session, budget):
        super().__init__(args, decode, seed, [], session)
        self.budget = budget

    def generate(self, *, input_ids, max_new_tokens, **kwargs):
        prompt_tokens = input_ids.shape[-1]
        try:
            self.budget.check(prompt_tokens, max_new_tokens)
        except BudgetExhausted:
            raise StopSampleAtBudget from None
        output = super().generate(input_ids=input_ids, max_new_tokens=max_new_tokens, **kwargs)
        completion_tokens = output.shape[-1] - prompt_tokens
        self.budget.record("sample", prompt_tokens, completion_tokens)
        self.calls.append(completion_tokens)
        return output


class ResidentSelector(runner.PRMSelector):
    def __init__(self, port, session):
        self.url = f"http://127.0.0.1:{port}/score"
        self.session = session
        self.calls = []

    def score_reasoning(self, problem, reasoning_steps, system_prompt):
        start = time.perf_counter()
        response = self.session.post(self.url, json={
            "problem": problem, "reasoning_steps": reasoning_steps, "system_prompt": system_prompt,
        }, timeout=(3, 240))
        if response.status_code == 422 and response.json().get("error") == "The scorer returned no step rewards":
            scores = {"step_rewards": [], "mean_reward": 0., "min_reward": 0., "final_reward": 0., "num_steps": 0}
        else:
            response.raise_for_status()
            scores = response.json()
            if scores.pop("model") != "Qwen2.5-Math-PRM-7B":
                raise ValueError("Unexpected PRM model")
        self.calls.append({"seconds": time.perf_counter() - start, "num_steps": scores["num_steps"]})
        return scores


def solve(question, args, config, tokenizer, decode):
    path = args.output / "problems" / f'{question["problem_id"]:03d}.json'
    if path.exists():
        result = baseline.read(path)
        if result["dataset_unique_id"] != question["dataset_unique_id"]:
            raise ValueError(f"Unexpected dataset row in {path}")
        return result

    problem = question["problem"]
    index = question["problem_id"]
    evaluator = runner.upstream.get_evaluator("MATH")
    with baseline.requests.Session() as session:
        session.trust_env = False
        client = baseline.TransformerWrapper(runner.upstream.DictConfig(config["client"]))
        client._initialized = True
        client.tokenizer = tokenizer
        baseline.THREAD.client = client
        agent = runner.upstream.create_universal_agent(config)
        core = runner.math_core.MATHCore(agent, runner.upstream.get_prompts())

        def candidate(text, token_ids, entry):
            prediction = evaluator._extract_answer_from_model_completion(entry["answer_unstructured"])
            return Candidate(text, tuple(token_ids), entry, prediction)

        def sample(sequence, budget):
            # Keep independent root seeds aligned across ablations even when revisions interleave.
            seed = args.seed + index + 1_000_000_000 * sequence
            client.model = BudgetedGeneration(args, decode, seed, session, budget)
            core.reset()
            try:
                response = core.solve(problem)
            except StopSampleAtBudget:
                raise BudgetExhausted from None
            text = response.answer_unstructured
            if not text:
                return None
            entry = {
                "answer_unstructured": text, "plan": response.plan,
                "reasoning_steps": response.reasoning_steps,
                "tools_used": response.metadata.get("tools_used", []),
                "compute_configs_used": response.metadata.get("compute_configs_used") or [{}],
            }
            with tokenizer.lock:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
            return candidate(text, token_ids, entry)

        def revise(original, budget):
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": REVISION_PROMPT},
                {"role": "user", "content": f"Problem:\n{problem}\n\nCurrent solution:\n{original.text}"},
            ], tokenize=True, add_generation_prompt=True)
            max_tokens = min(args.context - len(prompt), max(decode["max_tokens"], len(original.token_ids) + 1))
            if max_tokens <= 0:
                raise ValueError("Revision prompt exceeds model context")
            budget.check(len(prompt), max_tokens)
            response = session.post(f"http://127.0.0.1:{args.port}/v1/completions", json={
                "model": args.model, "prompt": prompt, "temperature": 0.0, "max_tokens": max_tokens,
                "return_token_ids": True, "skip_special_tokens": True,
            }, timeout=(3, 480))
            response.raise_for_status()
            body = response.json()
            choice, = body["choices"]
            completion_tokens = body["usage"]["completion_tokens"]
            if choice["prompt_token_ids"] != prompt or len(choice["token_ids"]) != completion_tokens:
                raise ValueError("Mismatched vLLM revision token counts")
            budget.record("revise", len(prompt), completion_tokens)
            text = choice["text"]
            if choice["finish_reason"] != "stop" or not text:
                return None
            token_ids = choice["token_ids"]
            if token_ids and token_ids[-1] == tokenizer.eos_token_id:
                token_ids = token_ids[:-1]
            if tuple(token_ids) == original.token_ids:
                return original
            entry = {**original.scoring_entry, "reasoning_steps": [text], "answer_unstructured": text}
            revision = candidate(text, token_ids, entry)
            return revision if revision.prediction is not None else None

        initial_start = time.perf_counter()
        initial, initial_history = [], []
        shared_budget = TokenBudget(args.token_budget)
        for sequence in range(args.initial_roots):
            before = shared_budget.used
            try:
                first = sample(sequence, shared_budget)
            except BudgetExhausted:
                initial_history.append({"outcome": "budget", "generation_tokens": shared_budget.used - before})
                break
            initial_history.append({"outcome": "sampled" if first else "invalid",
                                    "generation_tokens": shared_budget.used - before})
            if first:
                initial.append(first)
        initial_seconds = time.perf_counter() - initial_start
        result = {
            "problem_id": index, "dataset_unique_id": question["dataset_unique_id"],
            "initial_predictions": [root.prediction for root in initial], "initial_actions": initial_history,
            "initial_seconds": initial_seconds, "initial_tokens": shared_budget.used, "policies": {},
        }

        if not initial:
            unresolved = {
                "prediction": None, "correct": False, "selected_root": None, "roots": 0, "candidates": [],
                "generation_input_tokens": sum(call["input_tokens"] for call in shared_budget.calls),
                "generation_output_tokens": sum(call["output_tokens"] for call in shared_budget.calls),
                "generation_calls": len(shared_budget.calls), "prm_calls": 0, "prm_request_seconds": 0.0,
                "additional_seconds": 0.0, "actions": [],
            }
            result["policies"] = {name: unresolved for name in POLICIES}
            baseline.write(path, result)
            return result

        # Gold is used by the grader after the policy and selector have finished.
        def correct(answer):
            return bool(evaluator._check_answers_equiv(answer, question["gold_answer_extracted"]))

        # Rotate execution order by row to spread warm-cache and queue position across arms.
        rotation = index % len(POLICIES)
        for name in POLICIES[rotation:] + POLICIES[:rotation]:
            started = time.perf_counter()
            budget = TokenBudget(args.token_budget, shared_budget.calls)
            roots, history = schedule(name, initial, len(initial_history), sample, revise, budget)
            scorer = ResidentSelector(args.prm_port, session)
            selected, _, scores = scorer.select_best_iteration(
                problem, [root.scoring_entry for root in roots], "mean_reward", SYSTEM_PROMPT,
            )
            for event in history:
                if event["outcome"] == "revised":
                    event["before_correct"] = correct(event["before"])
                    event["after_correct"] = correct(event["after"])
            prediction = roots[selected].prediction
            result["policies"][name] = {
                "prediction": prediction, "correct": correct(prediction),
                "selected_root": selected, "roots": len(roots),
                "candidates": [{"prediction": root.prediction, "correct": correct(root.prediction),
                                "mean_reward": score["mean_reward"]} for root, score in zip(roots, scores)],
                "generation_input_tokens": sum(call["input_tokens"] for call in budget.calls),
                "generation_output_tokens": sum(call["output_tokens"] for call in budget.calls),
                "generation_calls": len(budget.calls), "prm_calls": len(scorer.calls),
                "prm_request_seconds": sum(call["seconds"] for call in scorer.calls),
                "additional_seconds": time.perf_counter() - started,
                "actions": history,
            }
    baseline.write(path, result)
    return result


def run(args, *, load_benchmark=None):
    repo = runner.ROOT / "upstream"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != runner.UPSTREAM_COMMIT or subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
    ):
        raise ValueError("Expected an unmodified checkout of the pinned upstream commit")
    args.case, args.priority_file = "E4", None
    if load_benchmark is None:
        dataset = runner.asset_path(runner.ASSETS[0]) / "test.jsonl"
        if hashlib.sha256(dataset.read_bytes()).hexdigest() != baseline.DATASET_SHA256:
            raise ValueError("Unexpected MATH-500 data digest")
        questions = baseline.dataset_for_run(args)
        dataset_config = {"dataset_sha256": baseline.DATASET_SHA256}
    else:
        questions, dataset_config = load_benchmark(args)
    config = baseline.configuration("E4")
    raw_tokenizer = baseline.AutoTokenizer.from_pretrained(config["client"]["model_id"], local_files_only=True)
    decode = baseline.effective_decode(config, raw_tokenizer)
    tokenizer = baseline.LockedTokenizer(raw_tokenizer)
    questions = sorted(questions, key=lambda q: q["problem_id"])
    if args.only_row is not None:
        selected = set(args.only_row)
        questions = [q for q in questions if q["problem_id"] in selected]
        if {q["problem_id"] for q in questions} != selected:
            raise ValueError("Requested retry IDs do not match this shard")
    if not questions:
        raise ValueError("No questions in this shard")
    setup = {
        "baseline": "zhaqian21/adaptive-ttc-lab#2", "upstream_commit": commit,
        "source_sha256": {
            Path(name).name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (__file__, baseline.__file__, Path(__file__).with_name("revision_policy.py"))
        },
        **dataset_config,
        "assets": [list(asset) for asset in runner.ASSETS],
        "vllm_version": args.vllm_version, "selectors": list(POLICIES), "sample_decode": decode,
        "revision_prompt": REVISION_PROMPT, "revision_temperature": 0.0, "prm_metric": "mean_reward",
        "token_budget": args.token_budget, "initial_roots": args.initial_roots, "seed": args.seed,
        "workers": args.workers, "shard": args.shard, "shards": args.shards,
        "question_ids": [q["problem_id"] for q in questions],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "problems").mkdir(exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and baseline.read(config_path) != setup:
        raise ValueError("Checkpoint configuration differs from this run")
    baseline.write(config_path, setup)

    start = time.perf_counter()
    failures = {}
    completed = 0
    with (args.output / "details.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(log), patch.object(runner.upstream, "create_llm_client", lambda _: baseline.THREAD.client):
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(solve, q, args, config, tokenizer, decode): q["problem_id"] for q in questions}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        traceback.print_exc(file=log)
                        failures[futures[future]] = f"{type(error).__name__}: {error}"
                    else:
                        completed += 1
                    if (completed + len(failures)) % 8 == 0 or completed + len(failures) == len(questions):
                        print(f"shard {args.shard}: {completed}/{len(questions)} complete, {len(failures)} failed",
                              file=sys.__stdout__, flush=True)
    baseline.write(args.output / "execution.json", {
        "seconds": time.perf_counter() - start, "complete": completed, "requested": len(questions), "failures": failures,
    })
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--prm-port", type=int, default=18422)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--model", default="published-qwen25")
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token-budget", type=int, default=125000)
    parser.add_argument("--initial-roots", type=int, default=3)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--only-row", type=int, action="append", help="Run only these row IDs from the shard")
    args = parser.parse_args()
    if (min(args.workers, args.token_budget, args.initial_roots) <= 0
            or not 0 <= args.shard < args.shards
            or args.sample is not None and not 1 <= args.sample <= 500):
        parser.error("Invalid worker, budget, initial root, shard, or sample setting")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
