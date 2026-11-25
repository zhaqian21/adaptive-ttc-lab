"""Run a shared-prefix online E4 comparison on two H20s."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import requests

from protocol import (
    BRANCHES, COUNTS, K, PREFIX, TOKEN_LIMIT, TRIM_FIELDS, alternative_prompt, branch_order,
    candidate_seed, chosen, compare_results, digest, encode, prefix_agrees, select, sha, snippet,
)
from audit import validate_row
from online_ab import (
    CONTEXT, MODEL, PORTS, PRM_PROMPT, WORKERS, MeasuredSession, load_plan, released, start_service, wait_ready,
)


PHASES = ("smoke", "math", "aime24")
TRANSIENT = (AttributeError, UnicodeDecodeError, ValueError, requests.RequestException)


class BudgetReached(BaseException):
    """Exit upstream transport retries and stop this candidate at the budget boundary."""

    pass


class MeteredSession(MeasuredSession):
    def __init__(self, limit=None):
        super().__init__()
        self.limit = limit
        self.charged = 0

    def post(self, url, *args, **kwargs):
        if url.endswith("/score"):
            return super().post(url, *args, **kwargs)
        if not url.endswith("/v1/completions"):
            raise ValueError("Unexpected upstream request")
        payload = kwargs["json"]
        ceiling = len(payload["prompt"]) + payload["max_tokens"]
        stage = self.current
        if self.limit is not None and self.charged + ceiling > self.limit:
            stage["denied_tokens"] = ceiling
            raise BudgetReached()
        if stage["first_prompt_sha256"] is None:
            stage["first_prompt_sha256"] = hashlib.sha256(encode(payload["prompt"])).hexdigest()
        before = stage["input_tokens"] + stage["output_tokens"]
        missing = stage["generation_without_usage"]
        try:
            return super().post(url, *args, **kwargs)
        finally:
            used = stage["input_tokens"] + stage["output_tokens"] - before
            charge = ceiling if stage["generation_without_usage"] > missing else used
            if charge > ceiling:
                raise RuntimeError("vLLM usage exceeded the admitted request")
            self.charged += charge
            stage["charged_tokens"] += charge


def next_stage(session, name, attempt, seed=None, context=""):
    session.stage(name)
    session.current.update(
        attempt=attempt, seed=seed, planner_context_sha256=digest(context) if seed is not None else None,
        first_prompt_sha256=None, charged_tokens=0, denied_tokens=0, status="complete",
    )


def generate(question, iteration, previous, branch, session, settings):
    runner = released.runner
    index = question["problem_id"]
    extra = alternative_prompt("", previous) if branch == "diverse" else ""
    prompts = settings.prompts
    if extra:
        prompts = {**prompts, "planning_prompt_template": prompts["planning_prompt_template"] + extra}
    for retry in range(3):
        seed = candidate_seed(index, iteration, retry)
        next_stage(session, iteration, retry, seed, extra)
        released.THREAD.session = session
        calls = []
        client = released.TransformerWrapper(runner.upstream.DictConfig(settings.config["client"]))
        client._initialized = True
        client.tokenizer = settings.tokenizer
        client.model = released.GenerationProxy(
            SimpleNamespace(port=settings.gen_port, context=CONTEXT, model=MODEL[1]),
            settings.decode, seed, calls, session,
        )
        released.THREAD.client = released.instrument_client(client, calls)
        try:
            agent = runner.upstream.create_universal_agent(settings.config)
            core = runner.math_core.MATHCore(agent, prompts)
            core.reset()
            result = core.solve(question["problem"])
            completion = result.answer_unstructured
            answer = settings.evaluator._extract_answer_from_model_completion(completion)
            good = settings.evaluator._check_answers_equiv(answer, question["gold_answer_extracted"])
            if session.current["denied_tokens"]:
                raise BudgetReached()
            return {
                "iteration": iteration, "predicted_answer": answer, "is_correct": good,
                "answer_unstructured": completion, "completion_sha256": digest(completion),
                "reasoning_steps": result.reasoning_steps, "plan": result.plan, "plan_snippet": snippet(result.plan),
                "tools_used": result.metadata.get("tools_used", []),
                "compute_configs_used": result.metadata.get("compute_configs_used", []),
                "mean_reward": None,
            }
        except BudgetReached:
            session.current["status"] = "budget"
            return None
        except TRANSIENT as error:
            if session.current["denied_tokens"]:
                session.current["status"] = "budget"
                return None
            session.current["status"] = type(error).__name__
            print(f"retry question {index} {branch} candidate {iteration}: {type(error).__name__}",
                  file=sys.__stdout__, flush=True)
    raise RuntimeError(f"Candidate {index}/{branch}/{iteration} failed after 3 attempts")


def score(question, candidates, session, port):
    scorer = released.ResidentSelector(port)
    released.THREAD.session = session
    released.THREAD.prm_calls = []
    for candidate in candidates:
        for retry in range(3):
            next_stage(session, f"prm-{candidate['iteration']}", retry)
            try:
                result = scorer.score_reasoning(question["problem"], scorer.combine_fields(candidate), PRM_PROMPT)
                candidate["mean_reward"] = result["mean_reward"]
                break
            except TRANSIENT as error:
                session.current["status"] = type(error).__name__
        else:
            raise RuntimeError(f"PRM failed for question {question['problem_id']}, candidate {candidate['iteration']}")


def trim(rows):
    return [{key: c[key] for key in TRIM_FIELDS} for c in rows]


def answer_question(question, settings):
    index = question["problem_id"]
    prefix_start = time.monotonic()
    with MeteredSession() as session:
        prefix = [generate(question, i, [], "prefix", session, settings) for i in range(1, PREFIX + 1)]
        stop = prefix_agrees(prefix)
        if not stop:
            score(question, prefix, session, settings.prm_port)
        shared = {"candidates": trim(prefix), "stages": session.stages, "seconds": time.monotonic() - prefix_start}
    answers, branches = {}, {}
    if stop:
        answers = dict.fromkeys(BRANCHES, chosen(prefix, 1))
    else:
        for branch in branch_order(index):
            started = time.monotonic()
            with MeteredSession(TOKEN_LIMIT) as session:
                suffix = []
                for i in range(PREFIX + 1, K + 1):
                    candidate = generate(question, i, prefix + suffix, branch, session, settings)
                    if candidate is None:
                        break
                    suffix.append(candidate)
                score(question, suffix, session, settings.prm_port)
                answers[branch] = select(prefix + suffix)
                branches[branch] = {
                    "candidates": trim(suffix), "stages": session.stages, "budget_charge": session.charged,
                    "budget_limited": len(suffix) < K - PREFIX, "seconds": time.monotonic() - started,
                }
    return {
        "problem_id": index, "dataset_unique_id": question["dataset_unique_id"], "lane": index % 2,
        "stopped_early": stop, "prefix": shared, "branch_order": list(branches),
        "branches": branches, "answers": answers,
    }


def indices_for(phase):
    if phase == "aime24":
        return list(range(30))
    return load_plan()["smoke" if phase == "smoke" else "formal"]


def worker(args):
    from answer_fields import install

    runner = released.runner
    git = ["git", "-C", str(runner.ROOT / "upstream")]
    if (subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != runner.UPSTREAM_COMMIT
            or subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"])):
        raise ValueError("Expected the pinned clean upstream code")
    install()
    dataset = "aime24" if args.worker == "aime24" else "math"
    questions, spec = released.load_benchmark(SimpleNamespace(dataset=dataset, dataset_manifest=args.manifest))
    if [q["problem_id"] for q in questions] != list(range(30 if dataset == "aime24" else 500)):
        raise ValueError("Unexpected question ordering")
    cfg = released.configuration("E4")
    raw = released.AutoTokenizer.from_pretrained(cfg["client"]["model_id"], local_files_only=True)
    settings = SimpleNamespace(
        config=cfg, tokenizer=released.LockedTokenizer(raw), decode=released.effective_decode(cfg, raw),
        evaluator=runner.upstream.get_evaluator("MATH"), prompts=runner.upstream.get_prompts(),
        gen_port=args.gen_port, prm_port=args.prm_port,
    )
    indices = [i for i in indices_for(args.worker) if i % 2 == args.lane]
    config = {
        "phase": args.worker, "lane": args.lane, "indices": indices, "dataset": spec,
        "workers": WORKERS, "vllm": args.vllm_version, "model": list(MODEL), "decode": settings.decode,
        "agent": cfg["agent"], "prior_planner_sha256": digest(settings.prompts["planning_prompt_template"]),
        "sources": {name: sha(Path(__file__).with_name(name)) for name in ("run.py", "protocol.py", "audit.py")},
        "upstream_commit": runner.UPSTREAM_COMMIT, "suffix_token_limit": TOKEN_LIMIT,
    }
    problems = args.output / "problems"
    problems.mkdir(parents=True, exist_ok=True)
    path = args.output / "config.json"
    if path.exists() and released.read(path) != config:
        raise ValueError("Checkpoint options differ from the current run")
    released.write(path, config)
    files = {i: problems / f"{i:03d}.json" for i in indices}
    missing = [i for i in indices if not files[i].exists()]
    with (args.output / "upstream.log").open("a", encoding="utf-8", buffering=262144) as log:
        with redirect_stdout(log), redirect_stderr(log), patch.object(
            runner.upstream, "create_llm_client", lambda _: released.THREAD.client
        ):
            def evaluate(i):
                row = answer_question(questions[i], settings)
                validate_row(row)
                released.write(files[i], row)

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = [pool.submit(evaluate, i) for i in missing]
                for count, future in enumerate(as_completed(futures), start=len(indices) - len(missing) + 1):
                    future.result()
                    if count % 8 == 0 or count == len(indices):
                        print(f"{args.worker} GPU {args.lane}: {count}/{len(indices)}",
                              file=sys.__stdout__, flush=True)


def phase(args, devices, name, version):
    target = args.output / name
    previous = str(target / "summary.json")
    if Path(previous).is_file():
        summary = released.read(Path(previous))
        if sha(target / "predictions.jsonl") != summary["predictions_sha256"]:
            raise ValueError(f"{name} predictions changed")
        return summary
    target.mkdir(exist_ok=True)
    counts = [len(list((target / f"gpu-{lane}/problems").glob("*.json"))) for lane in (0, 1)]
    cfg = {"phase": name, "completed_before_start": counts, "started_unix": time.time()}
    released.write(args.output / "status.json", cfg)
    begun = time.monotonic()
    children = []
    with ExitStack() as stack:
        for lane, device in enumerate(devices):
            log = stack.enter_context((target / f"gpu-{lane}.log").open("a", encoding="utf-8"))
            command = [
                sys.executable, __file__, "--worker", name, "--output", str(target / f"gpu-{lane}"),
                "--manifest", str(args.manifest), "--lane", str(lane), "--gen-port", str(PORTS[lane]),
                "--prm-port", str(PORTS[lane + 2]), "--vllm-version", version,
            ]
            children.append(subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
                                             stdout=log, stderr=subprocess.STDOUT))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                for child in children:
                    child.wait()
                raise RuntimeError(f"{name} worker failed")
            time.sleep(2)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError(f"{name} worker failed")
    elapsed = time.monotonic() - begun
    rows = [released.read(path) for path in target.glob("gpu-*/problems/*.json")]
    rows.sort(key=lambda row: row["problem_id"])
    if [row["problem_id"] for row in rows] != sorted(indices_for(name)):
        raise ValueError(f"{name} did not produce the assigned question set")
    for row in rows:
        validate_row(row)
    payload = b"".join(encode(row) for row in rows)
    (target / "predictions.jsonl").write_bytes(payload)
    summary = {
        **cfg, **compare_results(rows), "wall_seconds": elapsed, "finished_unix": time.time(),
        "predictions_sha256": hashlib.sha256(payload).hexdigest(),
    }
    released.write(target / "summary.json", summary)
    print(f"{name}: verified {len(rows)} paired rows", flush=True)
    return summary


def coordinate(args):
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("The experiment needs two distinct GPUs")
    version = subprocess.check_output([
        args.engine_python, "-c", "from importlib.metadata import version; print(version('vllm'))",
    ], text=True).strip()
    if version != "0.33.0":
        raise ValueError(f"Expected vLLM 0.33.0, got {version}")
    weights = released.runner.asset_path(MODEL)
    released.write(args.output / "status.json", {"phase": "starting_services"})
    with ExitStack() as stack:
        services = []
        for lane, device in enumerate(devices):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            prm = start_service(stack, [
                sys.executable, str(Path(__file__).resolve().parents[1] / "llama-table1-825/llama_table1.py"),
                "--serve-prm", "--prm-port", str(PORTS[lane + 2]),
            ], env, args.output / f"prm-{lane}.log")
            llm = start_service(stack, [
                args.engine_python, "-m", "vllm.entrypoints.cli.main", "serve", str(weights),
                "--host", "127.0.0.1", "--port", str(PORTS[lane]), "--served-model-name", MODEL[1],
                "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--generation-config", "vllm",
                "--gpu-memory-utilization", "0.60", "--max-model-len", str(CONTEXT),
                "--enable-prefix-caching", "--max-num-seqs", "112", "--max-num-batched-tokens", "8192",
            ], {k: v for k, v in env.items() if k != "PYTHONPATH"}, args.output / f"engine-{lane}.log")
            services.extend(((prm, PORTS[lane + 2]), (llm, PORTS[lane])))
        for service, port in services:
            wait_ready(service, port)
        smoke = phase(args, devices, "smoke", version)
        if (not 0 < smoke["stops"] < smoke["samples"]
                or any(v["suffix_candidates"] == 0 for v in smoke["branches"].values())):
            raise ValueError("Smoke test did not cover both stopping and continuation")
        formal = phase(args, devices, "math", version)
        if formal["gate_passed"]:
            phase(args, devices, "aime24", version)
    released.write(args.output / "status.json", {"phase": "verified", "aime24_required": formal["gate_passed"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=PHASES)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-python")
    parser.add_argument("--lane", type=int, choices=(0, 1))
    parser.add_argument("--gen-port", type=int)
    parser.add_argument("--prm-port", type=int)
    parser.add_argument("--vllm-version")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.worker:
            worker(args)
        else:
            coordinate(args)
    except Exception as error:
        released.write(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
