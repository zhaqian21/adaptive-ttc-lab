"""Run fixed-K and early-stop Llama E4 online under the same dual-H20 setup."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import requests


HERE = Path(__file__).resolve().parent
for name in ("llama-table1-825", "llama-math500-answer-prm", "llama-math500-five-answer-stop"):
    sys.path.insert(0, str(HERE.parent / name))

from llama_table1 import CONTEXT, MODEL, released, start_service, wait_ready
from replay_early_stop import prefix_agrees
from replay_selection import E4_BASELINE_SHA, K, chosen, encode, selectors


PRM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."
WORKERS = 24
PORTS = (22500, 22501, 22502, 22503)
PHASES = {"smoke": True, "fixed": False, "early": True}
COST = ("generation_requests", "input_tokens", "output_tokens", "prm_requests", "generation_without_usage")
STUDY = os.environ.get("TTC_E4_STUDY", "llama-math500")
PLAN_PATH = HERE / "question_plan.json"
SAMPLES, SMOKE_SAMPLES = 500, 20
HISTORY_SHA = E4_BASELINE_SHA

if STUDY != "llama-math500":
    from followup_benchmarks import FOLLOWUPS, load_followup_benchmark

    study = FOLLOWUPS[STUDY]
    MODEL = study["model"]
    PLAN_PATH = HERE / study["plan"]
    SAMPLES, SMOKE_SAMPLES = study["samples"], study["smoke_samples"]
    HISTORY_SHA = study["history_sha256"]
    datasets, _, prm = released.runner.ASSETS
    released.runner.ASSETS = (datasets, MODEL, prm)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MeasuredSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.trust_env = False
        self.stages = []
        self.current = None

    def stage(self, name):
        self.current = {"stage": name, **dict.fromkeys(COST, 0)}
        self.stages.append(self.current)

    def post(self, url, *args, **kwargs):
        costs = self.current
        if url.endswith("/score"):
            costs["prm_requests"] += 1
            return super().post(url, *args, **kwargs)
        if not url.endswith("/v1/completions"):
            raise ValueError("Unexpected upstream request")
        costs["generation_requests"] += 1
        costs["generation_without_usage"] += 1
        response = super().post(url, *args, **kwargs)
        if response.ok:
            usage = response.json()["usage"]
            costs["input_tokens"] += usage["prompt_tokens"]
            costs["output_tokens"] += usage["completion_tokens"]
            costs["generation_without_usage"] -= 1
        return response


def solve(question, args, config, tokenizer, decode, evaluator, prompts, *, model=None, retain_traces=False,
          candidate_limit=K, agreement_at=5, agreement_rule=prefix_agrees, original_prm_only=False):
    runner = released.runner
    model = MODEL if model is None else model
    start = time.monotonic()
    index = question["problem_id"]
    outcomes = []
    with MeasuredSession() as session:
        for retry in range(3):
            seed = index + 1_000_000 * retry
            prior = len(session.stages)
            calls = []
            released.THREAD.session = session
            released.THREAD.prm_calls = []
            client = released.TransformerWrapper(runner.upstream.DictConfig(config["client"]))
            client._initialized = True
            client.tokenizer = tokenizer
            client.model = released.GenerationProxy(
                SimpleNamespace(port=args.gen_port, context=CONTEXT, model=model[1]), decode, seed, calls, session)
            released.THREAD.client = released.instrument_client(client, calls)
            agent = runner.upstream.create_universal_agent(config)
            core = runner.math_core.MATHCore(agent, prompts)
            candidates = []
            stopped = False
            try:
                for i in range(1, candidate_limit + 1):
                    session.stage(i)
                    core.reset()
                    result = core.solve(question["problem"])
                    completion = result.answer_unstructured
                    answer = evaluator._extract_answer_from_model_completion(completion)
                    correct = evaluator._check_answers_equiv(answer, question["gold_answer_extracted"])
                    candidates.append({
                        "iteration": i, "predicted_answer": answer, "is_correct": correct,
                        "answer_unstructured": completion,
                        "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                        "reasoning_steps": result.reasoning_steps, "plan": result.plan,
                        "tools_used": result.metadata.get("tools_used", []),
                        "compute_configs_used": result.metadata.get("compute_configs_used", []),
                        "mean_reward": None,
                    })
                    if PHASES[args.worker] and i == agreement_at and agreement_rule(candidates):
                        stopped = True
                        break

                selections = {"sum_prm": 1}
                if not stopped:
                    session.stage("prm")
                    selector = released.ResidentSelector(args.prm_port)
                    old_index, _, scores = selector.select_best_iteration(
                        problem=question["problem"], iteration_results=candidates,
                        selection_metric="mean_reward", system_prompt=PRM_PROMPT)
                    for row, value in zip(candidates, scores):
                        row["mean_reward"] = value["mean_reward"]
                        if retain_traces:
                            row["prm_input"] = selector.combine_fields(row)
                            row["prm_score"] = value
                    if original_prm_only:
                        selected = old_index + 1
                        selections = {"sum_prm": selected, "original": selected}
                    else:
                        selections, _ = selectors(candidates)
                        if old_index + 1 != selections["original"]:
                            raise ValueError("Original PRM choice disagrees with the archived selector")
                successful = session.stages[prior:]
                valid_stages(successful, stopped, candidate_limit, agreement_at)
                outcomes.append({"seed": seed, "error": None, "stages": successful})
                return {
                    "problem_id": index, "dataset_unique_id": question["dataset_unique_id"], "level": question["level"],
                    "lane": args.lane, "policy": args.worker, "seed": seed, "stopped_early": stopped,
                    "candidates": candidates if retain_traces else [{key: row[key] for key in (
                        "iteration", "predicted_answer", "is_correct", "mean_reward", "completion_sha256")}
                        for row in candidates],
                    "answer": chosen(candidates, selections["sum_prm"]),
                    "original_choice": None if stopped else chosen(candidates, selections["original"]),
                    "attempts": outcomes, "seconds": time.monotonic() - start,
                    **{key: sum(stage[key] for stage in session.stages) for key in COST},
                }
            except (AttributeError, UnicodeDecodeError, ValueError, requests.RequestException) as error:
                outcomes.append({"seed": seed, "error": type(error).__name__, "stages": session.stages[prior:]})
                print(f"retry {args.worker} problem {index}: {type(error).__name__}", file=sys.__stdout__, flush=True)
    raise RuntimeError(f"{args.worker}/{index} failed after 3 attempts: {outcomes[-1]['error']}")


def valid_stages(stages, stopped, candidate_limit=K, agreement_at=5):
    expected = list(range(1, (agreement_at if stopped else candidate_limit) + 1))
    if not stopped:
        expected.append("prm")
    if ([stage["stage"] for stage in stages] != expected
            or sum(stage["prm_requests"] for stage in stages) != (0 if stopped else candidate_limit)
            or any(stage["generation_requests"] < 1 for stage in stages if stage["stage"] != "prm")
            or any(stage["generation_requests"] for stage in stages if stage["stage"] == "prm")):
        raise ValueError("The successful attempt made unexpected requests")


def load_plan():
    plan = released.read(PLAN_PATH)
    if plan["historical"] != HISTORY_SHA or sorted(plan["formal"]) != list(range(SAMPLES)):
        raise ValueError("Expected the frozen historical question plan")
    if (len(plan["smoke"]) != SMOKE_SAMPLES or len(set(plan["smoke"])) != SMOKE_SAMPLES
            or not set(plan["smoke"]).issubset(plan["formal"])):
        raise ValueError(f"Smoke plan must contain {SMOKE_SAMPLES} of the {SAMPLES} questions")
    return plan


def load_questions():
    if STUDY != "llama-math500":
        return load_followup_benchmark(STUDY, released)
    return released.load_benchmark(SimpleNamespace(dataset="math"))


def worker(args):
    from answer_fields import install
    from unittest.mock import patch

    runner = released.runner
    root = runner.ROOT / "upstream"
    git = ["git", "-C", str(root)]
    if (subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != runner.UPSTREAM_COMMIT
            or subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"])):
        raise ValueError("Expected the pinned clean upstream checkout")
    install()
    questions, spec = load_questions()
    if len(questions) != SAMPLES or [row["problem_id"] for row in questions] != list(range(SAMPLES)):
        raise ValueError("The dataset and frozen question plan do not match")
    plan = load_plan()
    indices = plan["smoke" if args.worker == "smoke" else "formal"]
    indices = [i for i in indices if i % 2 == args.lane]
    config = released.configuration("E4")
    raw = released.AutoTokenizer.from_pretrained(config["client"]["model_id"], local_files_only=True)
    tokenizer = released.LockedTokenizer(raw)
    decode = released.effective_decode(config, raw)
    evaluator = runner.upstream.get_evaluator("MATH")
    prompts = runner.upstream.get_prompts()
    args.output.mkdir(parents=True, exist_ok=True)
    problem_dir = args.output / "problems"
    problem_dir.mkdir(exist_ok=True)
    paths = {i: problem_dir / f"{i:03d}.json" for i in indices}
    setup = {
        "phase": args.worker, "lane": args.lane, "submission_order": indices,
        "workers": WORKERS, "dataset": spec, "decode": decode, "vllm": args.vllm_version,
        "model": list(MODEL), "agent": config["agent"],
        "upstream_commit": runner.UPSTREAM_COMMIT,
        "source_sha256": fingerprint(Path(__file__)), "plan_sha256": fingerprint(PLAN_PATH),
        "study": STUDY,
    }
    settings = args.output / "config.json"
    if settings.exists() and released.read(settings) != setup:
        raise ValueError("Resume requires identical scoring settings")
    released.write(settings, setup)
    missing = [i for i in indices if not paths[i].is_file()]
    with (args.output / "upstream.log").open("a", encoding="utf-8", buffering=262144) as log:
        with redirect_stdout(log), redirect_stderr(log), patch.object(
                runner.upstream, "create_llm_client", lambda _: released.THREAD.client):
            def evaluate(i):
                row = solve(questions[i], args, config, tokenizer, decode, evaluator, prompts)
                released.write(paths[i], row)

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = [pool.submit(evaluate, i) for i in missing]
                for count, future in enumerate(as_completed(futures), start=len(indices)-len(missing)+1):
                    future.result()
                    if count % 8 == 0 or count == len(indices):
                        print(f"{args.worker} GPU {args.lane}: {count}/{len(indices)} complete",
                              file=sys.__stdout__, flush=True)
    ordered = [released.read(paths[i]) for i in sorted(indices)]
    payload = b"".join(encode(row) for row in ordered)
    (args.output / "predictions.jsonl").write_bytes(payload)
    released.write(args.output / "worker_summary.json", {
        "samples": len(ordered), "predictions_sha256": hashlib.sha256(payload).hexdigest(),
    })


def phase(args, devices, name, version):
    target = args.output / name
    target.mkdir(exist_ok=True)
    started = time.monotonic()
    completed = [len(list((target / f"gpu-{lane}" / "problems").glob("*.json"))) for lane in (0, 1)]
    cfg = {"phase": name, "completed_before_start": completed, "started_unix": time.time()}
    released.write(args.output / "status.json", cfg)
    children, files = [], []
    with ExitStack() as stack:
        for lane, device in enumerate(devices):
            output = target / f"gpu-{lane}"
            log = stack.enter_context((target / f"gpu-{lane}.log").open("a", encoding="utf-8"))
            command = [
                sys.executable, str(Path(__file__)), "--worker", name, "--output", str(output),
                "--lane", str(lane), "--gen-port", str(PORTS[lane]), "--prm-port", str(PORTS[lane + 2]),
                "--vllm-version", version,
            ]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
                                     stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            files.append(output / "predictions.jsonl")
        for child in children:
            if child.wait() != 0:
                for other in children:
                    if other.poll() is None:
                        other.terminate()
                raise RuntimeError(f"{name} worker failed")
    wall = time.monotonic() - started
    rows = [json.loads(line) for file in files for line in file.read_bytes().splitlines()]
    rows.sort(key=lambda row: row["problem_id"])
    ids = load_plan()["smoke" if name == "smoke" else "formal"]
    if [row["problem_id"] for row in rows] != sorted(ids):
        raise ValueError(f"{name}: expected a complete unique question set")
    for row in rows:
        successful = row["attempts"][-1]
        valid_stages(successful["stages"], row["stopped_early"])
        if row["policy"] != name or row["lane"] != row["problem_id"] % 2 or successful["error"]:
            raise ValueError(f"{name}: wrong shard, policy, or verdict")
        if row["stopped_early"] != (PHASES[name] and prefix_agrees(row["candidates"])):
            raise ValueError(f"{name}: unexpected agreement route")
    (target / "predictions.jsonl").write_bytes(b"".join(encode(row) for row in rows))
    released.write(target / "summary.json", {
        **cfg, "samples": len(rows), "correct": sum(row["answer"]["is_correct"] for row in rows),
        "early_stops": sum(row["stopped_early"] for row in rows),
        "wall_seconds": wall, "finished_unix": time.time(),
        "predictions_sha256": fingerprint(target / "predictions.jsonl"),
        **{key: sum(row[key] for row in rows) for key in COST},
    })
    print(f"{name}: verified {len(rows)} rows (wall {wall:.1f}s)", flush=True)
    return rows


def coordinate(args):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 2 or not all(devices) or len(set(devices)) != 2:
        raise ValueError("The coordinator requires two distinct GPUs")
    version = subprocess.check_output([
        args.engine_python, "-c", "from importlib.metadata import version; print(version('vllm'))",
    ], text=True).strip()
    if version != "0.33.0":
        raise ValueError(f"Expected vLLM 0.33.0, got {version}")
    args.output.mkdir(parents=True, exist_ok=True)
    model = released.runner.asset_path(MODEL)
    for name in PHASES:
        target = args.output / name
        if (target / "summary.json").is_file():
            summary = released.read(target / "summary.json")
            if summary["predictions_sha256"] != fingerprint(target / "predictions.jsonl"):
                raise ValueError(f"{name}: completed phase changed after export")
            continue
        released.write(args.output / "status.json", {"phase": "starting_services", "next_phase": name})
        with ExitStack() as stack:
            processes = []
            for lane, device in enumerate(devices):
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
                prm = start_service(stack, [
                    sys.executable, str(HERE.parent / "llama-table1-825" / "llama_table1.py"),
                    "--serve-prm", "--prm-port", str(PORTS[lane + 2]),
                ], env, args.output / f"prm-{lane}.log")
                llm = start_service(stack, [
                    args.engine_python, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
                    "--host", "127.0.0.1", "--port", str(PORTS[lane]), "--served-model-name", MODEL[1],
                    "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--generation-config", "vllm",
                    "--gpu-memory-utilization", "0.60", "--max-model-len", str(CONTEXT),
                    "--enable-prefix-caching", "--max-num-seqs", "112", "--max-num-batched-tokens", "8192",
                ], {k: v for k, v in env.items() if k != "PYTHONPATH"}, args.output / f"engine-{lane}.log")
                processes.extend(((prm, PORTS[lane + 2]), (llm, PORTS[lane])))
            for child, port in processes:
                wait_ready(child, port)
            phase(args, devices, name, version)
    released.write(args.output / "status.json", {"phase": "verified"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=PHASES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--engine-python")
    parser.add_argument("--lane", type=int, choices=(0, 1))
    parser.add_argument("--gen-port", type=int)
    parser.add_argument("--prm-port", type=int)
    parser.add_argument("--vllm-version")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if not args.engine_python:
            parser.error("The coordinator requires --engine-python")
        try:
            coordinate(args)
        except Exception as error:
            released.write(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
            raise


if __name__ == "__main__":
    main()
