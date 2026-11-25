"""Grade and score each candidate using the released whole-trajectory selector."""

import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import sqlite3
import time

import requests

from prepare_sample import MODEL_CHOICES, PRM_PROMPT, digest, encode, load_manifest, read, write_json


BACKENDS = ("transformers", "vllm")
PRM_MODEL = "Qwen2.5-Math-PRM-7B"
PRM_REVISION = "0610740060112df12585d00a1c5f4624d2f59051"


def load_generation(folder, backend, manifest_path, manifest):
    path = folder / "candidates.jsonl"
    config, runtime = read(folder / "config.json"), read(folder / "runtime.json")
    tasks = [(row, i) for row in manifest["questions"] for i in range(1, MODEL_CHOICES + 1)]
    if ((config["backend"], config["manifest_sha256"], config["decode"], config["model_revision"])
            != (backend, digest(manifest_path), manifest["decode"], manifest["model_revision"])
            or config["source_sha256"] != digest(Path(__file__).with_name("generate_candidates.py"))
            or runtime["candidates_sha256"] != digest(path) or runtime["candidates"] != len(tasks)):
        raise ValueError(f"{backend}: generation setup or checksum mismatch")
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len(rows) != len(tasks):
        raise ValueError(f"{backend}: incomplete candidate pool")
    for (row, i), candidate in zip(tasks, rows):
        if ((row["problem_id"], i, row["dataset_unique_id"], row["seeds"][i - 1], row["prompt_sha256"])
                != tuple(candidate[key] for key in ("problem_id", "iteration", "dataset_unique_id", "seed", "prompt_sha256"))
                or not 0 < candidate["completion_tokens"] <= manifest["decode"]["max_tokens"]
                or candidate["finish_reason"] not in ("length", "eos")
                or not isinstance(candidate["completion"], str)):
            raise ValueError(f"{backend}: misaligned generation candidate at {row['problem_id']}/{i}")
    return rows


def request_score(session, endpoint, problem, steps):
    payload = {"problem": problem, "reasoning_steps": steps, "system_prompt": PRM_PROMPT}
    for attempt in range(3):
        try:
            response = session.post(endpoint + "/score", json=payload, timeout=(3, 180))
            if response.status_code == 422 and response.json().get("error") == "The scorer returned no step rewards":
                return 0.0
            response.raise_for_status()
            score = response.json()
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
            time.sleep(1 << attempt)
    values = score["step_rewards"]
    if (score["model"] != PRM_MODEL or score["num_steps"] != 1 or len(values) != 1
            or not math.isfinite(values[0]) or not 0 <= values[0] <= 1 or score["mean_reward"] != values[0]):
        raise ValueError("Expected one valid score from the pinned whole-trajectory PRM")
    return values[0]


def direct_fields(text):
    return {
        "plan": None, "reasoning_steps": ["Direct solution approach used."],
        "tools_used": ["direct"], "compute_configs_used": [{"strategy": "none", "param": 0}],
        "answer_unstructured": text,
    }


def check_legacy(history, combine, session, endpoint):
    matches = 0
    deltas = []
    for index in (0, 99, 199, 299, 399, 499):
        raw = read(history / f"formal/math/gpu-{index % 2}/E2/problems/{index:03d}/result.json")
        scores = []
        for entry in raw["iterations"]:
            steps = combine(entry)
            if steps != combine(direct_fields(entry["answer_unstructured"])):
                raise ValueError("Replay changed the historical E2 PRM input")
            score = request_score(session, endpoint, raw["problem"], steps)
            deltas.append(abs(score - entry["prm_scores"]["mean_reward"]))
            scores.append(score)
        winner = scores.index(max(scores)) + 1 if any(scores) else MODEL_CHOICES
        matches += winner == raw["selected_iteration_idx"]
    return {
        "questions": 6, "candidates": len(deltas), "same_selected_iterations": matches,
        "exact_score_matches": deltas.count(0.0), "max_absolute_score_delta": max(deltas),
    }


def score(args):
    from fcntl import LOCK_EX, flock

    from src.Evaluator import MATHEvaluator
    from src.help_functions.prm_selector import PRMSelector

    def combine(entry):
        return PRMSelector.combine_fields(None, entry)

    session = requests.Session()
    session.trust_env = False
    health = session.get(args.endpoint + "/health", timeout=5)
    health.raise_for_status()
    if health.json()["model"] != PRM_MODEL:
        raise ValueError("The expected PRM endpoint is unavailable")
    manifest = load_manifest(args.manifest)
    questions = {r["problem_id"]: r for r in manifest["questions"]}
    evaluator = MATHEvaluator()
    canary = check_legacy(args.history, combine, session, args.endpoint) if args.history else None
    if canary:
        print("Historical PRM canary", canary, flush=True)

    for backend in (args.backend,) if args.backend else BACKENDS:
        folder = args.generation_root / backend
        pool = load_generation(folder, backend, args.manifest, manifest)
        start = time.monotonic()
        config = {
            "manifest_sha256": digest(args.manifest), "generation_sha256": digest(folder / "candidates.jsonl"),
            "prm_model": PRM_MODEL, "prm_revision": PRM_REVISION, "prm_prompt": PRM_PROMPT,
            "source_sha256": digest(Path(__file__)),
        }
        with (folder / ".scoring.lock").open("a") as lock, sqlite3.connect(folder / "scoring.sqlite3") as db:
            flock(lock, LOCK_EX)
            db.execute("CREATE TABLE IF NOT EXISTS score (problem INTEGER, iteration INTEGER, row BLOB, "
                       "PRIMARY KEY(problem, iteration))")
            done = set(db.execute("SELECT problem, iteration FROM score"))
            settings = folder / "scorer_config.json"
            if settings.exists() and read(settings) != config:
                raise ValueError("Scoring checkpoint uses different inputs or settings")
            if not settings.exists():
                if done:
                    raise ValueError("Scoring checkpoint lacks its configuration")
                write_json(settings, config)
            if not done.issubset({(r["problem_id"], r["iteration"]) for r in pool}):
                raise ValueError("Unexpected scoring checkpoint identities")
            for candidate in pool:
                key = (candidate["problem_id"], candidate["iteration"])
                if key in done:
                    continue
                row = questions[key[0]]
                answer = evaluator._extract_answer_from_model_completion(candidate["completion"])
                with redirect_stdout(io.StringIO()):
                    verdict = evaluator._check_answers_equiv(answer, row["gold"])
                reward = request_score(session, args.endpoint, row["problem"], combine(direct_fields(candidate["completion"])))
                result = {
                    "problem_id": key[0], "iteration": key[1], "predicted_answer": answer,
                    "is_correct": verdict, "mean_reward": reward,
                }
                with db:
                    db.execute("INSERT INTO score VALUES (?, ?, ?)", (*key, encode(result)))
                done.add(key)
                if len(done) % 100 == 0 or len(done) == len(pool):
                    print(f"score {backend}: {len(done)}/{len(pool)} candidates in {time.monotonic()-start:.1f}s", flush=True)
            path = folder / "scored.jsonl"
            with path.with_suffix(".tmp").open("wb") as file:
                for payload, in db.execute("SELECT row FROM score ORDER BY problem, iteration"):
                    file.write(payload)
            path.with_suffix(".tmp").replace(path)
        write_json(folder / "scorer_runtime.json", {
            "candidates": len(done), "scored_sha256": digest(path), "seconds": time.monotonic() - start,
            "historical_prm_canary": canary,
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18422")
    parser.add_argument("--history", type=Path)
    parser.add_argument("--backend", choices=BACKENDS)
    score(parser.parse_args())
