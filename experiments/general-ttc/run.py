"""Run an E1–E4 experiment against a Qwen vLLM endpoint and a step scorer."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from inference import STRATEGIES, Task, solve
from prepare import write_json


opener = build_opener(ProxyHandler({}))


def request(url, body=None):
    req = Request(url) if body is None else Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with opener.open(req, timeout=240) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"{url} HTTP {error.code}: {error.read(600).decode(errors='replace')}") from error


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["category"], []).append(row)

    def metrics(items):
        return {
            "questions": len(items), "correct": sum(row["correct"] for row in items),
            "accuracy": fmean(row["correct"] for row in items),
            "invalid_answers": sum(row["prediction"] is None for row in items),
            **{key: sum(row[key] for row in items) for key in (
                "generator_calls", "input_tokens", "output_tokens", "prm_calls", "prm_input_tokens")}}
    return {
        **metrics(rows), "by_category": {name: metrics(items) for name, items in groups.items()},
        "macro_accuracy": fmean(fmean(row["correct"] for row in items) for items in groups.values()),
    }


def run(args):
    data = args.manifest.read_bytes()
    manifest = json.loads(data)
    selection = [row for row in manifest["questions"] if not args.category or row["category"] in args.category]
    questions = selection[:args.limit]
    if not questions or len({row["id"] for row in questions}) != len(questions):
        raise ValueError("Expected questions with unique IDs")
    gen_url = args.generator_url.rstrip("/")
    prm_url = args.prm_url.rstrip("/")
    models = request(gen_url + "/models")["data"]
    model_id = args.generator_model or models[0]["id"]
    if model_id not in {model["id"] for model in models}:
        raise ValueError(f"Requested generator model is unavailable: {model_id}")
    scorer = request(prm_url + "/health") if args.method != "E1" else None
    if scorer and scorer.get("status") != "ready":
        raise ValueError("Process reward model is unavailable")

    def generate(prompt, seed, max_tokens, temperature):
        response = request(gen_url + "/chat/completions", {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You solve multiple-choice reasoning problems carefully."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature, "seed": seed, "top_p": 0.9 if temperature else 1.0,
            "max_tokens": max_tokens,
        })
        return response["choices"][0]["message"]["content"], response["usage"]

    def score(question, steps):
        response = request(prm_url + "/score", {"question": question, "steps": steps})
        return response["step_scores"], response["input_tokens"]

    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "method": args.method, "outer_k": args.k if args.method in {"E2", "E4"} else 1,
        "seed": args.seed, "max_tokens": args.max_tokens, "search_steps": args.search_steps,
        "fixed_strategy": args.fixed_strategy, "generator": model_id, "scorer": scorer,
        "manifest_sha256": hashlib.sha256(data).hexdigest(),
        "source_url": manifest["source_url"], "source_sha256": manifest["source_sha256"],
        "questions": [row["id"] for row in questions], "code_revision": args.code_revision,
    }
    write_json(args.output / "config.json", config)
    rows = []
    started = perf_counter()
    for item in questions:
        task = Task(item["id"], item["category"], item["question"], tuple(item["options"]),
                    tuple(item.get("evidence", ())))
        began = perf_counter()
        row = solve(task, args.method, args.k, args.seed, args.max_tokens, args.search_steps,
                    generate, score, args.fixed_strategy)
        calls = [event["usage"] for event in row["events"] if event["kind"] == "generation"]
        scores = [event for event in row["events"] if event["kind"] == "prm"]
        row.update({
            "task_id": task.id, "category": task.category, "gold": item["answer"],
            "correct": row["prediction"] == item["answer"], "seconds": perf_counter() - began,
            "generator_calls": len(calls), "input_tokens": sum(c["prompt_tokens"] for c in calls),
            "output_tokens": sum(c["completion_tokens"] for c in calls),
            "prm_calls": len(scores), "prm_input_tokens": sum(e["input_tokens"] for e in scores),
        })
        filename = hashlib.sha256(task.id.encode()).hexdigest()[:20] + ".json"
        write_json(args.output / "questions" / filename, row)
        rows.append(row)
        print(f"{args.method} {task.id} {row['prediction']} {row['correct']} {row['seconds']:.1f}s", flush=True)
    write_json(args.output / "summary.json", {
        **summarize(rows), "elapsed_seconds": perf_counter() - started, "method": args.method,
        "outer_k": config["outer_k"], "manifest_sha256": config["manifest_sha256"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("E1", "E2", "E3", "E4"), required=True)
    parser.add_argument("--generator-url", default="http://127.0.0.1:18420/v1")
    parser.add_argument("--generator-model")
    parser.add_argument("--prm-url", default="http://127.0.0.1:18423")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--search-steps", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category", action="append")
    parser.add_argument("--fixed-strategy", choices=STRATEGIES)
    parser.add_argument("--code-revision", help="Record the Git commit of this run")
    args = parser.parse_args()
    if min(args.k, args.max_tokens, args.search_steps, args.limit or 1) < 1 or args.limit == 0:
        parser.error("Budgets and question limits must be positive")
    if args.fixed_strategy and args.method not in {"E3", "E4"}:
        parser.error("--fixed-strategy requires E3 or E4")
    run(args)


if __name__ == "__main__":
    main()