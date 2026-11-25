"""Recover the generation-cost boundary of each historical E4 candidate."""

import argparse
from contextlib import redirect_stdout
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


UPSTREAM = "d9ca5f71f5c7d155153e8f7decd8b1c524968775"
DATASET = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
REASONER = "Qwen--Qwen2.5-7B-Instruct--a09a35458c702b33eeacc393d103063234e8bc28"
BASELINE_SHA = "df1f70b320c724b184df1a76af105a6b55c5e01e05b79ae6b8dd51ad9c6a8e82"
CANDIDATES = 10
SAMPLES = 500
FINAL_METHODS = ("generate_with_structured", "generate")


def read_baseline(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA:
        raise ValueError("Expected the published PR #2 E4 predictions")
    rows = [json.loads(line) for line in raw.splitlines()]
    if [row["problem_id"] for row in rows] != list(range(SAMPLES)):
        raise ValueError("Wrong or incomplete baseline question IDs")
    return rows


def call_boundaries(calls, signatures, elapsed):
    """Require a unique ordered partition with the planner and final prompts."""
    plan = (calls[0]["method"], calls[0]["input_tokens"])
    keys = [(call["method"], call["input_tokens"]) for call in calls]
    if plan[0] != "generate":
        raise ValueError("Missing initial planner call")
    prefix_seconds = [0.0]
    for call in calls:
        prefix_seconds.append(prefix_seconds[-1] + call["seconds"])

    @lru_cache(maxsize=None)
    def search(iteration, start):
        if iteration == len(signatures):
            return ((),) if start == len(keys) else ()
        if start >= len(keys) or keys[start] != plan:
            return ()
        paths = []
        for pos in range(start + 1, len(keys) - 1):
            end = pos + 2
            if prefix_seconds[end] - prefix_seconds[start] > elapsed[iteration] + 0.001:
                break
            if keys[pos:pos + 2] == signatures[iteration]:
                paths.extend((end, *tail) for tail in search(iteration + 1, end))
                if len(paths) > 1:
                    return tuple(paths[:2])
        return tuple(paths)

    paths = search(0, 0)
    if len(paths) != 1:
        raise ValueError(f"Expected one independent-cost partition, found {len(paths)}")
    return paths[0]


def capture_prompts(source, config):
    git = ["git", "-C", str(source / "upstream")]
    if subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != UPSTREAM:
        raise ValueError("Unexpected upstream checkout")
    if subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Expected the clean upstream evaluation source")
    sys.path.insert(0, str(source / "upstream"))
    import torch
    from transformers import AutoTokenizer
    from BALROG.balrog.client import TransformerWrapper
    from main import DictConfig, get_prompts
    from src.universal_agent import UniversalAgent

    model = source / "assets" / "model" / REASONER
    client = TransformerWrapper(DictConfig({**config["client"], "model_id": str(model)}))
    client._initialized = True
    client.tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    dummy = client.tokenizer.encode('{"answer":"replay"}', add_special_tokens=False)
    input_lengths = []

    class CaptureOnly:
        device = torch.device("cpu")

        def generate(self, *, input_ids, **kwargs):
            input_lengths.append(int(input_ids.shape[-1]))
            return torch.cat((input_ids, input_ids.new_tensor([dummy])), dim=-1)

    client.model = CaptureOnly()
    prompts = get_prompts()

    def final_prompt_signature(problem, iteration):
        input_lengths.clear()
        agent = SimpleNamespace(client=client, plan=iteration["plan"])
        with redirect_stdout(io.StringIO()):
            UniversalAgent._extract_final_answers(agent, problem, iteration["reasoning"], prompts)
        if len(input_lengths) != len(FINAL_METHODS):
            raise ValueError("Expected both final-answer prompts")
        return list(zip(FINAL_METHODS, input_lengths))

    return final_prompt_signature


def build_pool(history, baseline, output, source):
    configs = [json.loads((history / f"gpu-{lane}/E4/config.json").read_bytes()) for lane in (0, 1)]
    for config in configs:
        if (config["case"], config["backend"], config["dataset_sha256"], config["upstream_commit"],
                config["seed"], config["eval"]["num_iterations"]) != ("E4", "vllm", DATASET, UPSTREAM, 0, CANDIDATES):
            raise ValueError("History uses a different model evaluation setup")
        if Path(config["client"]["model_id"]).name != REASONER:
            raise ValueError("Expected the original pinned Qwen tokenizer")
    if configs[0]["client"] != configs[1]["client"] or configs[0]["decode"] != configs[1]["decode"]:
        raise ValueError("Historical workers used different tokenization or decode settings")

    final_prompt_signature = capture_prompts(source, configs[0])
    reference = read_baseline(baseline)
    rows = []
    for expected in reference:
        index, lane = expected["problem_id"], expected["gpu"]
        raw = (history / f"gpu-{lane}/E4/problems/{index:03d}/result.json").read_bytes()
        row = json.loads(raw)
        iterations, obs = row["iterations"], row["observation"]
        calls, scores = obs["llm_calls"], obs["prm_calls"]
        if (index != row["problem_id"] or expected["dataset_unique_id"] != row["dataset_unique_id"]
                or (row["final_predicted_answer"], row["final_is_correct"], row["selected_iteration_idx"],
                    obs["seed"], len(calls), sum(call["output_tokens"] for call in calls)) !=
                   (expected["predicted_answer"], expected["is_correct"], expected["selected_iteration"],
                    expected["seed"], expected["llm_calls"], expected["output_tokens"])
                or [candidate["iteration"] for candidate in iterations] != list(range(1, CANDIDATES + 1))
                or len(scores) != CANDIDATES or obs["failed_attempts"] or obs["classification"] != "published_code"):
            raise ValueError(f"Historical record differs from PR #2 at question {index}")
        signatures = [final_prompt_signature(row["problem"], iteration) for iteration in iterations]
        try:
            boundaries = call_boundaries(calls, signatures, [item["time_seconds"] for item in iterations])
        except ValueError as error:
            raise ValueError(f"Question {index}: {error}") from error
        candidates = []
        for start, end, item, reward in zip((0, *boundaries[:-1]), boundaries, iterations, scores):
            period = calls[start:end]
            if item["prm_scores"]["mean_reward"] != reward["mean_reward"]:
                raise ValueError(f"Unmatched historical PRM requests at question {index}")
            candidates.append({
                "answer": item["predicted_answer"], "correct": item["is_correct"], "mean_reward": reward["mean_reward"],
                "generation_calls": len(period),
                "input_tokens": sum(call["input_tokens"] for call in period),
                "output_tokens": sum(call["output_tokens"] for call in period),
                "final_prompt_input_tokens": [call["input_tokens"] for call in period[-2:]],
            })
        rows.append({
            "problem_id": index, "dataset_unique_id": row["dataset_unique_id"],
            "selected_iteration": row["selected_iteration_idx"], "candidates": candidates,
            "historical_result_sha256": hashlib.sha256(raw).hexdigest(),
        })
        if len(rows) % 100 == 0:
            print(f"Aligned {len(rows)}/{SAMPLES} historical questions", flush=True)
    raw = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    print("Candidate pool SHA-256", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_pool(args.history, args.baseline, args.output, args.source)
