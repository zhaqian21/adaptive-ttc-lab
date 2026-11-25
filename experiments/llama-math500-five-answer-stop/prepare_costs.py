"""Recover the committed Llama E4 generation-token ledger per candidate."""

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


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "llama-math500-answer-prm"))
from replay_selection import encode, load_e4


UPSTREAM = "d9ca5f71f5c7d155153e8f7decd8b1c524968775"
DATASET = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
MODEL = "meta-llama--Llama-3.1-8B-Instruct--359efdbb8af05b788a4ad4185215c6b8caa9052c"
FINAL_METHODS = ("generate_with_structured", "generate")


def partition(calls, signatures, elapsed):
    """Port the pinned Qwen E4 replay's exact-partition check to Llama."""
    keys = [(call["method"], call["input_tokens"]) for call in calls]
    planner = keys[0]
    if planner[0] != "generate":
        raise ValueError("Missing initial planner call")
    cumulative = [0.0]
    for call in calls:
        cumulative.append(cumulative[-1] + call["seconds"])

    @lru_cache(maxsize=None)
    def search(iteration, start):
        if iteration == len(signatures):
            return ((),) if start == len(keys) else ()
        if start >= len(keys) or keys[start] != planner:
            return ()
        matches = []
        for pos in range(start + 1, len(keys) - 1):
            end = pos + 2
            if cumulative[end] - cumulative[start] > elapsed[iteration] + 0.001:
                break
            if keys[pos:pos + 2] == signatures[iteration]:
                matches.extend((end, *tail) for tail in search(iteration + 1, end))
                if len(matches) > 1:
                    return tuple(matches[:2])
        return tuple(matches)

    paths = search(0, 0)
    if len(paths) != 1:
        raise ValueError(f"Expected a unique generation-call partition, found {len(paths)}")
    return paths[0]


def capture_signature(source, client_config):
    upstream = source / "upstream"
    git = ["git", "-C", str(upstream)]
    if (subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != UPSTREAM
            or subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"])):
        raise ValueError("Expected the pinned upstream checkout")
    sys.path.insert(0, str(upstream))
    import torch
    from transformers import AutoTokenizer
    from BALROG.balrog.client import TransformerWrapper
    from main import DictConfig, get_prompts
    from src.universal_agent import UniversalAgent

    model = source / "assets" / "model" / MODEL
    client = TransformerWrapper(DictConfig({**client_config, "model_id": str(model)}))
    client._initialized = True
    client.tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    reply = client.tokenizer.encode('{"answer":"replay"}', add_special_tokens=False)
    lengths = []

    class CaptureOnly:
        device = torch.device("cpu")

        def generate(self, *, input_ids, **kwargs):
            lengths.append(input_ids.shape[-1])
            return torch.cat((input_ids, input_ids.new_tensor([reply])), dim=-1)

    client.model = CaptureOnly()
    prompts = get_prompts()

    def check(problem, iteration):
        lengths.clear()
        agent = SimpleNamespace(client=client, plan=iteration["plan"])
        with redirect_stdout(io.StringIO()):
            UniversalAgent._extract_final_answers(agent, problem, iteration["reasoning"], prompts)
        if len(lengths) != 2:
            raise ValueError("Both final-answer calls must be present")
        return list(zip(FINAL_METHODS, lengths))

    return check


def read(path):
    return json.loads(path.read_bytes())


def build(args):
    paths = [args.history / f"formal/math/gpu-{lane}/E4" for lane in (0, 1)]
    configs = [read(path / "config.json") for path in paths]
    for config in configs:
        if ((config["case"], config["dataset"], config["backend"], config["upstream_commit"],
             config["dataset_sha256"], config["seed"], config["eval"]["num_iterations"])
                != ("E4", "math", "vllm", UPSTREAM, DATASET, 0, 10)
                or Path(config["client"]["model_id"]).name != MODEL):
            raise ValueError("Unexpected historical evaluation config")
    if any(configs[0][key] != configs[1][key] for key in ("client", "decode")):
        raise ValueError("The two evaluation shards used different model settings")

    signature = capture_signature(args.source, configs[0]["client"])
    prepared = []
    for identity, candidates, ref in load_e4(args.pool, args.baseline):
        i = identity["problem_id"]
        data = (paths[ref["lane"]] / f"problems/{i:03d}/result.json").read_bytes()
        if hashlib.sha256(data).hexdigest() != identity["historical_result_sha256"]:
            raise ValueError(f"Raw result no longer matches PR #21 for question {i}")
        row = json.loads(data)
        obs = row["observation"]
        calls = obs["llm_calls"]
        iterations = row["iterations"]
        scores = obs["prm_calls"]
        if ((row["problem_id"], row["dataset_unique_id"], len(calls), sum(c["output_tokens"] for c in calls))
                != (i, identity["dataset_unique_id"], ref["llm_calls"], ref["output_tokens"])
                or len(iterations) != 10 or len(scores) != 10
                or [c["mean_reward"] for c in candidates] != [s["mean_reward"] for s in scores]):
            raise ValueError(f"Historical audit mismatch for question {i}")

        signatures = [signature(row["problem"], it) for it in iterations]
        try:
            bounds = partition(calls, signatures, [item["time_seconds"] for item in iterations])
        except ValueError as error:
            raise ValueError(f"Question {i}: {error}") from error
        entries = []
        for k, (start, end) in enumerate(zip((0, *bounds[:-1]), bounds), start=1):
            period = calls[start:end]
            entries.append({
                "iteration": k, "generation_calls": end - start,
                "input_tokens": sum(c["input_tokens"] for c in period),
                "output_tokens": sum(c["output_tokens"] for c in period),
                "final_prompt_input_tokens": [c["input_tokens"] for c in period[-2:]],
            })
        prepared.append({**identity, "failed_attempts_without_recorded_tokens": len(obs["failed_attempts"]),
                         "candidate_costs": entries})
        if len(prepared) % 100 == 0:
            print(f"Reconstructed {len(prepared)}/500 E4 ledgers", flush=True)
    payload = b"".join(encode(row) for row in prepared)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_bytes(payload)
    print("Cost ledger SHA-256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "history", "pool", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    build(parser.parse_args())
