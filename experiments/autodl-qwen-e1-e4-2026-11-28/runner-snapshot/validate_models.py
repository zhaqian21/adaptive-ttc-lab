"""Validate BF16 inference and step scoring on one visible NVIDIA H20."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from fetch_assets import ASSETS, ROOT

sys.path.insert(0, str(ROOT / "upstream"))
from src.help_functions.prm_selector import PRMSelector


def asset_path(asset):
    kind, repo, revision = asset
    return ROOT / "assets" / kind / f"{repo.replace('/', '--')}--{revision}"


def require_h20():
    assert torch.cuda.device_count() == 1, "Expose only the assigned GPU with CUDA_VISIBLE_DEVICES"
    assert "H20" in torch.cuda.get_device_name(0) and torch.cuda.is_bf16_supported()
    return {
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def check_placement(model):
    floating = [p for p in model.parameters() if p.is_floating_point()]
    assert floating and all(p.dtype == torch.bfloat16 and p.device == torch.device("cuda:0") for p in floating)
    assert not model.training


def reason_once(model, tokenizer, *, use_cache=True):
    messages = [{"role": "user", "content": "What is 7 plus 5? Give the numerical result."}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to("cuda:0")
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=24, use_cache=use_cache)
    torch.cuda.synchronize()
    new = output[0, inputs["input_ids"].size(1):]
    answer = tokenizer.decode(new.tolist(), skip_special_tokens=True)
    assert new.numel() and "12" in answer, answer
    return {"answer": answer, "input_tokens": inputs["input_ids"].size(1),
            "output_tokens": new.numel(), "token_ids": new.tolist(), "seconds": time.perf_counter() - start}


def cache_probe(model, tokenizer):
    reason_once(model, tokenizer, use_cache=False)
    samples = {True: [], False: []}
    outputs = {True: [], False: []}
    for _ in range(3):
        for enabled in (True, False):
            trial = reason_once(model, tokenizer, use_cache=enabled)
            samples[enabled].append(trial["seconds"])
            outputs[enabled].append(trial["token_ids"])
    return {
        "same_output_token_ids": all(ids == outputs[True][0] for trials in outputs.values() for ids in trials),
        "median_with_cache_seconds": statistics.median(samples[True]),
        "median_without_cache_seconds": statistics.median(samples[False]),
        "scope": "One synthetic prompt, greedy decoding, three timed repetitions per setting",
    }


def prm_once(prm):
    steps = ["Adding 7 and 5 gives 12.", r"Therefore the result is \boxed{12}."]
    start = time.perf_counter()
    scores = prm.score_reasoning("What is 7 plus 5?", steps, "Please check each step.")
    torch.cuda.synchronize()
    assert scores["num_steps"] == len(steps), scores
    assert all(math.isfinite(score) and 0 <= score <= 1 for score in scores["step_rewards"]), scores
    scores["seconds"] = time.perf_counter() - start
    return scores



def prm_batch_probe(prm):
    problem = "What is 7 plus 5?"
    system = "Please check each step."
    candidates = (
        ["Adding 7 and 5 gives 12.", r"Therefore the result is \boxed{12}."],
        ["There are 7 objects to start.", "Add another group of 5.", "Counting them gives 11."],
        ["Combine 7 and 5 to get 12."],
    )
    tokenizer = prm.tokenizer
    separators = tokenizer.encode("<extra_0>")
    assert len(separators) == 1 and tokenizer.pad_token_id is not None

    def encode_steps(steps):
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": system},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ], tokenize=False, add_generation_prompt=False)
        return tokenizer.encode(prompt)

    def sequential():
        return [prm.score_reasoning(problem, steps, system)["step_rewards"] for steps in candidates]

    def batched():
        token_rows = [{"input_ids": encode_steps(steps)} for steps in candidates]
        batch_inputs = tokenizer.pad(token_rows, padding=True, return_tensors="pt").to(prm.model.device)
        with torch.inference_mode():
            logits = prm.model(**batch_inputs, use_cache=False)[0]
            step_mask = batch_inputs["input_ids"].eq(separators[0]) & batch_inputs["attention_mask"].bool()
            return prm.make_step_rewards(logits, step_mask)

    measurements = {}
    for name, score in (("sequential", sequential), ("batched", batched)):
        score()
        durations = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            rewards = score()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
        assert [len(row) for row in rewards] == [len(steps) for steps in candidates], rewards
        assert all(math.isfinite(reward) and 0 <= reward <= 1 for row in rewards for reward in row)
        means = [statistics.mean(row) for row in rewards]
        measurements[name] = {
            "median_seconds": statistics.median(durations),
            "step_rewards": rewards,
            "selected_candidate": len(means) - 1 if all(score == 0 for score in means) else means.index(max(means)),
        }
    baseline, batch = measurements["sequential"], measurements["batched"]
    return {
        "input_lengths": [len(encode_steps(steps)) for steps in candidates],
        "candidate_count": len(candidates),
        "measurements": measurements,
        "max_abs_step_reward_difference": max(
            abs(a - b)
            for expected, observed in zip(baseline["step_rewards"], batch["step_rewards"], strict=True)
            for a, b in zip(expected, observed, strict=True)
        ),
        "same_selected_candidate": baseline["selected_candidate"] == batch["selected_candidate"],
        "scope": "Three synthetic trajectories, one warm-up and three timed repetitions per method",
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=("reasoner", "prm", "both"), default="both")
    args = parser.parse_args()
    result = require_h20()
    torch.cuda.reset_peak_memory_stats()

    if args.only != "prm":
        reasoner_dir = asset_path(ASSETS[1])
        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(reasoner_dir)
        model = AutoModelForCausalLM.from_pretrained(reasoner_dir, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
        check_placement(model)
        result["reasoner_load_seconds"] = time.perf_counter() - start
        result["reasoner_only"] = reason_once(model, tokenizer)
        result["reasoner_allocated_bytes"] = torch.cuda.memory_allocated()
        result["cache_microbenchmark"] = cache_probe(model, tokenizer)
    if args.only != "reasoner":
        start = time.perf_counter()
        prm = PRMSelector(model_name=str(asset_path(ASSETS[2])), device="cuda:0")
        check_placement(prm.model)
        result["prm_load_seconds"] = time.perf_counter() - start
        result["prm"] = prm_once(prm)
        if args.only == "prm":
            try:
                result["batch_microbenchmark"] = prm_batch_probe(prm)
            except (AssertionError, RuntimeError, ValueError) as error:
                result["batch_microbenchmark_error"] = f"{type(error).__name__}: {error}"
    if args.only == "both":
        result["reasoner_with_prm_resident"] = reason_once(model, tokenizer)
        assert result["reasoner_with_prm_resident"]["answer"] == result["reasoner_only"]["answer"]
    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        print(json.dumps(result, indent=2), file=output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
