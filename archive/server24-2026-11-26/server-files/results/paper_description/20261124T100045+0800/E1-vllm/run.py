"""Run and audit all 500 E1 questions with a single vLLM engine."""
import hashlib
import json
from pathlib import Path
import sys
import time


CASE = Path(__file__).resolve().parent
BATCH = CASE.parent
CHUNK_SIZE = 64


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, content):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    import yaml
    from transformers import AutoTokenizer
    import run_smoke as runner

    reference = yaml.safe_load((BATCH / "E1" / "config.yaml").read_text(encoding="utf-8"))
    tested = read(BATCH / "vllm-e1-pilot" / "requests.json")
    config = runner.configure("E1", reference["reproduction"]["seed"])
    assert config["reproduction"]["source_sha256"] == reference["reproduction"]["source_sha256"]
    assert config["agent"] == reference["agent"] and config["client"] == reference["client"]
    assert tested["model"] == reference["client"]["model_id"]
    assert tested["dtype"] == reference["client"]["torch_dtype"] == "bfloat16"
    tokenizer = AutoTokenizer.from_pretrained(tested["model"], local_files_only=True, trust_remote_code=False)
    canonical = runner.asset_path(runner.ASSETS[0]) / "test.jsonl"
    dataset_hash = digest(canonical)
    assert dataset_hash == reference["reproduction"]["dataset_sha256"]
    prompts = runner.upstream.get_prompts()
    dataset = runner.dataset(500)
    assert [row["problem_id"] for row in dataset] == list(range(500))
    requests, gold = [], []
    for row in dataset:
        index = row["problem_id"]
        messages = [
            {"role": "system", "content": prompts["direct_unstructured_final_answer_system_prompt"]},
            {"role": "user", "content": prompts["direct_unstructured_final_answer_user_prompt"].format(problem=row["problem"])},
        ]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        request = {"id": index, "dataset_unique_id": row["dataset_unique_id"], "prompt_token_ids": ids,
                   "seed": index + reference["reproduction"]["seed"]}
        if index < len(tested["requests"]):
            earlier = tested["requests"][index]
            assert all(request[key] == earlier[key] for key in request), index
        requests.append(request)
        gold.append({"id": index, "dataset_unique_id": row["dataset_unique_id"], "problem": row["problem"],
                     "answer": row["gold_answer_extracted"], "answer_raw": row["gold_answer_raw"]})
    max_input = max(len(row["prompt_token_ids"]) for row in requests)
    context = (max_input + tested["decode"]["max_tokens"] + 255) // 256 * 256
    inputs = {"case": "E1", "classification": "paper_description", "backend": "vllm",
              "model": tested["model"], "dtype": tested["dtype"], "decode": tested["decode"],
              "dataset_sha256": dataset_hash, "source_sha256": config["reproduction"]["source_sha256"],
              "gpu_uuid": reference["reproduction"]["runtime"]["cuda_visible_devices"],
              "chunk_size": CHUNK_SIZE, "max_num_seqs": 16, "max_model_len": context, "requests": requests}
    write(CASE / "requests.json", inputs)
    write(CASE / "gold.json", gold)
    print(f"Prepared {len(requests)} requests; longest input {max_input} tokens; engine context {context}.", flush=True)


def check_chunk(payload, requests, input_hash, context):
    assert payload["requests_sha256"] == input_hash and payload["environment"] == context
    assert len(payload["outputs"]) == len(requests)
    for request, output in zip(requests, payload["outputs"], strict=True):
        assert output["id"] == request["id"] and output["dataset_unique_id"] == request["dataset_unique_id"]
        assert output["seed"] == request["seed"] and output["input_tokens"] == len(request["prompt_token_ids"])
        assert 0 < len(output["token_ids"]) <= 1024 and output["finish_reason"] in {"stop", "length"}


def chunks(requests, chunk_size):
    for offset in range(0, len(requests), chunk_size):
        yield CASE / "chunks" / f"{offset:03d}.json", requests[offset:offset + chunk_size]


def decode():
    from importlib.metadata import version
    import os
    from vllm import LLM, SamplingParams

    config_path = CASE / "requests.json"
    config = read(config_path)
    expected_hash = digest(config_path)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == config["gpu_uuid"]
    environment = {"vllm": version("vllm"), "torch": version("torch"),
                   "transformers": version("transformers"), "gpu_uuid": config["gpu_uuid"]}
    engine = {"model": config["model"], "dtype": config["dtype"], "max_model_len": config["max_model_len"],
              "max_num_seqs": config["max_num_seqs"], "max_num_batched_tokens": 4096,
              "gpu_memory_utilization": 0.40, "tensor_parallel_size": 1, "generation_config": "vllm",
              "enable_prefix_caching": False, "trust_remote_code": False}
    plan = list(chunks(config["requests"], config["chunk_size"]))
    (CASE / "chunks").mkdir(exist_ok=True)
    completed = 0
    pending = []
    for path, requests in plan:
        if path.is_file():
            check_chunk(read(path), requests, expected_hash, environment)
            completed += len(requests)
        else:
            pending.append((path, requests))
    if not pending:
        return

    start = time.perf_counter()
    model = LLM(**engine)
    warm_params = {**config["decode"], "max_tokens": 16, "seed": 12345}
    model.generate([{"prompt_token_ids": config["requests"][0]["prompt_token_ids"]}],
                   SamplingParams(**warm_params), use_tqdm=False)
    startup_seconds = time.perf_counter() - start
    print(f"Engine ready in {startup_seconds:.2f}s; {completed}/500 checkpointed.", flush=True)
    write(CASE / "engine.json", {"environment": environment, "options": engine, "startup_seconds": startup_seconds})
    for path, requests in pending:
        inputs = [{"prompt_token_ids": row["prompt_token_ids"]} for row in requests]
        params = [SamplingParams(seed=row["seed"], **config["decode"]) for row in requests]
        started = time.perf_counter()
        replies = model.generate(inputs, params, use_tqdm=False)
        seconds = time.perf_counter() - started
        outputs = []
        for request, reply in zip(requests, replies, strict=True):
            assert list(reply.prompt_token_ids) == request["prompt_token_ids"]
            output, = reply.outputs
            outputs.append({"id": request["id"], "dataset_unique_id": request["dataset_unique_id"],
                            "seed": request["seed"], "completion": output.text.strip(),
                            "input_tokens": len(reply.prompt_token_ids), "token_ids": list(output.token_ids),
                            "finish_reason": output.finish_reason})
        payload = {"requests_sha256": expected_hash, "environment": environment, "seconds": seconds, "outputs": outputs}
        check_chunk(payload, requests, expected_hash, environment)
        write(path, payload)
        completed += len(outputs)
        print(f"Completed {completed}/500 (chunk {path.stem}, {seconds:.2f}s)", flush=True)


def score():
    import run_smoke as runner
    from src.Evaluator import MATHEvaluator

    config_path = CASE / "requests.json"
    config = read(config_path)
    expected_hash = digest(config_path)
    engine = read(CASE / "engine.json")
    canonical = runner.asset_path(runner.ASSETS[0]) / "test.jsonl"
    assert config["dataset_sha256"] == digest(canonical)
    data = [json.loads(line) for line in canonical.read_text(encoding="utf-8").splitlines()]
    official = runner.dataset(500)
    references = read(CASE / "gold.json")
    output, times = [], []
    for path, requests in chunks(config["requests"], config["chunk_size"]):
        payload = read(path)
        check_chunk(payload, requests, expected_hash, engine["environment"])
        output.extend(payload["outputs"])
        times.append(payload["seconds"])
    assert len(output) == len(references) == len(data) == len(official) == 500
    evaluator = MATHEvaluator()
    rows = []
    for index, (row, reference, original, normalized) in enumerate(zip(output, references, data, official, strict=True)):
        assert row["id"] == reference["id"] == normalized["problem_id"] == index
        assert row["dataset_unique_id"] == reference["dataset_unique_id"] == original["unique_id"]
        assert reference["problem"] == original["problem"] == normalized["problem"]
        assert reference["answer_raw"] == original["answer"] and reference["answer"] == normalized["gold_answer_extracted"]
        answer = evaluator._extract_answer_from_model_completion(row["completion"])
        correct = bool(evaluator._check_answers_equiv(answer, reference["answer"]))
        rows.append({"problem_id": index, "dataset_unique_id": row["dataset_unique_id"], "seed": row["seed"],
                     "predicted_answer": answer, "gold_answer": reference["answer"], "is_correct": correct,
                     "output_tokens": len(row["token_ids"])})
    correct = sum(row["is_correct"] for row in rows)
    output_tokens = sum(row["output_tokens"] for row in rows)
    seconds = sum(times)
    summary = {"case": "E1", "classification": "paper_description", "backend": "vllm",
               "samples": len(rows), "correct": correct, "accuracy": correct / len(rows),
               "paper_accuracy": 0.712, "delta_percentage_points": (correct - 356) / 5,
               "generation_seconds": seconds, "startup_seconds_last_attempt": engine["startup_seconds"],
               "output_tokens": output_tokens, "tokens_per_second": output_tokens / seconds,
               "requests_sha256": expected_hash, "environment": engine["environment"],
               "summary_verified": True, "pinned_dataset_checked": True, "chunk_outputs_verified": True}
    write(CASE / "scored.json", rows)
    write(CASE / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    {"prepare": prepare, "decode": decode, "score": score}[sys.argv[1]]()
