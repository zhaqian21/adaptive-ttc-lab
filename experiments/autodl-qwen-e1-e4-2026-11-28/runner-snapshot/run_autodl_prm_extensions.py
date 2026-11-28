"""Run E2 and E4 on the five pinned datasets."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace

root = Path(__file__).resolve().parent
base = root.parent
data = base / "datasets"
archive = base.parent / "github/experiments/aime24-amo-bench-published-code-2026-11-24"
source = archive / "scripts/run_published_extensions.py"
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash == "ef1a73afb98fc0bb59e14c7be1cc84c76461b649ed5821596ffed3de5e7c68bf"

case, name, destination = sys.argv[1:]
assert case in {"E2", "E4"}
reference = json.loads((archive / "config/formal-runs.json").read_text())
specs = json.loads((data / "manifest.json").read_text())
module = runpy.run_path(str(source), run_name="autodl_prm_extensions")
runner = module["runner"]
torch = module["torch"]
assert str(torch.__version__) == "2.6.0+cu124" and torch.version.cuda == "12.4"
assert version("transformers") == "4.48.3" and torch.cuda.is_bf16_supported()
device = json.loads((root / "backend-device.json").read_text())
assert torch.cuda.get_device_name(0) == device["device_name"] and device["vllm"] == "0.33.0"
runner.require_h20 = lambda: {
    "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
    "transformers": version("transformers"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    **device,
}

if name in reference:
    cfg = reference[name][case]["gpu-0"]["config"]
    other = reference[name][case]["gpu-1"]["config"]
    for field in ("concurrency", "seed", "strict_errors", "decode"):
        assert cfg[field] == other[field]
    workers = cfg["concurrency"]
    seed, strict = cfg["seed"], cfg["strict_errors"]
    priority = archive / f"performance/{name}-profile.json"
    prm_port = 18422
else:
    import pyarrow.parquet as pq

    spec = specs[name]
    path = data / spec["filename"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == spec["samples"] == {"amc23": 40, "aime26": 30, "gsm8k": 1319}[name]
    ordered_ids = list(range(len(rows)))
    if name == "gsm8k":
        frozen = spec["historical_sample"]
        payload = (data / frozen["filename"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == frozen["sha256"]
        plan = json.loads(payload)["questions"]
        fixed = [int(entry["dataset_unique_id"].split(":")[1]) for entry in plan]
        assert len(fixed) == len(set(fixed)) == 500
        taken = set(fixed)
        ordered_ids = fixed + [i for i in ordered_ids if i not in taken]
    elif name == "aime26":
        assert [row["problem_idx"] for row in rows] == list(range(1, 31))
    else:
        evaluator = runner.upstream.get_evaluator("MATH")
    questions = []
    manifest = []
    for i, idx in enumerate(ordered_ids):
        row = rows[idx]
        if name == "gsm8k":
            problem = row["question"]
            gold = row["answer"].rsplit("#### ", 1)[1].strip().replace(",", "")
            uid = f"test:{idx}"
            assert re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", gold)
        elif name == "aime26":
            problem, gold, uid = row["problem"], str(row["answer"]), str(row["problem_idx"])
            assert type(row["answer"]) is int and 0 <= row["answer"] <= 999
        else:
            answer = str(row["answer"]).strip()
            gold = evaluator._extract_answer_from_model_completion(r"\boxed{" + answer + "}")
            problem, uid = row["question"], str(row["id"])
            assert gold and evaluator._check_answers_equiv(gold, answer)
        assert problem.strip() and gold
        item = {
            "problem_id": i, "dataset": name, "dataset_unique_id": uid,
            "question_sha256": hashlib.sha256(problem.encode()).hexdigest(), "gold_answer_extracted": gold,
        }
        if name == "gsm8k" and i < 500:
            assert item == plan[i], i
        manifest.append(item)
        questions.append({**item, "problem": problem, "level": None})
    assert len({q["dataset_unique_id"] for q in questions}) == len(questions)
    job_dir = Path(destination) / name
    job_dir.mkdir(parents=True, exist_ok=True)
    module["write"](job_dir / "question-manifest.json", manifest)

    # The released cores for AIME and AMO inherit the same generation and grading.
    # Route the three additional datasets through MATHCore while reporting their own IDs.
    namespace = module["run"].__globals__
    original_solve = namespace["solve"]

    def solve_with_math_core(question, case, args, config, tokenizer, decode, native_model):
        math_args = SimpleNamespace(**{**vars(args), "dataset": "math"})
        return original_solve(question, case, math_args, config, tokenizer, decode, native_model)

    namespace.update(load_benchmark=lambda _args: (questions, spec), solve=solve_with_math_core)
    workers = min(len(questions), {"E2":48,"E4":24}[case])
    seed, strict, priority, prm_port = 0, False, None, None

args = SimpleNamespace(
    case=case, dataset=name, dataset_manifest=data / "manifest.json", prm_port=prm_port,
    backend="vllm", workers=workers, sample=None, seed=seed, context=32768, port=18420,
    model="published-qwen25", vllm_version=device["vllm"],
    output=Path(destination) / name / "full", shards=1, shard=0,
    priority_file=priority, strict_errors=strict,
)
ok = module["run"](args)
raise SystemExit(0 if ok else 1)
