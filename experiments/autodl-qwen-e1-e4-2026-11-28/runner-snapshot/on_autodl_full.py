import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import os
import runpy
import sys

root = Path(__file__).resolve().parent
script = root / "performance/run_published_code_benchmark.py"
archive = json.loads((root / "performance/trials.json").read_text())
case, level = sys.argv[1:]
prior = archive[f"vllm-{case}-c{level}"]["config"]
assert hashlib.sha256(script.read_bytes()).hexdigest() == prior["source_sha256"]
device = json.loads((root / "backend-device.json").read_text())
assert "4090" in device["device_name"] and device["bf16_supported"]
assert device["vllm"] == prior["package_version"] == "0.33.0"

benchmark = runpy.run_path(str(script), run_name="autodl_full_benchmark")
torch = benchmark["torch"]
uses_prm = case in {"E2", "E4"}
if uses_prm:
    assert str(torch.__version__) == prior["runtime"]["torch"]
    assert torch.version.cuda == prior["runtime"]["cuda_runtime"]
    assert torch.cuda.get_device_name(0) == device["device_name"]
assert version("transformers") == prior["runtime"]["transformers"]

def actual_device():
    return {
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "transformers": version("transformers"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **device,
    }

benchmark["runner"].require_h20 = actual_device
output = root / "results-autodl" / f"{case.lower()}-c{level}"
from types import SimpleNamespace
args=SimpleNamespace(
    case=case, backend="vllm", workers=int(level), sample=len(prior["sample_ids"]),
    seed=prior["seed"], context=32768, port=18420, model="published-qwen25",
    vllm_version=device["vllm"], output=output, shards=1, shard=0,
    strict_errors=prior["strict_errors"],
)
raise SystemExit(0 if benchmark["run"](args) else 1)
