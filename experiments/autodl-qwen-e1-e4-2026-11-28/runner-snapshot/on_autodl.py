import hashlib
import json
from pathlib import Path
import runpy
import sys

root=Path(__file__).resolve().parent
script=root/"performance"/"run_published_code_benchmark.py"
historical=json.loads((root/"performance"/"trials.json").read_text())
count=int(sys.argv[1])
out=root/"results-autodl"/("warmup" if count==8 else f"e1-c{count}")
if count==8:
    seed=0
    samples=8
    strict=True
else:
    config=historical[f"vllm-E1-c{count}"]["config"]
    assert len(config["sample_ids"])==96
    assert hashlib.sha256(script.read_bytes()).hexdigest()==config["source_sha256"]
    seed=config["seed"]
    samples=len(config["sample_ids"])
    strict=config["strict_errors"]
device=json.loads((root/"backend-device.json").read_text())
assert "4090" in device["device_name"] and device["bf16_supported"] and device["vllm"]=="0.33.0"
module=runpy.run_path(str(script),run_name="e1_h20_benchmark")
runner=module["runner"]
def local_device():
    return {
        "torch": str(module["torch"].__version__),
        "cuda_runtime": module["torch"].version.cuda,
        "transformers": "4.48.3",
        **device,
    }
runner.require_h20=local_device
from types import SimpleNamespace
args=SimpleNamespace(
    case='E1', backend="vllm", workers=count, sample=samples,
    seed=seed, context=32768, port=18420, model="published-qwen25",
    vllm_version=device["vllm"], output=out, shards=1, shard=0,
    strict_errors=strict,
)
raise SystemExit(0 if module["run"](args) else 1)
