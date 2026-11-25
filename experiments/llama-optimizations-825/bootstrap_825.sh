#!/usr/bin/env bash
set -euo pipefail

bundle="$(cd -- "$(dirname -- "$0")" && pwd)"
root=$1
source="$root/source"
experiments="$bundle/experiments"
base="$experiments/math500-published-code-2026-11-24"
assets="$experiments/llama-table1-825"
existing="$HOME"
if [[ -x /data/users/chaqian/edge_cloud_risk/python/bin/python3.12 ]]; then
    existing=/data/users/chaqian/edge_cloud_risk
fi
python="$existing/python/bin/python3.12"
if [[ ! -x "$python" ]]; then python="$(command -v python3.12)"; fi
mkdir -p "$source/scripts" "$root/logs" "$root/cache" "$root/tmp" "$root/results"
export HF_HOME="$root/cache/huggingface" XDG_CACHE_HOME="$root/cache" TMPDIR="$root/tmp"
export UV_CACHE_DIR="$root/cache/uv" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0

phase() { printf '%s\n' "$1" > "$root/preparation.status"; }
trap 'rc=$?; printf "%s\n" "$rc" > "$root/exit"; if (( rc != 0 )); then phase "failed-$(cat "$root/preparation.status")"; fi' EXIT
phase preparing
cp -- "$base/scripts/"* "$source/scripts/"
cp -- "$base/environment.lock.txt" "$source/environment.lock.txt"

prepare_online() {
"$python" - "$assets/assets.json" "$bundle/asset-manifest.json" "$source" <<'PY'
import json
from pathlib import Path
import sys

assets, target, source = map(Path, sys.argv[1:])
entries = [entry for entry in json.loads(assets.read_bytes()) if not entry["path"].startswith("datasets/")]
suffix = "datasets/math-ai/amc23/resolve/80815d37005feb82cd7f8fbc6901d5d3eff43057/test-00000-of-00001.parquet"
entries.append({
    "path": "datasets/amc23.parquet",
    "sha256": "b696e87ba47be4e879a60fd4ef1d4aa522ba78c5ae013c6aff5cc9788a397c5e",
    "urls": [f"https://{host}/{suffix}" for host in ("huggingface.co", "hf-mirror.com")],
})
original = source.parent.parent / "llama-table1-825-20261124-v1" / "source"
for entry in entries:
    relative = Path(entry["path"])
    old, new = original / relative, source / relative
    if old.is_file() and not new.exists() and not new.is_symlink():
        new.parent.mkdir(parents=True, exist_ok=True)
        new.symlink_to(old.resolve())
target.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
PY

download() {
    for delay in 10 45 90; do
        if "$python" "$assets/prepare_assets.py" "$source" "$bundle/asset-manifest.json"; then return 0; fi
        sleep "$delay"
    done
    return 1
}
download >> "$root/logs/download.log" 2>&1 &
download_pid=$!

if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
else
    "$python" -m venv "$root/tools"
    "$root/tools/bin/python" -m pip install --disable-pip-version-check 'uv==0.13.7' >> "$root/logs/tools.log" 2>&1
    uv_bin="$root/tools/bin/uv"
fi
UV_BIN="$uv_bin" TTC_PYTHON="$python" bash "$source/scripts/bootstrap.sh" \
    >> "$root/logs/evaluation-environment.log" 2>&1

engine="$existing/venv/bin/python"
if ! "$engine" -c 'from importlib.metadata import version; assert version("vllm") == "0.33.0"' 2>/dev/null; then
    [[ -x "$root/vllm/bin/python" ]] || "$uv_bin" venv --python "$python" "$root/vllm"
    engine="$root/vllm/bin/python"
    "$uv_bin" pip install --python "$engine" 'vllm==0.33.0' >> "$root/logs/vllm-environment.log" 2>&1
fi
wait "$download_pid"

}

engine="$existing/venv/bin/python"
if [[ -n "${TTC_LOCAL_ASSET_STAGE:-}" ]]; then
    staged="$(realpath -- "$TTC_LOCAL_ASSET_STAGE")"
    [[ -s "$staged/transfer.ready" ]]
    for item in upstream assets datasets .venv; do
        [[ -d "$staged/source/$item" ]]
        if [[ -e "$source/$item" ]]; then
            [[ "$(realpath -- "$source/$item")" == "$(realpath -- "$staged/source/$item")" ]]
        else
            ln -s "$staged/source/$item" "$source/$item"
        fi
    done
    [[ "$(git -C "$source/upstream" rev-parse HEAD)" == d9ca5f71f5c7d155153e8f7decd8b1c524968775 ]]
    "$engine" -c 'from importlib.metadata import version; assert version("vllm") == "0.33.0"'
    "$staged/uv" pip check --python "$source/.venv/bin/python" >> "$root/logs/evaluation-environment.log" 2>&1
else
    prepare_online
fi

phase evaluating
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$source/scripts"
for exp in llama-table1-825 amc23-adaptive-revision math500-adaptive-revision math500-adaptive-k \
    math500-prm-answer-aggregation math500-online-prm-aggregation math500-prm-routing; do
    PYTHONPATH="$PYTHONPATH:$experiments/$exp"
done
PYTHONPATH="$PYTHONPATH:$experiments/aime24-amo-bench-published-code-2026-11-24/scripts"
"$source/.venv/bin/python" "$bundle/run_suite.py" --output "$root/results" \
    --dataset-file "$source/datasets/amc23.parquet" --engine-python "$engine" --workers 24
phase verified
