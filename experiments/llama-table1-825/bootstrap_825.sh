#!/usr/bin/env bash
set -euo pipefail

bundle="$(cd -- "$(dirname -- "$0")" && pwd)"
root=$1
source="$root/source"
output="$root/results"
existing=/data/users/chaqian/edge_cloud_risk
python="$existing/python/bin/python3.12"
mkdir -p "$source/scripts" "$source/datasets" "$root/logs" "$root/cache" "$root/tmp" "$output"
export HF_HOME="$root/cache/huggingface" XDG_CACHE_HOME="$root/cache" TMPDIR="$root/tmp"
export UV_CACHE_DIR="$root/cache/uv" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
cp "$bundle/baseline/"*.py "$bundle/baseline/bootstrap.sh" "$bundle/baseline/constraints.txt" "$source/scripts/"
cp "$bundle/baseline/environment.lock.txt" "$source/environment.lock.txt"
cp "$bundle/datasets.json" "$source/datasets/manifest.json"

phase() { echo "$1" > "$root/preparation.status"; }
trap 'rc=$?; echo "$rc" > "$root/exit"; if (( rc != 0 )); then phase "failed-$(cat "$root/preparation.status")"; fi' EXIT

phase downloading-assets-and-preparing-environment
"$python" "$bundle/prepare_assets.py" "$source" "$bundle/assets.json" > "$root/logs/download.log" 2>&1 &
download_pid=$!

if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
else
  "$python" -m venv "$root/tools"
  "$root/tools/bin/python" -m pip install --disable-pip-version-check 'uv==0.13.7' > "$root/logs/tools.log" 2>&1
  uv_bin="$root/tools/bin/uv"
fi
UV_BIN="$uv_bin" TTC_PYTHON="$python" bash "$source/scripts/bootstrap.sh" \
    > "$root/logs/evaluation-environment.log" 2>&1
engine="$existing/venv/bin/python"
if ! "$engine" -c 'from importlib.metadata import version; assert version("vllm") == "0.33.0"' 2>/dev/null; then
  "$uv_bin" venv --python "$python" "$root/vllm"
  engine="$root/vllm/bin/python"
  "$uv_bin" pip install --python "$engine" 'vllm==0.33.0' > "$root/logs/vllm-environment.log" 2>&1
fi
wait "$download_pid"

phase evaluating
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$bundle/released:$source/scripts"
"$source/.venv/bin/python" "$bundle/llama_table1.py" --manifest "$source/datasets/manifest.json" \
    --output "$output" --engine-python "$engine"
phase verified
