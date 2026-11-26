#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
case_id="$1"
run_dir="$2"
gpu_index="${TTC_GPU_INDEX:-0}"
shift 2

if [[ "$case_id" != A && "$case_id" != B && "$case_id" != C && "$case_id" != validate ]]; then
  echo "Choose smoke case A, B, C, or validate." >&2
  exit 2
fi

uuid="$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader)"
exec 9>".gpu-$gpu_index.lock"
flock -n 9
if [[ -n "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "GPU $gpu_index already has a compute process." >&2
  exit 1
fi

mkdir -p logs
tag="$(date -u +%Y%m%dT%H%M%S)-$$"
nvidia-smi -i "$uuid" --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv --loop=2 > "logs/gpu-$case_id-$tag.csv" &
sampling_pid=$!
trap 'kill "$sampling_pid" 2>/dev/null || true' EXIT

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$uuid"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
if [[ "$case_id" == validate ]]; then
  .venv/bin/python scripts/validate_models.py --output "$run_dir" "$@"
else
  .venv/bin/python scripts/run_smoke.py "$case_id" --output "$run_dir" "$@"
fi
