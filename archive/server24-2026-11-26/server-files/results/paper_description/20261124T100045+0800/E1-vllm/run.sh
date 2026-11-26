#!/usr/bin/env bash
set -euo pipefail
case_dir="$(cd "$(dirname "$0")" && pwd)"
batch="$(dirname "$case_dir")"
lab="$(cd "$batch/../../.." && pwd)"
py="$batch/source/.venv/bin/python"
vllm_py="/opt/inference-runtime/venv/bin/python"
cd "$case_dir"
exec 8>"$lab/.gpu-0.lock"
flock -n 8
uuid="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
set_status() {
  printf '%s\n' "$1" > "$batch/E1-vllm.status.tmp"
  mv "$batch/E1-vllm.status.tmp" "$batch/E1-vllm.status"
  "$py" "$batch/compare-vllm.py" "$batch"
}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
set_status running
for attempt in 1 2; do
  [[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)" ]] || { set_status failed; exit 1; }
  if CUDA_VISIBLE_DEVICES="$uuid" PYTHONPATH="" timeout --signal=TERM --kill-after=20s 1200s "$vllm_py" run.py decode; then
    if CUDA_VISIBLE_DEVICES="" PYTHONPATH="$batch/source/scripts" "$py" run.py score; then
      set_status verified
      flock -u 8
      exec 8>&-
      exec bash "$batch/run_remaining.sh"
    fi
  fi
  printf 'E1 vLLM attempt %s failed\n' "$attempt"
  sleep 8
done
set_status failed
exit 1
