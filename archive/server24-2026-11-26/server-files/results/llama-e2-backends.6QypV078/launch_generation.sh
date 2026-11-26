#!/usr/bin/env bash
set -uo pipefail
cd '/srv/adaptive-ttc-lab/results/llama-e2-backends.6QypV078'
phase="${1:?expected pilot or extension}"
case "$phase" in pilot|extension) ;; *) exit 2 ;; esac
source_root=../llama-table1-24/v1/source
model="$source_root/assets/model/meta-llama--Llama-3.1-8B-Instruct--359efdbb8af05b788a4ad4185215c6b8caa9052c"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2
mkdir -p "$phase"
worker() {
  local backend="$1" card="$2" batch="$3" rc
  CUDA_VISIBLE_DEVICES="$card" "$source_root/.venv/bin/python" generate_candidates.py \
    --backend "$backend" --manifest "sample/${phase}_manifest.json" --model "$model" \
    --output "$phase/$backend" --batch "$batch" > "$phase/$backend.log" 2>&1
  rc=$?
  printf '%s\n' "$rc" > "$phase/$backend.exit"
  return "$rc"
}
worker transformers 0 16 &
hf=$!
worker vllm '' 64 &
fast=$!
wait "$hf"
status_hf=$?
wait "$fast"
status_vllm=$?
printf '%s %s\n' "$status_hf" "$status_vllm" > "$phase/generation.exit"
