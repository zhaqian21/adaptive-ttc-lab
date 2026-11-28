#!/usr/bin/env bash
set -euo pipefail
base=/root/autodl-tmp/adaptive-ttc-lab/vgpu48
cd "$base"
engine=/root/ttc-vgpu48-runtime/e1-engine
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-13.0
export PATH="$engine/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME="$PWD/cache/hf" XDG_CACHE_HOME="$PWD/cache/xdg"
export VLLM_CACHE_ROOT="$PWD/cache/e1-vllm" TRITON_CACHE_DIR="$PWD/cache/e1-triton" FLASHINFER_WORKSPACE_BASE="$PWD/cache/e1-flashinfer"
exec 5>logs/model-check-gpu.lock
flock -n 5
exec 9>logs/generation.lock
flock -n 9
exec "$engine/bin/vllm" serve "$PWD/models/Qwen2.5-7B-Instruct" --host 127.0.0.1 --port 18420 \
  --served-model-name published-qwen25 --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 32768 \
  --max-num-seqs 112 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.60 \
  --enable-prefix-caching --generation-config vllm
