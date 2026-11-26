#!/usr/bin/env bash
set -euo pipefail
service=/srv/adaptive-ttc-lab/services/e4-gpu7
exp=/srv/adaptive-ttc-lab/results/e4-gsm8k-aime26.eM54G6xe
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 7 --query-gpu=uuid --format=csv,noheader)"
[[ "$CUDA_VISIBLE_DEVICES" == GPU-REPLACE-WITH-UUID ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export PYTHONPATH="$exp/experiments/aime24-amo-bench-published-code-2026-11-24/scripts:$exp/source/scripts"
cd "$service"
exec "$exp/source/.venv/bin/python" -B "$service/serve.py"
