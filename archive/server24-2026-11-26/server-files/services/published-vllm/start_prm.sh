#!/usr/bin/env bash
set -Eeuo pipefail
operation="$(cd "$(dirname "$0")" && pwd)"
lab="$(cd "$operation/../.." && pwd)"
source="$lab/results/paper_description/20261124T100045+0800/source"
exec 9>>"$operation/prm-gpu0.lock"
flock -n 9
[[ -z "$(ss -ltnH 'sport = :18422')" ]]
curl -fsS --max-time 5 http://127.0.0.1:18420/health >/dev/null
free_mib="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free_mib >= 20480 ))
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$source/scripts"
printf '\n%s Starting PRM on GPU 0\n' "$(date -Iseconds)"
exec "$source/.venv/bin/python" "$operation/serve_prm.py"
