#!/usr/bin/env bash
set -Eeuo pipefail
gpu="$1"
[[ "$gpu" == 0 || "$gpu" == 1 ]]
lab=/srv/adaptive-ttc-lab
legacy="$lab/results/paper_description/20261124T100045+0800"
runtime=/opt/inference-runtime/venv/bin
py="$legacy/source/.venv/bin/python"
port="$((18420 + gpu))"

exec 9>>"$lab/.gpu-$gpu.lock"
flock -n 9 || { echo "GPU $gpu already has an Adaptive TTC allocation"; exit 2; }
[[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]] ||
    { echo "GPU $gpu already has a compute process"; exit 3; }
[[ -z "$(ss -ltnH "sport = :$port")" ]] ||
    { echo "Port $port already has a listener"; exit 4; }

model="$("$py" -c 'from pathlib import Path; import json, sys; root=Path(sys.argv[1]); model=Path(json.loads((root/"E1-vllm"/"requests.json").read_text())["model"]); assert model.name=="Qwen--Qwen2.5-7B-Instruct--a09a35458c702b33eeacc393d103063234e8bc28" and (model/"config.json").is_file(); print(model)' "$legacy")"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
printf '\n%s Starting published-qwen25 on GPU %s, port %s\n' "$(date -Iseconds)" "$gpu" "$port"
exec env PYTHONPATH= "$runtime/vllm" serve "$model" --served-model-name published-qwen25 \
    --host 127.0.0.1 --port "$port" --dtype bfloat16 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.60 --max-model-len 32768 --max-num-seqs 112 \
    --max-num-batched-tokens 8192 --generation-config vllm --enable-prefix-caching
