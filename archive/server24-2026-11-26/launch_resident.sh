#!/usr/bin/env bash
set -euo pipefail

[[ $# == 2 ]] || { echo "usage: $0 generate|score GPU" >&2; exit 2; }
role="$1"
gpu="$2"
source_dir="${TTC_SOURCE_DIR:?Set TTC_SOURCE_DIR to the pinned source and asset checkout}"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^GPU-[0-9a-f-]+$ ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

case "$role:$gpu" in
    generate:0|generate:1)
        port="$((18420+gpu))"
        name=published-qwen25
        ;;
    generate:7)
        port=18428
        name=Qwen/Qwen2.5-7B-Instruct
        ;;
    score:0|score:7)
        repo_dir="${TTC_EXPERIMENT_CHECKOUT:?Check out a referenced experiment branch with the PRM runner}"
        if [[ "$gpu" == 0 ]]; then port=18422; else port=18438; fi
        export PYTHONPATH="$repo_dir/experiments/aime24-amo-bench-published-code-2026-11-24/scripts:$source_dir/scripts"
        exec "$source_dir/.venv/bin/python" -B "$repo_dir/experiments/llama-table1-825/llama_table1.py" \
            --serve-prm --prm-port "$port"
        ;;
    *) echo "Unsupported role or GPU slot" >&2; exit 2 ;;
esac

model="$source_dir/assets/model/Qwen--Qwen2.5-7B-Instruct--a09a35458c702b33eeacc393d103063234e8bc28"
exec env -u PYTHONPATH "${VLLM_PYTHON:?Set VLLM_PYTHON to the vLLM 0.33.0 interpreter}" \
    -m vllm.entrypoints.cli.main serve "$model" --host 127.0.0.1 --port "$port" \
    --served-model-name "$name" --dtype bfloat16 --tensor-parallel-size 1 \
    --generation-config vllm --gpu-memory-utilization 0.60 --max-model-len 32768 \
    --max-num-seqs 112 --max-num-batched-tokens 8192 --enable-prefix-caching
