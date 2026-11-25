#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "usage: $0 VLLM_PYTHON MODEL_DIR PORT" >&2
    exit 2
fi
exec "$1" -m vllm.entrypoints.cli.main serve "$2" \
  --host 127.0.0.1 --port "$3" --served-model-name meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 --tensor-parallel-size 1 --generation-config vllm \
  --gpu-memory-utilization 0.32 --max-model-len 32768 --enable-prefix-caching \
  --max-num-seqs 112 --max-num-batched-tokens 8192
