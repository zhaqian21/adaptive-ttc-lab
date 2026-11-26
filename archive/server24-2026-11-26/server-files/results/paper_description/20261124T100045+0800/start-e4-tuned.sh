#!/usr/bin/env bash
set -euo pipefail
batch="$(cd "$(dirname "$0")" && pwd)"
lab="$(cd "$batch/../../.." && pwd)"
source_dir="$batch/source"
shared="$batch/E4-parallel"
worker="$batch/vllm-remaining"
py="$source_dir/.venv/bin/python"
runtime=/opt/inference-runtime/venv/bin
gpu="$1"
slots="$2"
tag="$3"
[[ "$gpu" == 0 || "$gpu" == 1 ]]
[[ "$slots" == 32 || "$slots" == 64 ]]
[[ "$tag" =~ ^[A-Za-z0-9-]+$ ]]
port=18409
[[ "$gpu" == 0 ]] || port=18411
endpoint="http://127.0.0.1:$port"
served_model=adaptive-ttc-qwen25-7b
logs="$shared/tuning/$tag"
mkdir "$logs"
cd "$batch"
exec 6>"$lab/.gpu-$gpu.lock"
flock -n 6
[[ -s "$shared/work.json" && ! -s "$batch/E4/audit.json" ]]
[[ "$(cat E4.status)" == running ]]
[[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]
[[ -z "$(ss -ltnH "sport = :$port")" ]]
model="$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$batch/E1-vllm/requests.json")"
version="$("$runtime/python" -c 'from importlib.metadata import version; print(version("vllm"))')"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

"$runtime/vllm" serve "$model" --served-model-name "$served_model" --host 127.0.0.1 --port "$port" \
  --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.40 \
  --max-model-len 32768 --max-num-seqs "$slots" --max-num-batched-tokens 4096 \
  --generation-config vllm --no-enable-prefix-caching > "$logs/engine.log" 2>&1 &
engine=$!
nvidia-smi -i "$gpu" --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=5 > "$logs/gpu.csv" &
sampler=$!
cleanup() { kill -TERM "$engine" "$sampler" 2>/dev/null || true; }
trap cleanup EXIT
for ((second=0; second<240; second+=2)); do
  if curl --max-time 2 -fsS "$endpoint/health" >/dev/null 2>&1; then break; fi
  kill -0 "$engine" 2>/dev/null || { echo "GPU $gpu engine exited. See $logs/engine.log."; exit 1; }
  sleep 2
done
curl --max-time 5 -fsS "$endpoint/health" >/dev/null
printf '%s GPU %s vLLM %s ready with %s slots.\n' "$(date -Iseconds)" "$gpu" "$version" "$slots"
touch "$logs/engine.ready"
export PYTHONPATH="$source_dir/scripts:$worker"
status=0
"$py" "$worker/run_e4_parallel_tuned.py" worker --gpu "$gpu" --endpoint "$endpoint" \
  --served-model "$served_model" --vllm-version "$version" --workers "$slots" > "$logs/evaluator.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$logs/evaluator.exit"
exit "$status"
