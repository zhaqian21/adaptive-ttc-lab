#!/usr/bin/env bash
set -euo pipefail
worker="$(cd "$(dirname "$0")" && pwd)"
batch="$(dirname "$worker")"
lab="$(cd "$batch/../../.." && pwd)"
source_dir="$batch/source"
py="$source_dir/.venv/bin/python"
runtime=/opt/inference-runtime/venv/bin
config="$batch/E1-vllm/requests.json"
port=18409
endpoint="http://127.0.0.1:$port"
served_model=adaptive-ttc-qwen25-7b
cd "$worker"
exec 8>"$lab/.gpu-0.lock"
flock -n 8
[[ "$(cat "$batch/E1-vllm.status")" == verified ]]
[[ -z "$(ss -ltnH "sport = :$port")" ]]
[[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)" ]]
model="$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$config")"
version="$("$runtime/python" -c 'from importlib.metadata import version; print(version("vllm"))')"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

"$runtime/vllm" serve "$model" --served-model-name "$served_model" --host 127.0.0.1 --port "$port" \
  --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.40 \
  --max-model-len 32768 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --generation-config vllm --no-enable-prefix-caching > "$worker/engine-gpu0-e3.log" 2>&1 &
engine=$!
nvidia-smi -i 0 --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=5 > "$worker/gpu-0-e3.csv" &
sampler=$!
cleanup() { kill -TERM "$engine" "$sampler" 2>/dev/null || true; }
trap cleanup EXIT

for ((second=0; second<240; second+=2)); do
  if curl --max-time 2 -fsS "$endpoint/health" >/dev/null 2>&1; then break; fi
  kill -0 "$engine" 2>/dev/null || { echo "vLLM engine exited. See engine.log."; exit 1; }
  sleep 2
done
curl --max-time 5 -fsS "$endpoint/health" >/dev/null
curl --max-time 5 -fsS "$endpoint/v1/models" | "$py" -c 'import sys,json; assert sys.argv[1] in {r["id"] for r in json.load(sys.stdin)["data"]}' "$served_model"
echo "$(date -Iseconds) vLLM $version ready for E2 → E3 → E4."
export PYTHONPATH="$source_dir/scripts"
set_status() {
  printf '%s\n' "$2" > "$batch/$1.status.tmp"
  mv "$batch/$1.status.tmp" "$batch/$1.status"
  "$py" "$batch/compare-vllm.py" "$batch"
}

for case_id in E3; do
  destination="$batch/$case_id"
  options=(--e1-config "$config" --endpoint "$endpoint" --served-model "$served_model" --vllm-version "$version" --workers 16)
  set_status "$case_id" running
  if [[ ! -s "$worker/smoke-$case_id/audit.json" ]]; then
    smoke=()
    [[ ! -d "$worker/smoke-$case_id" ]] || smoke+=(--resume)
    if ! "$py" run_vllm_cases_impl.py "$case_id" "${options[@]}" --limit 2 --output "$worker/smoke-$case_id" "${smoke[@]}" > "$worker/$case_id.smoke.log" 2>&1; then
      set_status "$case_id" failed
      echo "$case_id smoke failed. See $worker/$case_id.smoke.log."
      exit 1
    fi
  fi
  for attempt in 1 2; do
    [[ ! -s "$destination/audit.json" ]] || break
    resume=()
    [[ ! -d "$destination" ]] || resume+=(--resume)
    printf '%s %s attempt %s\n' "$(date -Iseconds)" "$case_id" "$attempt"
    if "$py" run_vllm_cases_impl.py "$case_id" "${options[@]}" --output "$destination" "${resume[@]}" >> "$batch/$case_id.log" 2>&1; then
      break
    fi
    if [[ "$attempt" == 1 ]]; then sleep 8; fi
  done
  if [[ ! -s "$destination/audit.json" ]]; then
    set_status "$case_id" failed
    echo "$case_id stopped with unverified questions. See $batch/$case_id.log."
    exit 1
  fi
  cp "$destination/audit.json" "$batch/$case_id.audit.json"
  set_status "$case_id" verified
done
