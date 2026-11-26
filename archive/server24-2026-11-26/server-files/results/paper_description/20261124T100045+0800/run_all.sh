#!/usr/bin/env bash
set -euo pipefail
batch="$(cd "$(dirname "$0")" && pwd)"
lab="$(cd "$batch/../../.." && pwd)"
source_dir="$batch/source"
python="$source_dir/.venv/bin/python"
dataset="$source_dir/assets/dataset/HuggingFaceH4--MATH-500--6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl"
cd "$source_dir"

exec 8>"$lab/.gpu-0.lock"
flock -n 8
gpu_processes="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$gpu_processes" ]] || { printf 'GPU 0 is in use: %s\n' "$gpu_processes" >&2; exit 1; }

set_status() {
  printf '%s\n' "$2" > "$batch/$1.status.tmp"
  mv "$batch/$1.status.tmp" "$batch/$1.status"
  "$python" "$batch/compare.py" "$batch"
}

failed=0
for case_id in E1 E4 E2 E3; do
  destination="$batch/$case_id"
  set_status "$case_id" running
  for attempt in 1 2; do
    if [[ -f "$destination/summary.json" ]]; then
      break
    fi
    options=()
    [[ ! -d "$destination" ]] || options+=(--resume)
    printf '%s %s attempt %s\n' "$(date -Iseconds)" "$case_id" "$attempt"
    if TTC_GPU_INDEX=0 bash scripts/run_smoke.sh "$case_id" "$destination" --seed 0 "${options[@]}" >> "$batch/$case_id.log" 2>&1; then
      break
    fi
    if [[ "$attempt" == 1 ]]; then
      printf 'Retrying %s after a failed attempt. See %s.log\n' "$case_id" "$case_id"
      sleep 12
    fi
  done
  if [[ -f "$destination/summary.json" ]] &&
    "$python" scripts/audit_results.py "$destination" --dataset "$dataset" > "$batch/$case_id.audit.json.tmp" 2>> "$batch/$case_id.log"; then
    mv "$batch/$case_id.audit.json.tmp" "$batch/$case_id.audit.json"
    set_status "$case_id" verified
  else
    set_status "$case_id" failed
    failed=1
  fi
done
exit "$failed"
