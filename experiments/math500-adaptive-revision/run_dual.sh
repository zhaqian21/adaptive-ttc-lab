#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "usage: $0 SOURCE BASELINE_SCRIPTS BASELINE_E4_JSONL OUTPUT VLLM_VERSION [SAMPLE]" >&2
    exit 2
fi

entry="$(cd -- "$(dirname -- "$0")" && pwd)"
source_root="$(cd -- "$1" && pwd)"
baseline_scripts="$(cd -- "$2" && pwd)"
reference="$(realpath -- "$3")"
mkdir -p -- "$4"
output="$(cd -- "$4" && pwd)"
version=$5
python="$source_root/.venv/bin/python"
workers=24
sample=()
if [[ $# == 6 ]]; then
    workers=4
    sample=(--sample "$6")
fi

export PYTHONPATH="$baseline_scripts:$source_root/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
for port in 18420 18421 18422; do
    curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:$port/health" >/dev/null
done

pids=()
stop() {
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait || true
    exit 143
}
trap stop TERM INT

for lane in 0 1; do
    (
        rc=0
        "$python" "$entry/run_math500.py" --port "$((18420 + lane))" --prm-port 18422 \
            --vllm-version "$version" --workers "$workers" --seed 0 --token-budget 125000 \
            --initial-roots 3 --shards 2 --shard "$lane" "${sample[@]}" \
            --output "$output/shard-$lane" || rc=$?
        printf '%s\n' "$rc" > "$output/shard-$lane.exit"
        exit "$rc"
    ) > "$output/shard-$lane.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]]
"$python" "$entry/audit_math500.py" "$output" "$reference" "$output/export"
