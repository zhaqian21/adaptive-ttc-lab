#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 SOURCE BASELINE_SCRIPTS OUTPUT [SAMPLE]" >&2
    exit 2
fi

entry="$(cd -- "$(dirname -- "$0")" && pwd)"
source_root="$(realpath -- "$1")"
baseline="$(realpath -- "$2")"
mkdir -p -- "$3"
output="$(realpath -- "$3")"
sample=()
if [[ $# == 4 ]]; then sample=("$4"); fi

printf '%s\n' single-prm > "$output/status"
bash "$entry/../math500-adaptive-k/run_dual.sh" "$source_root" "$baseline" "$output/single-prm" adaptive "${sample[@]}"
printf '%s\n' answer-prm-sum > "$output/status"
bash "$entry/../math500-adaptive-k/run_dual.sh" "$source_root" "$baseline" "$output/answer-prm-sum" adaptive-answer-sum "${sample[@]}"
printf '%s\n' auditing > "$output/status"
export PYTHONPATH="$entry/../math500-adaptive-k:$entry/../math500-adaptive-revision:$entry/../math500-prm-answer-aggregation"
"$source_root/.venv/bin/python" "$entry/compare.py" "$output"
printf '%s\n' verified > "$output/status"
