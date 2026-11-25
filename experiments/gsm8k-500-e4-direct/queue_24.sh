#!/usr/bin/env bash
set -euo pipefail

[[ $# == 3 ]] || { echo "usage: $0 CURRENT_RUN FOLLOWUP_RUN VLLM_PYTHON" >&2; exit 2; }
here="$(cd -- "$(dirname -- "$0")" && pwd)"
previous="$(realpath -- "$1")"
next="$(realpath -- "$2")"
engine="$(realpath -s -- "$3")"

exec 8>"$next/queue.lock"
flock -n 8
trap 'rc=$?; printf "%s\n" "$rc" > "$next/queue.exit"' EXIT
trap 'exit 143' TERM INT
echo waiting-for-gsm8k-500 > "$next/queue.status"
until [[ -s "$previous/run.exit" ]]; do sleep 15; done
[[ "$(cat "$previous/run.exit")" == 0 ]]
[[ "$(cat "$previous/service-restoration")" == restored ]]
echo running-direct-followup > "$next/queue.status"
bash "$here/run_24.sh" "$next" "$engine" "$here/followup.py"
[[ "$(cat "$next/service-restoration")" == restored ]]
echo direct-followup-complete > "$next/queue.status"
