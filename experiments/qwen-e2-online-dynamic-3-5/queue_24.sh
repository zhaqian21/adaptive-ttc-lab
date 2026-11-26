#!/usr/bin/env bash
set -euo pipefail

[[ $# == 2 ]] || { echo "usage: $0 RUN_DIRECTORY VLLM_PYTHON" >&2; exit 2; }
here="$(cd -- "$(dirname -- "$0")" && pwd)"
run="$(realpath -- "$1")"
launcher="$here/../gsm8k-500-e4-direct/run_24.sh"
python="$run/source/.venv/bin/python"
export PYTHONPATH="$here/../aime24-amo-bench-published-code-2026-11-24/scripts:$run/source/scripts"

exec 8>"$run/queue.lock"
flock -n 8
trap 'rc=$?; printf "%s\n" "$rc" > "$run/queue.exit"' EXIT
trap 'exit 143' TERM INT

rm -f -- "$run/queue.exit" "$run/run.exit"
echo running > "$run/queue.status"
bash "$launcher" "$run" "$2" "$here/run.py" --replace-gpu7
[[ "$(cat "$run/service-restoration")" == restored ]]
echo auditing > "$run/queue.status"
"$python" -B "$here/archive.py" --data "$run/data" --results "$run/results" --output "$run/export"
tar -cf "$run/online-review.tar.tmp" -C "$run/export" .
mv -- "$run/online-review.tar.tmp" "$run/online-review.tar"
touch "$run/audit.ok"
echo complete > "$run/queue.status"
