#!/usr/bin/env bash
set -euo pipefail

[[ $# == 3 ]] || { echo "usage: $0 PREVIOUS_E4_RUN E2_RUN VLLM_PYTHON" >&2; exit 2; }
here="$(cd -- "$(dirname -- "$0")" && pwd)"
previous="$(realpath -- "$1")"
root="$(realpath -- "$2")"
engine="$(realpath -s -- "$3")"
launcher="$here/../gsm8k-500-e4-direct/run_24.sh"

exec 8>"$root/queue.lock"
flock -n 8
trap 'rc=$?; printf "%s\n" "$rc" > "$root/queue.exit"' EXIT
trap 'exit 143' TERM INT

echo waiting-for-e4 > "$root/queue.status"
until [[ -s "$previous/run.exit" ]]; do sleep 20; done
if [[ "$(cat "$previous/run.exit")" != 0 || "$(cat "$previous/service-restoration")" != restored ||
      ! -s "$previous/results/e4/generation.json" ]]; then
    echo blocked-by-e4 > "$root/queue.status"
    exit 1
fi

echo running-e2 > "$root/queue.status"
bash "$launcher" "$root" "$engine" "$here/run_e2.py"
[[ "$(cat "$root/service-restoration")" == restored ]]
echo auditing-e2 > "$root/queue.status"
"$root/source/.venv/bin/python" -B "$here/archive_e2.py" --run "$root/results" --archive "$root/export" --output "$root/export"
tar -cf "$root/e2-review.tar" -C "$root/export" .
touch "$root/audit.ok"
echo complete > "$root/queue.status"
