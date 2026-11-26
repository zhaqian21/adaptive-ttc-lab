#!/usr/bin/env bash
set -euo pipefail
root=$1 route=$2 launcher=$3
until [[ -s "$root/run.exit" && -s "$route/driver.exit" ]]; do sleep 5; done
[[ "$(cat "$root/run.exit")" == 1 && "$(cat "$route/driver.exit")" == 1 ]]
[[ "$(cat "$root/service-restoration")" == restored ]]
if tmux list-sessions -F '#S' | grep -Fxq ttc-llama24-e2-gpu1-recovery; then
    tmux kill-session -t '=ttc-llama24-e2-gpu1-recovery'
fi
for f in "$root/run.exit" "$root/driver.log" "$root/results/failure.json" "$route/driver.exit" "$root/service-restoration"; do
    [[ ! -e "$f.before-prm-queue-retry" ]]
    mv -- "$f" "$f.before-prm-queue-retry"
done
for n in 0 1; do
    d="$root/results/formal/math/gpu-$n/E2"
    for f in summary.json predictions.json; do
        if [[ -f "$d/$f" ]]; then cp -n -- "$d/$f" "$d/$f.before-prm-queue-retry"; fi
    done
done
printf '%s\n' "Recovered PRM listener backlog at $(date -Is)" > "$root/prm-queue-recovery.log"
exec bash "$route/resume-llama-24.sh" "$route" "$launcher"
