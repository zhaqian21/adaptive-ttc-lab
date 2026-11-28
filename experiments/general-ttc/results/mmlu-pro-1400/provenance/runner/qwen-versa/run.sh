#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/adaptive-ttc-lab/general-ttc
cd "$root"
py=/root/ttc-vgpu48-runtime/e1-prm/bin/python
revision=2046e098ed83869a4d71fe772cc18b62336ac3de
queue="$root/queue-e3-e2-e4"
export PYTHONPATH="$root/concurrent-source" PYTHONDONTWRITEBYTECODE=1
exec 9>"$queue/queue.lock"
flock -n 9
log() { printf '%s %s\n' "$(date -Is)" "$*"; }
stage() { printf '%s\n' "$*" >"$queue/stage.tmp"; mv "$queue/stage.tmp" "$queue/stage"; log "$*"; }
run() {
    local method="$1" workers="$2" manifest="$3" output="$4"
    shift 4
    "$py" -u concurrent-source/run.py --manifest "$manifest" --method "$method" --workers "$workers" \
        --generator-model published-qwen25 --code-revision "$revision" --output "$output" "$@"
}
stable_health() {
    curl -fsS --max-time 4 http://127.0.0.1:18420/health >/dev/null
    curl -fsS --max-time 8 http://127.0.0.1:18423/health >/dev/null
}
if [[ "${1:-}" == --resume ]]; then
    stable_health
    winner="$("$py" "$queue/audit.py" choose)"
else
    stage "Preparing frozen E3 stress test"
    stable_health
    "$py" "$queue/audit.py" prepare
    sample="$queue/benchmark-manifest.json"
    mkdir -p "$queue/benchmark"
    for workers in 24 48 72 96 112; do
        stage "Benchmark E3 c$workers (140 paired questions)"
        folder="$queue/benchmark/c$workers"
        if run E3 "$workers" "$sample" "$folder" >"$queue/benchmark/c$workers.log" 2>&1; then
            log "c$workers completed"
        else
            log "c$workers failed; see its log"
            stable_health
        fi
    done
    winner="$("$py" "$queue/audit.py" choose)"
    stage "Confirm E3 c$winner (same 140 questions)"
    if ! run E3 "$winner" "$sample" "$queue/benchmark/c$winner-confirm" >"$queue/benchmark/c$winner-confirm.log" 2>&1; then
        log "Confirmation failed at c$winner"
        stable_health
        mv "$queue/benchmark/c$winner" "$queue/benchmark/c$winner-unconfirmed"
        # Keep the failed confirmation alongside the logs, outside the eligible trials.
        mkdir -p "$queue/rejected"
        mv "$queue/benchmark/c$winner-unconfirmed" "$queue/benchmark/c$winner-confirm" "$queue/rejected/"
    fi
    winner="$("$py" "$queue/audit.py" choose)"
fi
manifest="$root/mmlu-pro-1400.json"
run_full() {
    local method="$1" workers="$2" folder="$root/full/$1-1400" retries=0
    local -a flags=()
    if [[ -s "$folder/summary.json" ]]; then
        "$py" "$queue/audit.py" compare --method "$method"
        return
    fi
    if [[ -s "$folder/config.json" ]]; then flags=(--resume); fi
    while true; do
        stage "Running $method (1400 questions, c$workers)"
        if run "$method" "$workers" "$manifest" "$folder" "${flags[@]}" >>"$queue/$method.log" 2>&1; then
            "$py" "$queue/audit.py" compare --method "$method"
            break
        fi
        if (( retries >= 2 )) || [[ ! -s "$folder/config.json" ]]; then
            log "$method needs attention; see $queue/$method.log"
            return 1
        fi
        stable_health
        retries=$((retries + 1))
        workers=$((workers > 8 ? workers / 2 : workers))
        flags=(--resume)
        log "$method retry $retries will preserve completed questions"
    done
}
run_full E3 "$winner"
# The E2/E4 outer loop sends ten PRM-scored candidates per question.
workers=$((winner < 48 ? winner : 48))
run_full E2 "${E2_WORKERS:-$workers}"
run_full E4 "${E4_WORKERS:-$workers}"
stage "Complete: E3, E2 and E4; paired reports saved"
