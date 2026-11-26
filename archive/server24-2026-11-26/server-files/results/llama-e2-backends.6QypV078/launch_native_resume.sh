#!/usr/bin/env bash
set -uo pipefail
cd '/srv/adaptive-ttc-lab/results/llama-e2-backends.6QypV078' || exit
source_root=../llama-table1-24/v1/source
python="$source_root/.venv/bin/python"
model="$source_root/assets/model/meta-llama--Llama-3.1-8B-Instruct--359efdbb8af05b788a4ad4185215c6b8caa9052c"
phase=pilot
manifest="sample/${phase}_manifest.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2 PYTHONPATH="$source_root/upstream"
mv -- "$phase/transformers.exit" "$phase/initial-transformers.exit" || exit
mv -- "$phase/generation.exit" "$phase/initial-generation.exit" || exit
CUDA_VISIBLE_DEVICES= "$python" resume_native.py plan --manifest "$manifest" --work "$phase" > "$phase/native-plan.log" 2>&1 || exit
worker() {
    local card="$1" rc
    CUDA_VISIBLE_DEVICES="$card" "$python" resume_native.py run --manifest "$manifest" \
      --work "$phase" --model "$model" --lane "$card" > "$phase/native-gpu-${card}.log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$phase/native-gpu-${card}.exit"
    return "$rc"
}
worker 0 &
lane0=$!
# Wait until the experiment's vLLM process has released its GPU allocation.
for ((try=0;try<90;try++)); do
    vram="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$vram" =~ ^[0-9]+$ && "$vram" -lt 64000 ]]; then break; fi
    sleep 2
done
if [[ ! "$vram" =~ ^[0-9]+$ || "$vram" -ge 64000 ]]; then
    printf 'GPU 1 still using %s MiB\n' "$vram" > "$phase/native-gpu-1.log"
    printf '1\n' > "$phase/native-gpu-1.exit"
    exit 1
fi
worker 1 &
lane1=$!
wait "$lane0"; status0=$?
wait "$lane1"; status1=$?
if ((status0 || status1)); then exit 1; fi
CUDA_VISIBLE_DEVICES= "$python" resume_native.py merge --manifest "$manifest" --work "$phase" \
  > "$phase/native-merge.log" 2>&1 || exit
printf '0\n' > "$phase/transformers.exit"
printf '0 0\n' > "$phase/generation.exit"
CUDA_VISIBLE_DEVICES= "$python" score_candidates.py --manifest "$manifest" --generation-root "$phase" \
  --history "$source_root/../results" > "$phase/scoring.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$phase/scoring.exit"
exit "$rc"
