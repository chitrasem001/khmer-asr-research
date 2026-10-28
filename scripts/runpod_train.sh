#!/usr/bin/env bash
# Run one training job on a RunPod pod, keeping the full console log with the results.
#
#   bash runpod_train.sh <output_dir> <train.py args...>
#   e.g. bash runpod_train.sh experiments/m1_whisper_small \
#          --config configs/m1_whisper_small.yaml --train-hours 10 --max-steps 1000
#
# Needs /workspace/.hf_token (Hugging Face read token, mode 600) to avoid rate limits.
# The console log is written to <output_dir>/console.log, so copying <output_dir> back
# (without checkpoints/) brings the log too. Copy results back BEFORE deleting the pod.
set -euo pipefail
OUT="$1"; shift
cd /workspace
export HF_HOME=/workspace/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONIOENCODING=utf-8
[ -f /workspace/.hf_token ] && export HF_TOKEN="$(cat /workspace/.hf_token)"

mkdir -p "/workspace/khmer-asr-research/$OUT"
LOG="/workspace/khmer-asr-research/$OUT/console.log"
{
  echo "== START $(date -u)"
  bash /workspace/khmer-asr-research/scripts/runpod_setup.sh
  cd /workspace/khmer-asr-research
  echo "== TRAIN $(date -u) :: $*"
  python -u scripts/train.py --output-dir "$OUT" "$@"
  echo "== PACK $(date -u)"
  tar -cf "/workspace/$(basename "$OUT")_results.tar" --exclude="$(basename "$OUT")/checkpoints" \
    -C "$(dirname "$OUT")" "$(basename "$OUT")"
  echo "== DONE $(date -u)"
} 2>&1 | tee "$LOG"
