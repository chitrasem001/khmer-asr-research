#!/usr/bin/env bash
# Round 4 on one RunPod pod: M2 (MMS-1B with the adapter started from the released khm adapter) on
# the full training split for one pass (24,000 steps, batch 16) — the same budget as the full-data
# Whisper run (f_whisper_small_872h), so the two families can be compared at full data.
# The model is scored on test, FLEURS and test_seen; best/ weights go to the private HF repo.
#
#   nohup bash scripts/runpod_round4.sh > /workspace/round4_launch.log 2>&1 &
#
# Needs /workspace/.hf_token (read) and /workspace/.hf_write (write). Results without weights end
# up in /workspace/round4_results.tar.
set -uo pipefail   # no -e: one failed run must not stop the others
cd /workspace/khmer-asr-research
export HF_HOME=/workspace/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONIOENCODING=utf-8
export TRANSFORMERS_VERBOSITY=error HF_TOKEN="$(cat /workspace/.hf_token)"
LOG=experiments/round4_console.log
MODEL_REPO=chitra168/khmer-asr-thesis-models
DATA_REV=e872b0c58b4bbbe30c0244def61e39e7b57ea093
mkdir -p experiments
exec > >(tee -a "$LOG") 2>&1

# evaluate <name> <name_or_path> <revision> <family> <language> <set...>; set = test|fleurs|seen
evaluate() {
  local name=$1 path=$2 rev=$3 family=$4 lang=$5; shift 5
  local bs; bs=$([ "$family" = whisper ] && echo 16 || echo 8)
  cat > "/workspace/eval_$name.yaml" <<EOF
experiment: {name: $name, seed: 42, output_dir: experiments/m1_eval}
models:
  - {name_or_path: $path, revision: $rev, family: $family, language: $lang, batch_size: $bs}
data:
  dataset: Digital-Divide-Data/khmer-speech-dataset
  revision: $DATA_REV
  test_manifest: data/manifests/test.csv
  audio_dir: data/processed/audio
  sampling_rate: 16000
inference: {fp16: true, decoding: greedy}
EOF
  for set in "$@"; do
    echo "== EVAL $name ${set^^} $(date -u)"
    case $set in
      test) python -u scripts/evaluate.py --config "/workspace/eval_$name.yaml" ;;
      fleurs) python -u scripts/evaluate.py --config "/workspace/eval_$name.yaml" \
        --manifest data/manifests/fleurs_km_test.csv --audio-dir data/processed/fleurs_audio \
        --output-dir experiments/m1_eval_fleurs ;;
      seen) python -u scripts/evaluate.py --config "/workspace/eval_$name.yaml" \
        --manifest data/manifests/test_seen_speakers.csv --output-dir experiments/m1_eval_seen ;;
    esac
  done
}

upload() {   # upload <local dir> <path in repo>
  HF_TOKEN="$(cat /workspace/.hf_write)" python - "$1" "$2" "$MODEL_REPO" <<'PY'
import sys
from huggingface_hub import HfApi
local, remote, repo = sys.argv[1:]
HfApi().upload_folder(folder_path=local, path_in_repo=remote, repo_id=repo, repo_type="model",
                      ignore_patterns=["checkpoints/*", "train_subset.csv"])
print("uploaded", remote)
PY
}

echo "== ROUND4 START $(date -u)"
git pull --ff-only
export SKIP_PULL=1   # pin the code for the whole round (runpod_setup.sh skips pulls)
python -m pip install -q -e .
bash scripts/runpod_setup.sh    # rebuilds train.parquet and checks the split hashes
echo "== PREPARE EVAL AUDIO $(date -u)"
python -u scripts/extract_audio.py --manifest data/manifests/test.csv --workers 8
python -u scripts/extract_audio.py --manifest data/manifests/test_seen_speakers.csv --workers 8
python -u scripts/prepare_fleurs.py --split test

# name | config | train.py args
RUNS=(
  "f_mms_khm_init_872h|configs/m2_mms_khm_init.yaml|--max-steps 24000 --eval-steps 4000 --warmup-steps 100"
)
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name cfg extra <<< "$spec"
  out="experiments/$name"
  echo "== RUN $name START $(date -u)"
  # shellcheck disable=SC2086  # extra holds several arguments
  if bash scripts/runpod_train.sh "$out" --config "$cfg" $extra; then
    family=$(grep -m1 "family:" "$cfg" | awk '{print $2}')
    lang=$([ "$family" = whisper ] && echo km || echo null)
    evaluate "$name" "$out/best" null "$family" "$lang" test fleurs seen
    echo "== UPLOAD $name $(date -u)"
    upload "$out" "$name"
    echo "== RUN $name DONE $(date -u)"
  else
    echo "== RUN $name FAILED $(date -u)"
  fi
  rm -rf "$out/checkpoints"
done

# The full-data run's train_subset.csv is the whole training split (rebuilt and hash-checked).
rm -f experiments/f_mms_khm_init_872h/train_subset.csv
echo "== PACK $(date -u)"
tar -cf /workspace/round4_results.tar --exclude='*.safetensors' --exclude='*/checkpoints'   experiments/f_mms_khm_init_872h experiments/m1_eval/f_mms_khm_init_872h   experiments/m1_eval_fleurs/f_mms_khm_init_872h experiments/m1_eval_seen/f_mms_khm_init_872h "$LOG"
echo "== ROUND4 DONE $(date -u)"
