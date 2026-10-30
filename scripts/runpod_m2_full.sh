#!/usr/bin/env bash
# M2 + full-data run on one RunPod pod:
#   1. M2: MMS-1B with the adapter started from the released `khm` adapter, 10 h, same steps as
#      the M1 MMS pilot (1,000 steps = 3.5 passes), so only the initialisation changes.
#   2. Whisper small on the full training split for one pass (main in-domain model).
# Each run is scored on the test set and FLEURS, and its best/ weights are uploaded to the
# private HF repo when /workspace/.hf_write exists.
#
#   nohup bash scripts/runpod_m2_full.sh > /workspace/m2_full_launch.log 2>&1 &
#
# Needs /workspace/.hf_token (read). Everything a run produces (incl. console logs) ends up in
# /workspace/m2_full_results.tar.
set -uo pipefail   # no -e: one failed run must not stop the others
cd /workspace/khmer-asr-research
export HF_HOME=/workspace/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONIOENCODING=utf-8
export TRANSFORMERS_VERBOSITY=error HF_TOKEN="$(cat /workspace/.hf_token)"
LOG=experiments/m2_full_console.log
MODEL_REPO=chitra168/khmer-asr-thesis-models
mkdir -p experiments
exec > >(tee -a "$LOG") 2>&1

echo "== M2_FULL START $(date -u)"
git pull --ff-only
python -m pip install -q -e .   # the audio prep below needs the package before the first setup
bash scripts/runpod_setup.sh    # rebuilds train.parquet and checks the split hashes
echo "== PREPARE TEST + FLEURS AUDIO $(date -u)"
python -u scripts/extract_audio.py --manifest data/manifests/test.csv --workers 8
python -u scripts/prepare_fleurs.py --split test

# name | config | train.py args
RUNS=(
  "m2_mms_khm_init_10h|configs/m2_mms_khm_init.yaml|--train-hours 10 --max-steps 1000 --eval-steps 250 --warmup-steps 100"
  "f_whisper_small_872h|configs/m1_whisper_small.yaml|--max-steps 24000 --eval-steps 4000 --warmup-steps 100"
)

for spec in "${RUNS[@]}"; do
  IFS='|' read -r name cfg extra <<< "$spec"
  out="experiments/$name"
  echo "== RUN $name START $(date -u)"
  # shellcheck disable=SC2086  # extra holds several arguments
  if bash scripts/runpod_train.sh "$out" --config "$cfg" $extra; then
    family=$(grep -m1 "family:" "$cfg" | awk '{print $2}')
    lang=$([ "$family" = whisper ] && echo km || echo null)
    bs=$([ "$family" = whisper ] && echo 16 || echo 8)
    cat > "/workspace/eval_$name.yaml" <<EOF
experiment: {name: $name, seed: 42, output_dir: experiments/m1_eval}
models:
  - {name_or_path: $out/best, revision: null, family: $family, language: $lang, batch_size: $bs}
data:
  dataset: Digital-Divide-Data/khmer-speech-dataset
  revision: e872b0c58b4bbbe30c0244def61e39e7b57ea093
  test_manifest: data/manifests/test.csv
  audio_dir: data/processed/audio
  sampling_rate: 16000
inference: {fp16: true, decoding: greedy}
EOF
    echo "== EVAL $name TEST $(date -u)"
    python -u scripts/evaluate.py --config "/workspace/eval_$name.yaml"
    echo "== EVAL $name FLEURS $(date -u)"
    python -u scripts/evaluate.py --config "/workspace/eval_$name.yaml" \
      --manifest data/manifests/fleurs_km_test.csv --audio-dir data/processed/fleurs_audio \
      --output-dir experiments/m1_eval_fleurs
    if [ -f /workspace/.hf_write ]; then
      echo "== UPLOAD $name $(date -u)"
      HF_TOKEN="$(cat /workspace/.hf_write)" python - "$out" "$name" "$MODEL_REPO" <<'PY'
import sys
from huggingface_hub import HfApi
out, name, repo = sys.argv[1:]
HfApi().upload_folder(folder_path=out, path_in_repo=name, repo_id=repo, repo_type="model",
                      ignore_patterns=["checkpoints/*", "train_subset.csv"])
print("uploaded", name)
PY
    fi
    echo "== RUN $name DONE $(date -u)"
  else
    echo "== RUN $name FAILED $(date -u)"
  fi
  rm -rf "$out/checkpoints"   # keep only best/ to save disk; best/ is in the results tar
done

# The full run's train_subset.csv is the whole training split (rebuilt from configs/data.yaml and
# checked by hash), so it is not kept.
rm -f experiments/f_whisper_small_872h/train_subset.csv
echo "== PACK $(date -u)"
tar -cf /workspace/m2_full_results.tar experiments/m2_* experiments/f_* \
  experiments/m1_eval/m2_* experiments/m1_eval/f_* \
  experiments/m1_eval_fleurs/m2_* experiments/m1_eval_fleurs/f_* "$LOG"
echo "== M2_FULL DONE $(date -u)"
