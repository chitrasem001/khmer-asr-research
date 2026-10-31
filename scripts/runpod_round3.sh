#!/usr/bin/env bash
# Round 3 on one RunPod pod, in priority order:
#   C. Score test_seen_speakers (unseen sentences, known speakers) with existing models.
#   B. Second training seed (43) for the 10 h MMS runs, M1 (scratch) and M2 (khm init): same
#      data subset, only initialisation and data order change.
#   A. M2 at 100 h (MMS from the released khm adapter), same steps as s_mms_100h.
#   D. Second pass for Whisper small on the full split, continuing from f_whisper_small_872h.
# Every new model is scored on test, FLEURS and test_seen; best/ weights go to the private HF repo.
#
#   nohup bash scripts/runpod_round3.sh > /workspace/round3_launch.log 2>&1 &
#
# Needs /workspace/.hf_token (read) and /workspace/.hf_write (write). Results without weights end
# up in /workspace/round3_results.tar.
set -uo pipefail   # no -e: one failed run must not stop the others
cd /workspace/khmer-asr-research
export HF_HOME=/workspace/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONIOENCODING=utf-8
export TRANSFORMERS_VERBOSITY=error HF_TOKEN="$(cat /workspace/.hf_token)"
LOG=experiments/round3_console.log
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

echo "== ROUND3 START $(date -u)"
git pull --ff-only
python -m pip install -q -e .
bash scripts/runpod_setup.sh    # rebuilds train.parquet and checks the split hashes
echo "== PREPARE EVAL AUDIO $(date -u)"
python -u scripts/extract_audio.py --manifest data/manifests/test.csv --workers 8
python -u scripts/extract_audio.py --manifest data/manifests/test_seen_speakers.csv --workers 8
python -u scripts/prepare_fleurs.py --split test

echo "== DOWNLOAD EXISTING MODELS $(date -u)"
python - <<'PY'
from huggingface_hub import snapshot_download
runs = ["f_whisper_small_872h", "s_whisper_small_100h", "s_mms_100h", "m2_mms_khm_init_10h"]
snapshot_download("chitra168/khmer-asr-thesis-models", local_dir="experiments",
                  allow_patterns=[f"{r}/best/*" for r in runs])
print("downloaded", runs)
PY

# ---- C: test_seen_speakers for existing models (test and FLEURS are already scored)
evaluate mms-1b-all facebook/mms-1b-all 3d33597edbdaaba14a8e858e2c8caa76e3cec0cd mms khm seen
for spec in "s_mms_100h|mms|null" "m2_mms_khm_init_10h|mms|null" \
            "s_whisper_small_100h|whisper|km" "f_whisper_small_872h|whisper|km"; do
  IFS='|' read -r name family lang <<< "$spec"
  evaluate "$name" "experiments/$name/best" null "$family" "$lang" seen
done

# ---- B, A, D: name | config | train.py args
RUNS=(
  "m1_mms_seed43|configs/m1_mms.yaml|--train-hours 10 --max-steps 1000 --eval-steps 250 --warmup-steps 100 --train-seed 43"
  "m2_mms_khm_init_10h_seed43|configs/m2_mms_khm_init.yaml|--train-hours 10 --max-steps 1000 --eval-steps 250 --warmup-steps 100 --train-seed 43"
  "m2_mms_khm_init_100h|configs/m2_mms_khm_init.yaml|--train-hours 100 --max-steps 9485 --eval-steps 2372 --warmup-steps 100"
  "f_whisper_small_872h_pass2|configs/f_whisper_small_872h_pass2.yaml|--max-steps 24000 --eval-steps 4000 --warmup-steps 100"
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
rm -f experiments/f_whisper_small_872h_pass2/train_subset.csv
echo "== PACK $(date -u)"
tar -cf /workspace/round3_results.tar --exclude='*.safetensors' --exclude='*/checkpoints' \
  --exclude='experiments/f_whisper_small_872h' --exclude='experiments/s_*' \
  --exclude='experiments/m2_mms_khm_init_10h' \
  experiments/m1_mms_seed43 experiments/m2_mms_khm_init_* experiments/f_whisper_small_872h_pass2 \
  experiments/m1_eval experiments/m1_eval_fleurs experiments/m1_eval_seen "$LOG"
echo "== ROUND3 DONE $(date -u)"
