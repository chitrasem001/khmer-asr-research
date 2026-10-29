#!/usr/bin/env bash
# Data-scaling experiment (RQ2) on one RunPod pod: 50 h and 100 h for Whisper small and MMS-1B,
# each trained for 3.5 passes over its subset (same as the 10 h pilots), then scored on the test
# set and FLEURS. Subsets are nested (10 h ⊂ 50 h ⊂ 100 h, seed 42).
#
#   nohup bash scripts/runpod_scaling.sh > /workspace/scaling_launch.log 2>&1 &
#
# Needs /workspace/.hf_token and data/processed/metadata.parquet (sent from the laptop).
# Everything a run produces (incl. console logs) ends up in /workspace/scaling_results.tar.
set -uo pipefail   # no -e: one failed run must not stop the others
cd /workspace/khmer-asr-research
export HF_HOME=/workspace/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONIOENCODING=utf-8
export TRANSFORMERS_VERBOSITY=error HF_TOKEN="$(cat /workspace/.hf_token)"
LOG=experiments/scaling_console.log
mkdir -p experiments
exec > >(tee -a "$LOG") 2>&1

echo "== SCALING START $(date -u)"
git pull --ff-only
python -m pip install -q -e .   # the audio prep below needs the package before the first setup
echo "== PREPARE TEST + FLEURS AUDIO $(date -u)"
python -u scripts/extract_audio.py --manifest data/manifests/test.csv --workers 8
python -u scripts/prepare_fleurs.py --split test

# name | config | hours | steps (3.5 passes) | eval/save every
RUNS=(
  "s_whisper_small_50h|configs/m1_whisper_small.yaml|50|4638|1160"
  "s_whisper_small_100h|configs/m1_whisper_small.yaml|100|9485|2372"
  "s_mms_50h|configs/m1_mms.yaml|50|4638|1160"
  "s_mms_100h|configs/m1_mms.yaml|100|9485|2372"
)

for spec in "${RUNS[@]}"; do
  IFS='|' read -r name cfg hours steps every <<< "$spec"
  out="experiments/$name"
  echo "== RUN $name START $(date -u)"
  if bash scripts/runpod_train.sh "$out" --config "$cfg" --train-hours "$hours" \
       --max-steps "$steps" --eval-steps "$every" --warmup-steps 100; then
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
    echo "== RUN $name DONE $(date -u)"
  else
    echo "== RUN $name FAILED $(date -u)"
  fi
  rm -rf "$out/checkpoints"   # keep only best/ to save disk; best/ is in the results tar
done

echo "== PACK $(date -u)"
tar -cf /workspace/scaling_results.tar experiments/s_* experiments/m1_eval experiments/m1_eval_fleurs "$LOG"
echo "== SCALING DONE $(date -u)"
