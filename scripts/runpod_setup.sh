#!/usr/bin/env bash
# Prepare a RunPod pod for training: clone the repo, install, rebuild and verify the split.
#
#   bash runpod_setup.sh            (run on the pod, from /workspace)
#
# The training manifest (train.parquet) is not in git. It is rebuilt from the dataset metadata and
# checked against the hash recorded in data/manifests/split_report.json, which also proves the
# split is reproducible on another machine.
set -euo pipefail

REPO_URL="https://github.com/chitrasem001/khmer-asr-research.git"
WORKDIR="${WORKDIR:-/workspace}"
export HF_HOME="$WORKDIR/.cache/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd "$WORKDIR"
if [ -d khmer-asr-research ]; then
  git -C khmer-asr-research pull --ff-only
else
  git clone "$REPO_URL"
fi
cd khmer-asr-research

python -m pip install -q --upgrade pip
python -m pip install -q -e .
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', '| transformers', transformers.__version__)"

if [ ! -f data/processed/metadata.parquet ]; then
  python scripts/audit_dataset.py --revision e872b0c58b4bbbe30c0244def61e39e7b57ea093 --workers 32 \
    --output "$WORKDIR/audit_report_pod.json"
fi
# Rebuild the split; keep the committed reports untouched and compare hashes.
cp data/manifests/split_report.json "$WORKDIR/split_report_committed.json"
python scripts/prepare_dataset.py --config configs/data.yaml
python - "$WORKDIR/split_report_committed.json" <<'PY'
import json, sys
committed = json.load(open(sys.argv[1], encoding="utf-8"))
rebuilt = json.load(open("data/manifests/split_report.json", encoding="utf-8"))
for key in ("test_csv_sha256", "train_utt_ids_sha256"):
    ok = committed[key] == rebuilt[key]
    print(f"{key}: {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit(f"Rebuilt split differs from the committed one ({key}); stop and investigate.")
PY
git checkout -- data/manifests/   # keep the working tree identical to the commit
echo "Pod ready."
