#!/usr/bin/env bash
# Wait for the M2/full job (or its death, or a hard deadline), back up results to the private HF
# repo (sha256 verified), then STOP (not delete) this pod so it does not bill GPU time unattended.
LOG=/workspace/khmer-asr-research/experiments/m2_full_console.log
DEADLINE=$(date -u -d "2026-10-31 01:00" +%s)
exec >> /workspace/autostop.log 2>&1
echo "== AUTOSTOP armed $(date -u) (deadline 2026-10-31 01:00 UTC)"
dead=0
while true; do
  grep -q "== M2_FULL DONE" "$LOG" 2>/dev/null && { echo "== job done $(date -u)"; break; }
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && { echo "== DEADLINE reached $(date -u)"; break; }
  if pgrep -f runpod_m2_full.sh > /dev/null; then dead=0; else dead=$((dead + 1)); fi
  [ "$dead" -ge 5 ] && { echo "== job script not running for 5 min $(date -u)"; break; }
  sleep 60
done
cd /workspace
if [ ! -f m2_full_results.tar ]; then   # job did not pack (died or deadline): pack what exists
  tar -cf m2_full_results.tar khmer-asr-research/experiments/m2_* khmer-asr-research/experiments/f_* \
    khmer-asr-research/experiments/m1_eval khmer-asr-research/experiments/m1_eval_fleurs \
    khmer-asr-research/experiments/m2_full_console.log --exclude='*/checkpoints' 2>/dev/null
fi
sha256sum m2_full_results.tar | tee m2_full_results.tar.sha256
HF_TOKEN="$(cat /workspace/.hf_write)" timeout 3600 python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo = "chitra168/khmer-asr-thesis-models"
for f in ("m2_full_results.tar", "m2_full_results.tar.sha256", "m2_full_launch.log", "autostop.log"):
    api.upload_file(path_or_fileobj=f"/workspace/{f}", path_in_repo=f"pod_backups/m2_full/{f}", repo_id=repo)
local = open("/workspace/m2_full_results.tar.sha256").read().split()[0]
info = api.get_paths_info(repo, ["pod_backups/m2_full/m2_full_results.tar"], expand=True)[0]
remote = info.lfs.sha256 if info.lfs else None
print("sha256 local ", local)
print("sha256 remote", remote)
print("BACKUP_VERIFIED" if local == remote else "BACKUP_MISMATCH")
PY
echo "== stopping pod $(date -u)"
curl -s -X POST -o /dev/null -w "stop http %{http_code}\n" -H "Authorization: Bearer $(cat /workspace/.runpod_key)" \
  https://rest.runpod.io/v1/pods/3kbasyt0hig9ow/stop
