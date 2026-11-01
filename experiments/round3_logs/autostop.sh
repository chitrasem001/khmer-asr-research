#!/usr/bin/env bash
# Wait for the round-3 job (or its death, or a hard deadline), back up results without weights
# to the private HF repo (sha256 verified), then STOP (not delete) this pod.
LOG=/workspace/khmer-asr-research/experiments/round3_console.log
DEADLINE=$(date -u -d "2026-11-01 05:00" +%s)
exec >> /workspace/autostop.log 2>&1
echo "== AUTOSTOP armed $(date -u) (deadline 2026-11-01 05:00 UTC)"
dead=0
sleep 600   # give the job time to start
while true; do
  grep -q "== ROUND3 DONE" "$LOG" 2>/dev/null && { echo "== job done $(date -u)"; break; }
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && { echo "== DEADLINE reached $(date -u)"; break; }
  if pgrep -f runpod_round3.sh > /dev/null; then dead=0; else dead=$((dead + 1)); fi
  [ "$dead" -ge 5 ] && { echo "== job script not running for 5 min $(date -u)"; break; }
  sleep 60
done
cd /workspace
if [ ! -f round3_results.tar ]; then   # job did not pack (died or deadline): pack what exists
  cd khmer-asr-research
  tar -cf /workspace/round3_results.tar --exclude='*.safetensors' --exclude='*/checkpoints' \
    experiments/m1_mms_seed43 experiments/m2_mms_khm_init_10h_seed43 experiments/m2_mms_khm_init_100h \
    experiments/f_whisper_small_872h_pass2 experiments/m1_eval experiments/m1_eval_fleurs \
    experiments/m1_eval_seen experiments/round3_console.log 2>/dev/null
  cd /workspace
fi
sha256sum round3_results.tar | tee round3_results.tar.sha256
HF_TOKEN="$(cat /workspace/.hf_write)" timeout 3600 python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo = "chitra168/khmer-asr-thesis-models"
for f in ("round3_results.tar", "round3_results.tar.sha256", "round3_launch.log", "autostop.log"):
    api.upload_file(path_or_fileobj=f"/workspace/{f}", path_in_repo=f"pod_backups/round3/{f}", repo_id=repo)
local = open("/workspace/round3_results.tar.sha256").read().split()[0]
info = api.get_paths_info(repo, ["pod_backups/round3/round3_results.tar"], expand=True)[0]
remote = info.lfs.sha256 if info.lfs else None
print("sha256 local ", local)
print("sha256 remote", remote)
print("BACKUP_VERIFIED" if local == remote else "BACKUP_MISMATCH")
PY
echo "== stopping pod $(date -u)"
curl -s -X POST -o /dev/null -w "stop http %{http_code}\n" -H "Authorization: Bearer $(cat /workspace/.runpod_key)" \
  https://rest.runpod.io/v1/pods/lngxij6peq67g9/stop
