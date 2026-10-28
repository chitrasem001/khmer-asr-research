# M1 MMS (10 h pilot) — pod log summary

**Reconstructed.** The raw console log (`/workspace/run_m1.log` on the pod) was not copied before the
pod was deleted. The lines below were captured from that log during the run (milestone monitor and
manual checks). Per-step training loss and all validation results are in `log_history.json`.

Pod: `khmer-asr-m1-mms` (`oiipmd6850cwzn`), 1× NVIDIA A40 48 GB, SECURE cloud, $0.59/h,
image `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`. Created 2026-10-28 21:11 (UTC+7),
deleted 22:45. Account balance 16.29 → 15.36 USD (**0.93 USD** total, incl. setup and two restarts).

## Timeline (UTC)

| Time | Event |
|---|---|
| 14:11 | Pod created; image pull took about 5 min |
| 14:28 | First start: setup re-audit hit Hugging Face rate limits (HTTP 429) without a token → stopped |
| 14:32 | `metadata.parquet` sent from the laptop (sha256 c8b8114f…, identical on both sides) |
| 14:37 | Restart with HF token: `test_csv_sha256: MISMATCH` — CRLF vs LF only (content identical); fixed in commit 4b8909a |
| 14:42:04 | `== START` — torch 2.9.1+cu128, CUDA True, NVIDIA A40, transformers 5.21.0 |
| 14:42 | `test_csv_sha256: MATCH`, `train_utt_ids_sha256: MATCH` (split rebuilt on Linux = laptop split) |
| 14:42:42 | `== TRAIN` — train: 4561 clips, 10.40 h from 51 files; eval: 1288 clips |
| 14:47 | `audio ready in 262 s`; 0 rate-limit retries with the token |
| 14:47 | `setup: {'vocab_size': 111, 'trainable_parameters': 2295921}`; lm_head re-initialised (154 → 113 outputs, new vocabulary — expected) |
| 14:47–15:39 | Training, 2.2–3.5 s/step at batch 16; peak GPU memory 16.1 GB allocated |
| 15:39:41 | `== DONE` — best eval CER 0.0885 at checkpoint-1000; training 51.4 min |

## Validation CER (1,288 clips; zero-shot MMS on the same clips: 10.20%)

| Step | Epoch | eval_cer | eval_loss |
|---|---|---|---|
| 250 | 0.87 | 0.1093 | 0.3837 |
| 500 | 1.75 | 0.0970 | 0.3360 |
| 750 | 2.62 | 0.0913 | 0.3195 |
| 1000 | 3.50 | **0.0885** | 0.3083 |

Command: `python scripts/train.py --config configs/m1_mms.yaml --train-hours 10 --max-steps 1000 --eval-steps 250`
(`gpu_price_per_hour` set to 0.59 on the pod). Commit on the pod: see `run_info.json` (`git_commit`).
