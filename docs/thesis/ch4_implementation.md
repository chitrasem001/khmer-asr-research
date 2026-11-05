# Chapter 4 — Implementation and Experimental Setup

Draft. Configurations are in `configs/`, run records in `experiments/<run>/run_info.json`, and
design decisions with dates in `docs/decisions.md`.

---

## 4.1 Software

The code is a Python package (`src/khmer_asr`) with thin command-line scripts (`scripts/`):

| Component | Module / script | Role |
|---|---|---|
| Audit | `data/audit.py`, `scripts/audit_dataset.py` | Metadata-only audit of all files |
| Split | `data/split.py`, `scripts/prepare_dataset.py` | Filtering, leakage-free split, hashes |
| Audio | `data/audio.py`, `scripts/extract_audio.py` | Extract clips to 16 kHz FLAC, resampling the 48 kHz batch |
| Subsets | `data/subsets.py` | Nested whole-file training subsets |
| Normalization | `preprocessing/text.py` | Protocol v1 |
| Training | `training/data.py`, `training/finetune.py`, `scripts/train.py` | Datasets, collators, model setup, one-command runs |
| Evaluation | `evaluation/metrics.py`, `evaluation/run.py`, `scripts/evaluate.py` | CER, error rates, RTF, predictions |
| Analysis | `evaluation/error_analysis.py`, `scripts/error_analysis.py` | Character-level error analysis |
| Demo | `app/` | Web demo (FastAPI + React) comparing models on test clips or uploads |

Main libraries: PyTorch, Hugging Face Transformers, jiwer (alignment and error counts), soxr
(resampling), pandas. Exact versions are pinned in `requirements.lock.txt` and recorded per run
(`pip_freeze.txt`): on the laptop Python 3.12.10, PyTorch 2.6.0 (CUDA 12.4), Transformers 5.20.2; on
the cloud GPU Python 3.12.3, PyTorch 2.9.1 (CUDA 12.8), Transformers 5.21.0. The package has 49 unit
tests and passes a linter (ruff). Every run records its git commit, seed, configuration, GPU, time
and cost.

## 4.2 Hardware

- **Development and zero-shot evaluation:** a laptop with an NVIDIA GeForce RTX 4060 Laptop GPU
  (8 GB).
- **Fine-tuning:** one NVIDIA A40 (48 GB) rented from RunPod (secure cloud, 0.59 USD per hour).
  A pod is created per round of experiments; it rebuilds and hash-checks the split, extracts the
  audio it needs, runs the experiments in sequence, scores every model, uploads the weights to a
  private model repository, backs up all logs with a checksum, and stops itself.

## 4.3 Models

*Table 4.1. Pretrained models (revisions pinned in `configs/baseline.yaml`).*

| Model | Family | Parameters | Licence | Role |
|---|---|---|---|---|
| Whisper small | Encoder-decoder | 242 M | Apache-2.0 | Zero-shot, fine-tuned |
| Whisper medium | Encoder-decoder | 764 M | Apache-2.0 | Zero-shot |
| Whisper large-v3-turbo | Encoder-decoder | 809 M | Apache-2.0 | Zero-shot |
| MMS-1B-all (Khmer adapter) | CTC | 965 M | CC BY-NC 4.0 | Zero-shot, fine-tuned |

Whisper small and MMS-1B were chosen for fine-tuning: MMS is the best zero-shot model, and Whisper
small fails zero-shot exactly as the larger Whisper models do but is the cheapest to train.

## 4.4 Zero-shot evaluation (M0)

Whisper is forced to transcribe Khmer (language `km`, task `transcribe`); MMS loads its released
Khmer adapter (`khm`). Decoding is greedy in fp16. Zero-shot Whisper repeats a character until its
440-token limit on almost every clip, which would make the full test set take about 16 GPU-hours on
the laptop, so Whisper is scored on a seeded sample of 100 test clips per speaker (400 clips) and 200
FLEURS clips; MMS is scored on the full sets. A diagnostic run on 40 test clips (10 per speaker) checks
language identification without forcing, and whether fp32 decoding gives the same output as fp16.

## 4.5 Fine-tuning setup

**Whisper small — full fine-tuning.** All weights are trained with learning rate 1e-5, the decoder
prompt fixed to Khmer transcription. Whisper's decoder has 448 positions, so training and validation
transcripts longer than that are dropped (31 training clips at 10 h, 359 at 100 h, 3,447 in the full
split; 6 validation clips). The test set keeps every clip; the 24 test transcripts that exceed the
limit count against Whisper.

**MMS-1B — adapter training.** The 1B-parameter base is frozen; only the language adapter layers and
the output layer are trained (about 2.3 M parameters), with learning rate 1e-3 as in the MMS adapter
recipe. The released Khmer vocabulary has no symbols for the digits ៤ ៦ ៨ ៩, so the output layer uses
a new character vocabulary built from the training text (111 characters at 10 h, 116 at 50 and 100 h).

**M2 — adapter initialized from the released Khmer adapter.** Identical to the MMS runs except that,
before training, the 288 adapter tensors are copied from the released `khm` adapter, and every
output-layer row for a character that the released vocabulary also contains is copied (105 of 111
rows at 10 h, 110 of 116 at 100 h). The rest — the word delimiter, è and the four missing digits —
keep their new initialization. As a check, the initialized model scores 10.07% CER on 40 validation
clips before any training (zero-shot MMS: 10.2% on the full validation set), whereas a new output
layer scores 249.71%.

**Common settings.** Batch size 16, 100 warm-up steps then linear decay, fp16 mixed precision,
gradient checkpointing, seed 42. Each subset is trained for **3.5 passes** (1,000 / 4,638 / 9,485
steps for 10 / 50 / 100 h), so only the amount of data changes between sizes. The full-data Whisper
run makes **one pass** (24,000 steps) to fit the budget. Validation runs 4 times per subset run (every 250
steps at 10 h, 1,160 at 50 h, 2,372 at 100 h) and 6 times in the full run (every 4,000 steps); the checkpoint with the lowest
validation CER is kept. Validation uses all 1,288 clips for MMS and a 300-clip sample for Whisper,
whose autoregressive decoding is slower.

**Second seed.** The two 10-hour MMS runs are repeated with training seed 43 on the same data subset
(the subset is drawn with the experiment seed, 42, so only initialization and data order change).

## 4.6 Runs, time and cost

*Table 4.2. Fine-tuning runs on one A40 (training time excludes audio extraction and evaluation;
cost = (extraction + training) × 0.59 USD/h).*

| Run | Data | Steps | Best validation CER | Training (min) | Cost (USD) |
|---|---|---|---|---|---|
| Whisper small | 10 h | 1,000 | 17.86% | 18 | 0.18 |
| Whisper small | 50 h | 4,638 | 3.81% | 59 | 0.63 |
| Whisper small | 100 h | 9,485 | 2.40% | 113 | 1.14 |
| Whisper small | 872 h | 24,000 | 1.26% | 282 | 3.22 |
| MMS-1B, new adapter | 10 h | 1,000 | 8.85% | 51 | 0.55 |
| MMS-1B, new adapter | 50 h | 4,638 | 6.38% | 234 | 2.37 |
| MMS-1B, new adapter | 100 h | 9,485 | 5.33% | 464 | 4.57 |
| MMS-1B, M2 | 10 h | 1,000 | 7.82% | 51 | 0.55 |
| MMS-1B, M2 | 100 h | 9,485 | 4.54% | 469 | 4.67 |
| MMS-1B, new adapter, seed 43 | 10 h | 1,000 | 8.85% | 52 | 0.58 |
| MMS-1B, M2, seed 43 | 10 h | 1,000 | 7.68% | 52 | 0.51 |

Training and audio extraction for the 11 runs cost 18.97 USD; including pod setup, evaluation and
storage, the cloud bill for all fine-tuning was about 23 USD of the 200 USD budget. Peak GPU memory
was 8.6 GB for Whisper small and 16.1–16.7 GB for MMS-1B, so every run fits a 24 GB GPU as well.
MMS trains only 2.3 M parameters but is slower per step than Whisper small, because every step runs
its 1B-parameter encoder.

## 4.7 Evaluation of fine-tuned models

Every fine-tuned model is scored with the same script as the zero-shot baselines (greedy decoding,
fp16; batch 16 for Whisper, 8 for MMS) on the test set, FLEURS and the seen-speaker test set, and its
per-clip predictions are saved. All bootstrap intervals, paired tests and the error analysis are
computed from these saved predictions, so they need no further GPU time. Speed (RTF) is recorded but
not compared across runs, because models were scored on different GPUs (laptop for zero-shot and the
first 10-hour models, A40 for the rest).

## 4.8 Reproducibility

- The dataset, every model and FLEURS are pinned to revisions; the split is rebuilt from
  configuration and verified by hash.
- Every run directory holds its resolved configuration, console log, training log
  (`log_history.json`), run record (`run_info.json`) and package list; all are in git. Model weights
  (0.9–3.6 GB each) are in a private model repository and are not committed.
- Unattended cloud runs back up their results with a SHA-256 checksum before the pod stops; results
  are copied back and the checksum verified before a pod is deleted.
- Code: github.com/chitrasem001/khmer-asr-research.
