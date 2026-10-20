# Khmer ASR Research

Master's thesis project: **An Experimental Study of Fine-Tuning Strategies for
Pretrained Models in Khmer Automatic Speech Recognition**.

See [`khmer_asr_masters_thesis_brief.md`](khmer_asr_masters_thesis_brief.md) for the full brief.

## Layout

| Path | Purpose |
|---|---|
| `configs/` | YAML configs for baseline (M0), fine-tuning (M1/M2) and evaluation |
| `data/raw/`, `data/processed/` | Local data, **not committed** |
| `data/manifests/` | Dataset manifests and locked split files (committed) |
| `data/samples/` | Small approved samples for tests and demos |
| `notebooks/` | Dataset audit, exploration, baseline evaluation |
| `src/khmer_asr/` | Installable package (data, preprocessing, models, training, evaluation, inference) |
| `scripts/` | CLI entry points |
| `tests/` | Unit tests (`pytest`) |
| `experiments/` | Per-run configs, logs and metrics |
| `docs/` | Proposal, literature review, experiment plan, decision log |
| `app/` | Demo speech-to-text application |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env             # then fill in tokens
pytest
```

## Workflow

1. `python scripts/audit_dataset.py` — audit the dataset (streaming, no full download).
2. `python scripts/prepare_dataset.py --config configs/finetune.yaml` — build manifests and lock splits.
3. `python scripts/evaluate.py --config configs/baseline.yaml` — M0 baseline.
4. `python scripts/train.py --config configs/finetune.yaml` — M1/M2 fine-tuning (RunPod).
5. `python scripts/evaluate.py --config configs/evaluation.yaml` — evaluate on the locked test set.
6. `python scripts/transcribe.py --audio path/to/file.wav` — single-file inference.

## Dataset

[Digital Divide Data — Khmer Speech Dataset](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset).
Verify the current dataset card, license (reported as CC BY-SA 4.0) and attribution
requirements before use.
