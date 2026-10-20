# CLAUDE.md

Master's thesis project: fine-tuning and evaluating pretrained ASR models for Khmer.
Full brief: `khmer_asr_masters_thesis_brief.md` — read it for research questions, experiments (M0–M3) and risks.

## Commands

```bash
pip install -e ".[dev]"     # install package + dev deps (use a venv)
pytest                      # run tests
ruff check .                # lint
python scripts/<name>.py --help
```

## Layout

- `src/khmer_asr/` — package code (data, preprocessing, models, training, evaluation, inference). Put logic here; `scripts/` are thin CLI wrappers.
- `configs/*.yaml` — all experiment settings. Don't hardcode hyperparameters in code.
- `data/manifests/` — committed manifests and locked splits. `data/raw/`, `data/processed/` are never committed.
- `experiments/<run>/` — per-run config, env, metrics, predictions (see `experiments/README.md`).
- `docs/decisions.md` — log every protocol/design decision here.

## Research rules (non-negotiable)

- Never fabricate metrics, citations, dataset statistics, or GPU costs. If unknown, say so.
- Distinguish published results from our own experimental results.
- Verify dataset/model cards live before relying on them; record revisions (commit hashes).
- The test set (`data/manifests/test.csv`) is locked: never use it for model selection or tuning.
- Deduplicate before splitting; check split overlap with `khmer_asr.data.validation.find_split_overlap`.
- Change one major variable per experiment.
- CER is the primary metric. All scoring goes through `khmer_asr.evaluation.metrics` so normalization (`khmer_asr.preprocessing.text.normalize_khmer_text`) is identical everywhere. Changing normalization rules must be recorded in `docs/decisions.md`.
- Report WER only with a documented, consistent Khmer segmentation method.
- Record seed, config, `pip freeze`, GPU model, training time and cost for every run.

## Data and compute

- Dataset: `Digital-Divide-Data/khmer-speech-dataset` (~128 GB reported). Use streaming or small subsets; do not download the full dataset before the pilot succeeds.
- Respect the dataset license (reported CC BY-SA 4.0) and attribution requirements.
- Training runs on RunPod. Save checkpoints to persistent storage before stopping pods; remind the user to stop idle pods.
- Secrets live in `.env` (see `.env.example`); never commit them.

## Code style

- Python ≥3.10, type hints, line length 100 (ruff).
- Keep audio at 16 kHz unless the model requires otherwise.
- Add a unit test in `tests/` for any preprocessing, validation or metric logic.
