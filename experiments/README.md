# Experiments

One directory per run, e.g. `experiments/m1_finetune_10h_seed42/`, containing:

- `config.yaml` — exact config used
- `env.txt` — `pip freeze`, GPU model, CUDA/driver versions
- `metrics.json` — CER/WER, latency, model size, peak GPU memory
- `predictions.csv` — test-set references and hypotheses
- `notes.md` — training time, RunPod cost, observations

Checkpoints are not committed; store them on persistent RunPod storage or the HF Hub.

## Run log

| Run | Model | Train data | Change vs. previous | Test CER | Notes |
|---|---|---|---|---|---|
