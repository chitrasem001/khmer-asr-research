# Decision Log

Record each significant decision: date, decision, alternatives considered, rationale.

| Date | Decision | Alternatives | Rationale |
|---|---|---|---|
| 2026-10-22 | Plan for submission by end of Feb 2027 | Wait for an official date | No official deadline yet; last year's was ~March. Finish a month early for buffer. **Confirm with faculty.** |
| 2026-10-22 | RunPod budget cap $200 total, ≤ $25 per run, alert at $150 | $100 (too tight for M2 + scaling); $500+ (not justified before pilot) | Estimate: ~200 GPU-h at $0.33–0.59/h (48 GB A6000/A40) ≈ $70–120, plus network volume (~100 GB × $0.07 × 5 mo ≈ $35), plus ~30% buffer. GPU-hours are an **assumption** until the pilot measures real throughput. Prices from runpod.io/pricing, checked 2026-10-22. |
| 2026-10-22 | Record of earlier RunPod usage (reported by user): RTX 4090 at $0.74/h, ~7–7.5 h (~$5.20–5.55); ~10 h figure also reported (~$7.40) | — | Real price paid, used as a reference point. Workload (model, data hours, epochs) not yet recorded, so it can't serve as a throughput estimate yet. At $0.74/h, the $200 cap ≈ 270 GPU-hours on a 4090. |
| 2026-10-22 | Local environment: Python 3.12.10, torch 2.6.0+cu124, transformers 5.20.2, datasets 5.1.0, jiwer 4.0.0; full pins in `requirements.lock.txt` | Unpinned `requirements.txt` only | Reproducibility. Local RTX 4060 Laptop (8 GB) for development and small-model evaluation; RunPod for fine-tuning. |
| 2026-10-22 | Default GPU: 48 GB (RTX A6000 or A40); 24 GB (RTX 4090/A5000) for small models/eval | A100/H100 80 GB ($1.19–3.99/h) | 48 GB is enough to fine-tune Whisper small/medium with fp16, and large models with LoRA, at a fraction of A100 cost. Use Community Cloud for experiments and Secure Cloud only if availability is a problem. |
