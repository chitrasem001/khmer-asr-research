# Khmer ASR Studio (demo app)

A local demo of the thesis models. Thesis results come from the locked test set
(`scripts/evaluate.py`), not from this demo; the demo uses the same normalization and CER code,
so its numbers match the paper for the same clip and model.

- **Compare** — pick a test-set clip (unseen speaker and sentence) or upload WAV/FLAC/OGG (≤ 60 s),
  transcribe with several models, see per-model CER, S/I/D counts, speed, and a colored
  character alignment (substitution / insertion / deletion).
- **Results** — the test-set and FLEURS numbers, read from `experiments/*/metrics.json`.

## Run (Python only)

```bash
pip install -e ".[demo]"
python -m uvicorn app.server:app --port 8000      # then open http://localhost:8000
DEMO_DEVICE=cpu python -m uvicorn app.server:app  # keep the GPU free
```

The built frontend is committed in `app/static/`, so running the demo needs no Node.js.

## Layout

| Path | What |
|---|---|
| `server.py` | FastAPI backend: `/api/models`, `/api/samples`, `/api/audio/{id}`, `/api/transcribe`, `/api/results` |
| `web/` | React + TypeScript + Vite source |
| `static/` | Build output of `web/` (served by the backend) |
| `../configs/demo.yaml` | Models shown and test samples offered |

Add a fine-tuned model by adding an entry to `configs/demo.yaml` (e.g. `name_or_path:
experiments/m1_mms/best`); no code changes.

## Change the frontend (needs Node.js)

```bash
cd app/web
npm install
npm run dev     # http://localhost:5173, forwards /api to the backend on :8000
npm run build   # type-check, then write the app to ../static (commit it)
```
