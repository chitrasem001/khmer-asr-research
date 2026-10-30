"""Khmer ASR Studio: a local demo of the thesis models.

    python -m uvicorn app.server:app --port 8000      (from the repo root)
    DEMO_DEVICE=cpu python -m uvicorn app.server:app  (keep the GPU free)

Compare tab: transcribe a test-set sample or an uploaded file with several models and show
character errors against the reference. Results tab: the numbers in experiments/*/metrics.json.
"""
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from khmer_asr.data.audio import TARGET_SR, clip_path, to_target_rate
from khmer_asr.evaluation.alignment import align
from khmer_asr.models.asr import ModelSpec, Transcriber
from khmer_asr.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "demo.yaml")
DEVICE = os.environ.get("DEMO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {m["id"]: m for m in CFG["models"]}
MAX_UPLOAD_S = 60

app = FastAPI(title="Khmer ASR Studio")
_loaded: dict[str, Transcriber] = {}
_lock = threading.Lock()  # one model call at a time: the laptop GPU has 8 GB


def _samples() -> pd.DataFrame:
    s = CFG["samples"]
    df = pd.read_csv(ROOT / s["manifest"])
    df = df[df["duration"].between(s["min_duration_s"], s["max_duration_s"])]
    picked = df.groupby("speaker_id").sample(n=s["per_speaker"], random_state=s["seed"])
    return picked.sort_values(["speaker_id", "utt_id"]).set_index("utt_id")


SAMPLES = _samples()


def _transcriber(model_id: str) -> Transcriber:
    if model_id not in MODELS:
        raise HTTPException(404, f"Unknown model {model_id}")
    if model_id not in _loaded:
        m = MODELS[model_id]
        spec = ModelSpec(
            m["name_or_path"], m.get("revision"), m["family"], m["language"],
            m.get("max_new_tokens", 440),
        )
        _loaded[model_id] = Transcriber(spec, device=DEVICE, fp16=DEVICE == "cuda")
    return _loaded[model_id]


@app.get("/api/models")
def models() -> dict:
    return {
        "device": DEVICE,
        "models": [
            {"id": m["id"], "label": m["label"], "color": m.get("color", "#3A4250"),
             "loaded": m["id"] in _loaded, "max_new_tokens": m.get("max_new_tokens")}
            for m in CFG["models"]
        ],
    }


@app.get("/api/samples")
def samples() -> list[dict]:
    return [
        {"utt_id": u, "speaker_id": r["speaker_id"], "duration": round(r["duration"], 2),
         "reference": r["text"]}
        for u, r in SAMPLES.iterrows()
    ]


@app.get("/api/audio/{utt_id}")
def audio(utt_id: str) -> FileResponse:
    if utt_id not in SAMPLES.index:  # only the listed samples are served
        raise HTTPException(404, "Unknown sample")
    return FileResponse(clip_path(ROOT / CFG["samples"]["audio_dir"], utt_id), media_type="audio/flac")


@app.post("/api/transcribe")
def transcribe(
    models: Annotated[str, Form()],
    utt_id: Annotated[str | None, Form()] = None,
    reference: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> dict:
    if utt_id:
        if utt_id not in SAMPLES.index:
            raise HTTPException(404, "Unknown sample")
        wav, _ = sf.read(clip_path(ROOT / CFG["samples"]["audio_dir"], utt_id), dtype="float32")
        reference = SAMPLES.loc[utt_id, "text"]
    elif file is not None:
        try:
            data, sr = sf.read(io.BytesIO(file.file.read()), dtype="float32")
        except Exception as err:  # soundfile raises several types for unreadable input
            raise HTTPException(400, f"Could not read audio: {err}") from err
        wav = to_target_rate(data, sr).astype("float32")
    else:
        raise HTTPException(400, "Send a sample id or an audio file")

    duration = len(wav) / TARGET_SR
    if duration > MAX_UPLOAD_S:
        raise HTTPException(400, f"Clip is {duration:.0f} s; the demo accepts up to {MAX_UPLOAD_S} s")

    results = []
    for model_id in [m for m in models.split(",") if m]:
        with _lock:
            t = _transcriber(model_id)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            hyp = t.transcribe([wav])[0]
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
        entry = {
            "id": model_id, "label": MODELS[model_id]["label"], "hypothesis": hyp,
            "latency_ms": round(seconds * 1000), "rtf": round(seconds / duration, 3),
        }
        if reference and reference.strip():
            try:
                entry["alignment"] = align(reference, hyp)
            except ValueError:
                pass
        results.append(entry)
    return {"duration": round(duration, 2), "reference": reference, "device": DEVICE, "results": results}


def _runs(folder: str) -> dict[str, dict]:
    runs = {}
    for metrics in sorted((ROOT / folder).glob("*/metrics.json")):
        info_path = metrics.with_name("run_info.json")
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        runs[metrics.parent.name] = {"metrics": json.loads(metrics.read_text()), "info": info}
    return runs


@app.get("/api/results")
def results() -> dict:
    rows = []
    for group in CFG["results"]:
        fleurs = _runs(group["fleurs_runs"])
        names = group.get("names", {})
        for name, run in _runs(group["test_runs"]).items():
            m = run["metrics"]
            f = fleurs.get(name, {}).get("metrics")
            rows.append({
                "model": names.get(name, name), "stage": group["stage"], "clips": m["clips"],
                "cer": m["cer"], "cer_with_spaces": m["cer_with_spaces"], "rtf": m["rtf"],
                "gpu": run["info"].get("gpu"),
                "params_m": m["parameters_millions"], "peak_vram_mb": m["peak_gpu_memory_mb"],
                "fleurs_cer": f["cer"] if f else None, "fleurs_clips": f["clips"] if f else None,
                "sub": m["substitution_rate"], "ins": m["insertion_rate"], "del": m["deletion_rate"],
            })
    return {"rows": sorted(rows, key=lambda r: r["cer"])}


app.mount("/", StaticFiles(directory=ROOT / "app" / "static", html=True), name="static")
