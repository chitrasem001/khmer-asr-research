"""Run a transcriber over a manifest and compute the thesis metrics."""
import time

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from khmer_asr.data.audio import clip_path
from khmer_asr.evaluation.metrics import cer
from khmer_asr.models.asr import Transcriber
from khmer_asr.preprocessing.text import normalize_khmer_text, remove_spaces


def transcribe_manifest(
    transcriber: Transcriber, manifest: pd.DataFrame, audio_dir: str, batch_size: int
) -> tuple[pd.DataFrame, float]:
    """Return predictions and total model time in seconds (audio loading excluded)."""
    # Sort by length so batches pad little; restore manifest order afterwards.
    order = manifest.sort_values("duration", ascending=False)
    hyps: dict[str, str] = {}
    model_seconds = 0.0
    for start in range(0, len(order), batch_size):
        batch = order.iloc[start : start + batch_size]
        audio = [sf.read(clip_path(audio_dir, u), dtype="float32")[0] for u in batch["utt_id"]]
        if transcriber.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        texts = transcriber.transcribe(audio)
        if transcriber.device == "cuda":
            torch.cuda.synchronize()
        model_seconds += time.perf_counter() - t0
        hyps.update(zip(batch["utt_id"], texts, strict=True))
    out = manifest[["utt_id", "speaker_id", "duration", "text"]].rename(columns={"text": "reference"})
    out["hypothesis"] = out["utt_id"].map(hyps)
    return out, model_seconds


def error_breakdown(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Substitution / deletion / insertion rates over space-free normalized characters."""
    refs = [remove_spaces(normalize_khmer_text(r)) for r in references]
    hyps = [remove_spaces(normalize_khmer_text(h)) for h in hypotheses]
    out = jiwer.process_characters(refs, hyps)
    n = out.hits + out.substitutions + out.deletions
    return {
        "substitution_rate": out.substitutions / n,
        "deletion_rate": out.deletions / n,
        "insertion_rate": out.insertions / n,
        "reference_chars": n,
    }


def compute_metrics(predictions: pd.DataFrame, model_seconds: float, transcriber: Transcriber) -> dict:
    refs = predictions["reference"].tolist()
    hyps = predictions["hypothesis"].fillna("").tolist()
    audio_seconds = float(predictions["duration"].sum())
    per_speaker = {
        spk: round(cer(g["reference"].tolist(), g["hypothesis"].fillna("").tolist()), 4)
        for spk, g in predictions.groupby("speaker_id")
    }
    per_clip = np.array([cer([r], [h]) for r, h in zip(refs, hyps, strict=True)])
    return {
        "cer": cer(refs, hyps),
        "cer_with_spaces": cer(refs, hyps, ignore_spaces=False),
        **error_breakdown(refs, hyps),
        "cer_by_speaker": per_speaker,
        "clips_with_cer_over_100pct": int((per_clip > 1.0).sum()),
        "empty_hypotheses": int((predictions["hypothesis"].fillna("").str.strip() == "").sum()),
        "clips": len(predictions),
        "audio_hours": round(audio_seconds / 3600, 3),
        "model_seconds": round(model_seconds, 1),
        "rtf": model_seconds / audio_seconds,
        "parameters_millions": round(transcriber.num_parameters / 1e6, 1),
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated() / 2**20) if transcriber.device == "cuda" else None
        ),
    }
