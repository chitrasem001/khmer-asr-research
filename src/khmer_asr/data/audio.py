"""Extract manifest clips from the remote parquet files into local FLAC files.

Each parquet file stores the audio of all its rows in one column chunk, so a file is read once
and only the rows listed in the manifest are written out. All clips are stored as 16 kHz mono:
the corpus is mostly 16 kHz, but the `*-of-00724` files are 48 kHz despite the dataset card.
"""
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from huggingface_hub import HfFileSystem

TARGET_SR = 16000


def clip_path(audio_dir: str | Path, utt_id: str) -> Path:
    """Local path of a clip; ':' is not allowed in Windows file names."""
    return Path(audio_dir) / f"{utt_id.replace(':', '__')}.flac"


def to_target_rate(data: np.ndarray, sr: int) -> np.ndarray:
    """Downmix to mono and resample to 16 kHz (no-op for 16 kHz mono input)."""
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        data = soxr.resample(data, sr, TARGET_SR, quality="HQ")
    return data


def _extract_file(dataset: str, revision: str, source_file: str, rows: pd.DataFrame, out: Path) -> int:
    todo = rows[[not clip_path(out, u).exists() for u in rows["utt_id"]]]
    if todo.empty:
        return 0
    fs = HfFileSystem()
    with fs.open(f"datasets/{dataset}@{revision}/data/{source_file}", "rb") as f:
        audio = pq.ParquetFile(f).read(columns=["audio"]).column("audio")
    for utt_id, row in zip(todo["utt_id"], todo["row_in_file"], strict=True):
        data, sr = sf.read(io.BytesIO(audio[int(row)].as_py()["bytes"]))
        target = clip_path(out, utt_id)
        tmp = target.with_suffix(".tmp")
        sf.write(tmp, to_target_rate(data, sr), TARGET_SR, format="FLAC")
        tmp.replace(target)  # atomic, so an interrupted run never leaves a half-written clip
    return len(todo)


def extract_clips(
    manifest: pd.DataFrame, dataset: str, revision: str, audio_dir: str | Path, workers: int = 4
) -> int:
    """Write every manifest clip to `audio_dir`; already extracted clips are skipped."""
    out = Path(audio_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = list(manifest.groupby("source_file"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        written = pool.map(lambda g: _extract_file(dataset, revision, g[0], g[1], out), groups)
        return sum(written)
