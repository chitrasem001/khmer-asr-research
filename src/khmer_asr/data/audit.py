"""Dataset audit: load metadata columns remotely and summarize them.

Only non-audio columns are read, so the full corpus can be audited without downloading audio.
"""
import io
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, HfFileSystem

from khmer_asr.preprocessing.text import normalize_khmer_text

METADATA_COLUMNS = [
    "transcript",
    "speaker_id",
    "topic",
    "subtopic",
    "paragraph_id",
    "sentence_id",
    "duration",
]

# Khmer (U+1780–U+17FF) and Khmer Symbols (U+19E0–U+19FF) blocks.
_KHMER_RANGES = ((0x1780, 0x17FF), (0x19E0, 0x19FF))


def resolve_revision(dataset: str, revision: str | None = None) -> str:
    """Return the commit hash for a dataset revision, so audits are pinned."""
    return HfApi().dataset_info(dataset, revision=revision).sha


def list_parquet_files(dataset: str, revision: str) -> list[str]:
    fs = HfFileSystem()
    return sorted(fs.glob(f"datasets/{dataset}@{revision}/data/*.parquet"))


def _read_metadata(path: str) -> pd.DataFrame:
    fs = HfFileSystem()
    with fs.open(path, "rb") as f:
        df = pq.ParquetFile(f).read(columns=METADATA_COLUMNS).to_pandas()
    df["source_file"] = path.rsplit("/", 1)[-1]
    df["row_in_file"] = range(len(df))
    return df


def load_metadata(paths: list[str], workers: int = 8) -> pd.DataFrame:
    """Read metadata columns from all parquet files in parallel (no audio)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(_read_metadata, paths))
    return pd.concat(frames, ignore_index=True)


def check_audio_sample(path: str, max_rows: int = 20) -> list[dict]:
    """Decode the first rows of one file and compare real audio length with `duration`."""
    fs = HfFileSystem()
    with fs.open(path, "rb") as f:
        table = pq.ParquetFile(f).read_row_group(0, columns=["audio", "duration"])
    rows = []
    for audio, duration in zip(
        table.column("audio").to_pylist()[:max_rows],
        table.column("duration").to_pylist()[:max_rows],
        strict=True,
    ):
        data, sr = sf.read(io.BytesIO(audio["bytes"]))
        rows.append(
            {
                "sample_rate": sr,
                "channels": 1 if data.ndim == 1 else data.shape[1],
                "decoded_s": round(len(data) / sr, 3),
                "duration_col_s": duration,
            }
        )
    return rows


def non_khmer_chars(texts: pd.Series) -> dict[str, int]:
    """Count characters outside the Khmer blocks, excluding spaces."""
    counts: dict[str, int] = {}
    for text in texts:
        for ch in text:
            cp = ord(ch)
            if ch == " " or any(lo <= cp <= hi for lo, hi in _KHMER_RANGES):
                continue
            counts[ch] = counts.get(ch, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def char_inventory(texts: pd.Series) -> pd.DataFrame:
    """Every character in the raw transcripts: code point, Unicode name/category, counts."""
    chars: dict[str, list[int]] = {}
    for text in texts:
        for ch in text:
            chars.setdefault(ch, [0, 0])[0] += 1
        for ch in set(text):
            chars[ch][1] += 1
    rows = [
        {
            "char": ch,
            "codepoint": f"U+{ord(ch):04X}",
            "name": unicodedata.name(ch, "<unnamed>"),
            "category": unicodedata.category(ch),
            "occurrences": occ,
            "sentences": sent,
        }
        for ch, (occ, sent) in chars.items()
    ]
    return pd.DataFrame(rows).sort_values("codepoint").reset_index(drop=True)


# Non-standard Khmer character orders that make identical-looking text compare unequal.
ORDER_PATTERNS = {
    "coeng_ro_before_other_subscript": "\u17d2\u179a\u17d2[\u1780-\u17a2]",
    "vowel_before_subscript": "[\u17b6-\u17c5]\u17d2",
    "nikahit_before_u": "\u17c6\u17bb",
    "deprecated_chars": "[\u17a3\u17a4\u17b4\u17b5]",
}


def text_issues(texts: pd.Series) -> dict[str, int]:
    """Sentences affected by each formatting issue the normalization rules address."""
    counts = {
        name: int(texts.str.contains(pattern, regex=True).sum())
        for name, pattern in ORDER_PATTERNS.items()
    }
    counts.update(
        {
            "zero_width_chars": int(texts.str.contains("[\u200b\u200c\u200d\ufeff]").sum()),
            "latin_letters": int(texts.str.contains("[A-Za-z]").sum()),
            "arabic_digits": int(texts.str.contains("[0-9]").sum()),
            "khmer_digits": int(texts.str.contains("[\u17e0-\u17e9]").sum()),
            "newlines": int(texts.str.contains("\n").sum()),
            "double_spaces": int(texts.str.contains("  ").sum()),
        }
    )
    return counts


def speaker_table(df: pd.DataFrame) -> pd.DataFrame:
    """Utterances and hours per speaker; gender/age parsed from IDs like 'f-adt1-0001'."""
    table = (
        df.groupby("speaker_id")
        .agg(utterances=("duration", "size"), hours=("duration", lambda s: s.sum() / 3600))
        .reset_index()
        .sort_values("hours", ascending=False)
    )
    parts = table["speaker_id"].str.split("-", expand=True)
    table["gender_code"] = parts[0]
    table["group_code"] = parts[1] if parts.shape[1] > 1 else None
    table["hours"] = table["hours"].round(2)
    return table


def summarize(df: pd.DataFrame, min_s: float = 0.5, max_s: float = 30.0) -> dict:
    """Compute the audit statistics reported in the thesis."""
    dur = df["duration"].dropna()
    norm = df["transcript"].fillna("").map(normalize_khmer_text)
    lengths = norm.str.len()
    dup_mask = norm.duplicated(keep=False) & (norm != "")
    speakers_per_text = df.assign(_n=norm)[dup_mask].groupby("_n")["speaker_id"].nunique()
    per_sentence = df.assign(_n=norm)[norm != ""].groupby("_n")["speaker_id"]

    def dist(series: pd.Series) -> dict[str, int]:
        return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}

    def q(series: pd.Series, p: float) -> float:
        return round(float(np.quantile(series, p)), 3)

    return {
        "rows": len(df),
        "total_hours": round(float(dur.sum()) / 3600, 2),
        "duration_s": {
            "missing": int(df["duration"].isna().sum()),
            "min": round(float(dur.min()), 3),
            "p05": q(dur, 0.05),
            "median": q(dur, 0.5),
            "mean": round(float(dur.mean()), 3),
            "p95": q(dur, 0.95),
            "max": round(float(dur.max()), 3),
            f"below_{min_s}s": int((dur < min_s).sum()),
            f"above_{max_s}s": int((dur > max_s).sum()),
            f"hours_above_{max_s}s": round(float(dur[dur > max_s].sum()) / 3600, 2),
        },
        "transcript_chars": {
            "empty": int((lengths == 0).sum()),
            "median": q(lengths, 0.5),
            "max": int(lengths.max()),
        },
        "duplicates": {
            "rows_with_duplicated_transcript": int(dup_mask.sum()),
            "unique_duplicated_transcripts": int(norm[dup_mask].nunique()),
            "duplicated_transcripts_read_by_2plus_speakers": int((speakers_per_text > 1).sum()),
            "duplicated_sentence_ids": int(df["sentence_id"].duplicated().sum()),
            "unique_sentences": int(norm[norm != ""].nunique()),
            "readings_per_sentence": dist(per_sentence.size()),
            "speakers_per_sentence": dist(per_sentence.nunique()),
        },
        "speakers": int(df["speaker_id"].nunique()),
        "topics": int(df["topic"].nunique()),
        "subtopics": int(df["subtopic"].nunique()),
        "top_non_khmer_chars": dict(list(non_khmer_chars(norm).items())[:30]),
    }
