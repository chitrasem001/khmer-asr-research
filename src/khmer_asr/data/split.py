"""Filter the corpus and build speaker- and sentence-disjoint splits.

Every sentence in the corpus is read by several speakers, so held-out speakers alone would still
leak sentences. Rule: test speakers never appear in training, and no sentence read by a test speaker
appears in training. Training speakers' readings of those sentences form `test_seen_speakers`.
"""
import numpy as np
import pandas as pd

from khmer_asr.preprocessing.text import normalize_khmer_text, remove_spaces


def sentence_key(text: str) -> str:
    """Identity of a sentence for leakage checks: normalized and space-free."""
    return remove_spaces(normalize_khmer_text(text))


def add_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add `row_in_file`, a unique `utt_id` (`<file>:<row>`), normalized `text` and `sentence_key`."""
    df = df.copy()
    if "row_in_file" not in df:
        # Metadata frames are concatenated whole files in row order.
        df["row_in_file"] = df.groupby("source_file").cumcount()
    df["utt_id"] = df["source_file"].str.removesuffix(".parquet") + ":" + df["row_in_file"].astype(str)
    df["text"] = df["transcript"].fillna("").map(normalize_khmer_text)
    df["sentence_key"] = df["text"].map(remove_spaces)
    return df


def filter_examples(df: pd.DataFrame, min_s: float, max_s: float) -> tuple[pd.DataFrame, dict]:
    """Drop clips with a missing or out-of-range duration, or an empty transcript."""
    reasons = {
        "missing_duration": df["duration"].isna(),
        f"shorter_than_{min_s}s": df["duration"] < min_s,
        f"longer_than_{max_s}s": df["duration"] > max_s,
        "empty_text": df["text"] == "",
    }
    drop = pd.Series(False, index=df.index)
    for mask in reasons.values():
        drop |= mask
    counts = {name: int(mask.sum()) for name, mask in reasons.items()}
    counts["total_dropped"] = int(drop.sum())
    return df[~drop], counts


def _take_hours(df: pd.DataFrame, hours: float, rng: np.random.Generator) -> pd.DataFrame:
    """Random utterances until their total duration reaches `hours`."""
    shuffled = df.iloc[rng.permutation(len(df))]
    within = shuffled["duration"].cumsum() <= hours * 3600
    return shuffled[within]


def split_dataset(
    df: pd.DataFrame,
    test_speakers: list[str],
    test_hours_per_speaker: float,
    validation_hours: float,
    test_seen_speakers_hours: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Return train / validation / test / test_seen_speakers frames (see module docstring)."""
    rng = np.random.default_rng(seed)
    missing = set(test_speakers) - set(df["speaker_id"])
    if missing:
        raise ValueError(f"Test speakers not in data: {sorted(missing)}")

    is_test_speaker = df["speaker_id"].isin(test_speakers)
    touched = set(df.loc[is_test_speaker, "sentence_key"])
    is_touched = df["sentence_key"].isin(touched)

    test = pd.concat(
        _take_hours(df[df["speaker_id"] == s], test_hours_per_speaker, rng) for s in test_speakers
    )
    seen = _take_hours(df[~is_test_speaker & is_touched], test_seen_speakers_hours, rng)

    pool = df[~is_test_speaker & ~is_touched]
    # Validation holds out whole sentences (all their readings) so selection is on unseen text.
    hours_per_sentence = pool.groupby("sentence_key")["duration"].sum()
    order = hours_per_sentence.iloc[rng.permutation(len(hours_per_sentence))]
    val_keys = set(order[order.cumsum() <= validation_hours * 3600].index)
    is_val = pool["sentence_key"].isin(val_keys)

    splits = {
        "train": pool[~is_val],
        "validation": pool[is_val],
        "test": test,
        "test_seen_speakers": seen,
    }
    check_disjoint(splits)
    return {name: frame.sort_values("utt_id") for name, frame in splits.items()}


def check_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if any evaluation split shares utterances or sentences with train, or test shares speakers."""
    train = splits["train"]
    for name, frame in splits.items():
        if name == "train":
            continue
        if set(frame["utt_id"]) & set(train["utt_id"]):
            raise AssertionError(f"{name} shares utterances with train")
        if set(frame["sentence_key"]) & set(train["sentence_key"]):
            raise AssertionError(f"{name} shares sentences with train")
    if set(splits["test"]["speaker_id"]) & set(train["speaker_id"]):
        raise AssertionError("test shares speakers with train")


def split_summary(splits: dict[str, pd.DataFrame]) -> dict:
    return {
        name: {
            "utterances": len(frame),
            "hours": round(float(frame["duration"].sum()) / 3600, 2),
            "speakers": int(frame["speaker_id"].nunique()),
            "sentences": int(frame["sentence_key"].nunique()),
            "hours_by_speaker": {
                k: round(float(v) / 3600, 2)
                for k, v in frame.groupby("speaker_id")["duration"].sum().items()
            },
        }
        for name, frame in splits.items()
    }
