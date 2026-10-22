import pandas as pd
import pytest

from khmer_asr.data.split import add_ids, check_disjoint, filter_examples, split_dataset


def _corpus() -> pd.DataFrame:
    """Six sentences, each read by four of five speakers; durations of 1 h for easy arithmetic."""
    readers = {
        "s1": ["a", "b", "c", "d"],
        "s2": ["a", "b", "c", "d"],
        "s3": ["a", "b", "c", "e"],
        "s4": ["a", "b", "c", "e"],
        "s5": ["a", "b", "d", "e"],
        "s6": ["a", "b", "c", "d"],
    }
    texts = {"s1": "ក", "s2": "ខ", "s3": "គ", "s4": "ឃ", "s5": "ង", "s6": "ច"}
    rows = [
        {"transcript": texts[s], "speaker_id": spk, "duration": 3600.0, "source_file": "f.parquet"}
        for s, spks in readers.items()
        for spk in spks
    ]
    return add_ids(pd.DataFrame(rows))


def test_add_ids_makes_unique_utterance_ids():
    df = _corpus()
    assert df["utt_id"].is_unique
    assert df["utt_id"].iloc[0] == "f:0"


def test_filter_examples_counts_reasons():
    df = _corpus()
    df.loc[0, "duration"] = None
    df.loc[1, "duration"] = 0.1
    kept, dropped = filter_examples(df, 0.5, 30.0 * 3600)
    assert dropped["missing_duration"] == 1
    assert dropped["shorter_than_0.5s"] == 1
    assert len(kept) == len(df) - 2


def test_split_holds_out_speakers_and_their_sentences():
    splits = split_dataset(
        _corpus(),
        test_speakers=["e"],
        test_hours_per_speaker=10,
        validation_hours=4,
        test_seen_speakers_hours=100,
        seed=0,
    )
    train = splits["train"]
    assert "e" not in set(train["speaker_id"])
    # Speaker e read s3, s4, s5, so none of those sentences may be in train.
    assert not {"គ", "ឃ", "ង"} & set(train["text"])
    assert set(splits["test"]["speaker_id"]) == {"e"}
    assert set(splits["test_seen_speakers"]["text"]) == {"គ", "ឃ", "ង"}
    # Validation takes whole sentences: one sentence = 4 readings = 4 h.
    assert len(splits["validation"]) == 4
    assert splits["validation"]["text"].nunique() == 1


def test_split_is_reproducible_with_seed():
    kwargs = {
        "test_speakers": ["e"],
        "test_hours_per_speaker": 2,
        "validation_hours": 4,
        "test_seen_speakers_hours": 3,
        "seed": 7,
    }
    a = split_dataset(_corpus(), **kwargs)
    b = split_dataset(_corpus(), **kwargs)
    for name in a:
        assert a[name]["utt_id"].tolist() == b[name]["utt_id"].tolist()


def test_unknown_test_speaker_raises():
    with pytest.raises(ValueError):
        split_dataset(_corpus(), ["zzz"], 1, 1, 1, seed=0)


def test_check_disjoint_detects_sentence_leak():
    df = _corpus()
    with pytest.raises(AssertionError):
        # Rows 0 and 1 are different speakers reading the same sentence.
        check_disjoint({"train": df.iloc[:1], "test": df.iloc[1:2], "validation": df.iloc[:0]})
