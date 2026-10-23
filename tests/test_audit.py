import pandas as pd

from khmer_asr.data.audit import (
    char_inventory,
    file_batch,
    non_khmer_chars,
    speaker_table,
    summarize,
    text_issues,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transcript": ["សួស្តី", "សួស្តី ", "ពិភពលោក 1", ""],
            "speaker_id": ["f-adt1-0001", "m-adt1-0002", "f-adt1-0001", "m-adt1-0002"],
            "topic": ["a", "a", "b", "b"],
            "subtopic": ["x", "x", "y", "z"],
            "paragraph_id": [1, 1, 2, 3],
            "sentence_id": ["s1", "s2", "s3", "s3"],
            "duration": [3600.0, 0.2, 40.0, 5.0],
        }
    )


def test_summarize_counts_durations_and_duplicates():
    s = summarize(_frame())
    assert s["rows"] == 4
    assert s["speakers"] == 2
    assert s["duration_s"]["below_0.5s"] == 1
    assert s["duration_s"]["above_30.0s"] == 2
    assert s["transcript_chars"]["empty"] == 1
    # "សួស្តី" and "សួស្តី " match after normalization, read by two speakers.
    assert s["duplicates"]["rows_with_duplicated_transcript"] == 2
    assert s["duplicates"]["duplicated_transcripts_read_by_2plus_speakers"] == 1
    assert s["duplicates"]["duplicated_sentence_ids"] == 1
    assert s["duplicates"]["unique_sentences"] == 2
    assert s["duplicates"]["speakers_per_sentence"] == {"1": 1, "2": 1}


def test_summarize_handles_missing_durations():
    df = _frame()
    df.loc[0, "duration"] = float("nan")
    s = summarize(df)
    assert s["duration_s"]["missing"] == 1
    assert s["duration_s"]["median"] == 5.0


def test_non_khmer_chars_ignores_khmer_and_spaces():
    assert non_khmer_chars(pd.Series(["សួស្តី 12", "a"])) == {"1": 1, "2": 1, "a": 1}


def test_speaker_table_parses_gender_code():
    table = speaker_table(_frame()).set_index("speaker_id")
    assert table.loc["f-adt1-0001", "gender_code"] == "f"
    assert table.loc["f-adt1-0001", "utterances"] == 2


def test_char_inventory_counts_occurrences_and_sentences():
    inv = char_inventory(pd.Series(["aab", "a"])).set_index("char")
    assert inv.loc["a", "occurrences"] == 3
    assert inv.loc["a", "sentences"] == 2
    assert inv.loc["a", "codepoint"] == "U+0061"


def test_text_issues_flags_legacy_order_and_zero_width():
    issues = text_issues(pd.Series(["កា្លយ", "ក\u200bខ", "abc 12"]))
    assert issues["vowel_before_subscript"] == 1
    assert issues["zero_width_chars"] == 1
    assert issues["latin_letters"] == 1
    assert issues["arabic_digits"] == 1


def test_file_batch_names():
    assert file_batch("data/train-v313-00459.parquet") == "v313"
    assert file_batch("train-00653-of-00724.parquet") == "of-00724"
    assert file_batch("train-00000-of-00653.parquet") == "of-00653"
