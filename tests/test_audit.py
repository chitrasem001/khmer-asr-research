import pandas as pd

from khmer_asr.data.audit import non_khmer_chars, speaker_table, summarize


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
