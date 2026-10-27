import pytest

from khmer_asr.evaluation.alignment import align
from khmer_asr.evaluation.metrics import cer


def test_identical_text_is_one_equal_segment():
    out = align("abc", "abc")
    assert out["cer"] == 0.0
    assert out["segments"] == [{"type": "equal", "ref": "abc", "hyp": "abc"}]


def test_counts_each_error_type():
    out = align("abcd", "axcde")
    assert (out["substitutions"], out["insertions"], out["deletions"]) == (1, 1, 0)
    types = [s["type"] for s in out["segments"]]
    assert "substitute" in types and "insert" in types


def test_segments_rebuild_both_strings():
    out = align("សួស្តី ពិភពលោក", "សួស្ដីពិភព")
    assert "".join(s["ref"] for s in out["segments"]) == "សួស្តីពិភពលោក"
    assert "".join(s["hyp"] for s in out["segments"]) == "សួស្ដីពិភព"


def test_matches_primary_cer():
    ref, hyp = "ក ខ គ ឃ", "ក គ ង ឃ ច"
    assert align(ref, hyp)["cer"] == pytest.approx(cer([ref], [hyp]))


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        align("។", "ក")
