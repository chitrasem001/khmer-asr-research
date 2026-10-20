from khmer_asr.evaluation.metrics import cer
from khmer_asr.preprocessing.text import normalize_khmer_text


def test_collapses_whitespace():
    assert normalize_khmer_text("  សួស្តី   ពិភពលោក \n") == "សួស្តី ពិភពលោក"


def test_removes_zero_width_space():
    assert normalize_khmer_text("សួស្តី\u200bពិភពលោក") == "សួស្តីពិភពលោក"


def test_cer_identical_is_zero():
    assert cer(["សួស្តី"], ["សួស្តី"]) == 0.0


def test_cer_ignores_normalization_differences():
    assert cer(["សួស្តី  ពិភពលោក"], ["សួស្តី ពិភពលោក"]) == 0.0
