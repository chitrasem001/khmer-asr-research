from khmer_asr.evaluation.metrics import cer
from khmer_asr.preprocessing.text import normalize_khmer_text


def test_collapses_whitespace():
    assert normalize_khmer_text("  សួស្តី   ពិភពលោក \n") == "សួស្តី ពិភពលោក"


def test_removes_zero_width_space():
    assert normalize_khmer_text("សួស្តី\u200bពិភពលោក") == "សួស្តីពិភពលោក"


def test_fixes_legacy_coeng_ro_order():
    # ចង្រ្កាន typed with coeng-ro first -> canonical ចង្ក្រាន
    assert normalize_khmer_text("ចង្រ្កាន") == "ចង្ក្រាន"


def test_fixes_vowel_typed_before_subscript():
    assert normalize_khmer_text("កា្លយ") == "ក្លាយ"


def test_fixes_vowel_and_coeng_ro_together():
    # ស្រី្ត (vowel before ta subscript, ro first) -> ស្ត្រី
    assert normalize_khmer_text("ស្រី្ត") == "ស្ត្រី"


def test_fixes_nikahit_before_u():
    assert normalize_khmer_text("ដំុ") == "ដុំ"


def test_replaces_deprecated_characters():
    assert normalize_khmer_text("ឣូរ") == "អូរ"


def test_removes_punctuation_but_keeps_spoken_symbols():
    assert normalize_khmer_text("«សួស្តី»! ១០% ល្អៗ។") == "សួស្តី ១០% ល្អៗ"


def test_maps_digits_and_lowercases_latin():
    assert normalize_khmer_text("វីតាមីន B12") == "វីតាមីន b១២"


def test_collapses_newlines_and_no_break_spaces():
    assert normalize_khmer_text("ក\n   ខ") == "ក ខ"


def test_cer_ignores_spaces_by_default():
    assert cer(["សួស្តី ពិភពលោក"], ["សួស្តីពិភពលោក"]) == 0.0
    assert cer(["សួស្តី ពិភពលោក"], ["សួស្តីពិភពលោក"], ignore_spaces=False) > 0.0


def test_cer_identical_is_zero():
    assert cer(["សួស្តី"], ["សួស្តី"]) == 0.0


def test_cer_ignores_normalization_differences():
    assert cer(["សួស្តី  ពិភពលោក"], ["សួស្តី ពិភពលោក"]) == 0.0
