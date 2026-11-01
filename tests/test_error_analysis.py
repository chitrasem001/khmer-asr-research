from khmer_asr.evaluation.error_analysis import char_class, char_errors, is_looping


def test_char_classes():
    assert char_class("ក") == "consonant"
    assert char_class("ា") == "dependent vowel"
    assert char_class("្") == "coeng (subscript)"
    assert char_class("៤") == "digit"
    assert char_class("a") == "latin"
    assert char_class("%") == "other"


def test_char_errors_counts_each_outcome():
    # ref ក ខ គ ; hyp ក ង គ ច -> 2 hits, ខ->ង substituted, ច inserted
    errs = char_errors(["កខគ"], ["កងគច"])
    assert errs.hits == {"ក": 1, "គ": 1}
    assert errs.substitutions == {("ខ", "ង"): 1}
    assert errs.insertions == {"ច": 1}
    assert errs.reference_count("ខ") == 1
    cons = errs.by_class()["consonant"]
    assert cons["chars"] == 3 and cons["sub"] == 1 and cons["ins"] == 1
    assert abs(cons["error_rate"] - 2 / 3) < 1e-9


def test_char_errors_ignores_spaces_and_counts_deletions():
    errs = char_errors(["ក ខ"], ["ក"])
    assert errs.deletions == {"ខ": 1}


def test_is_looping():
    assert is_looping("កខ", "កខកខកខ")
    assert not is_looping("កខគ", "កខ")


def test_merge_spelling_variants_maps_subscript_da_and_qoo():
    from khmer_asr.evaluation.error_analysis import merge_spelling_variants

    assert merge_spelling_variants("ស្ដាប់") == merge_spelling_variants("ស្តាប់")
    assert merge_spelling_variants("ឲ្យ") == merge_spelling_variants("ឱ្យ")
    assert merge_spelling_variants("ដើម") == "ដើម"  # a full (non-subscript) da is not changed
