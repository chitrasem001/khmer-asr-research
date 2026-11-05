import pandas as pd
import pytest

from khmer_asr.evaluation.stats import bootstrap_ci, clip_errors, paired_diff_ci


def test_clip_errors_counts_edits_without_spaces():
    pred = pd.DataFrame({"reference": ["ក ខ", "គឃ"], "hypothesis": ["កខ", "គ"]})
    errs = clip_errors(pred)
    assert errs["errors"].tolist() == [0, 1]
    assert errs["ref_chars"].tolist() == [2, 2]


def test_bootstrap_interval_contains_corpus_cer():
    errs = pd.DataFrame({"errors": [0, 1, 2, 0, 1] * 20, "ref_chars": [10] * 100})
    lo, hi = bootstrap_ci(errs)
    assert lo <= 0.08 <= hi


def test_paired_diff_of_identical_systems_is_zero():
    errs = pd.DataFrame({"errors": [0, 1, 2], "ref_chars": [5, 5, 5]})
    assert paired_diff_ci(errs, errs) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        paired_diff_ci(errs, errs.iloc[:2])
