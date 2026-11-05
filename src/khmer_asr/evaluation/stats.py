"""Uncertainty for corpus CER: bootstrap intervals over clips and paired comparisons.

Clips are resampled with replacement (1,000 resamples, fixed seed). Errors of one speaker are
correlated, so intervals over clips may be optimistic (Bisani & Ney, ICASSP 2004).
"""
import jiwer
import numpy as np
import pandas as pd

from khmer_asr.preprocessing.text import normalize_khmer_text, remove_spaces


def clip_errors(pred: pd.DataFrame, protocol: str | None = None) -> pd.DataFrame:
    """Per-clip character edit counts on normalized, space-free text (columns errors, ref_chars)."""
    rows = []
    for ref, hyp in zip(pred["reference"], pred["hypothesis"].fillna(""), strict=True):
        r = remove_spaces(normalize_khmer_text(ref, protocol=protocol))
        h = remove_spaces(normalize_khmer_text(hyp, protocol=protocol))
        o = jiwer.process_characters([r], [h])
        rows.append((o.substitutions + o.deletions + o.insertions, o.hits + o.substitutions + o.deletions))
    return pd.DataFrame(rows, columns=["errors", "ref_chars"])


def bootstrap_ci(errs: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% bootstrap interval for corpus CER, resampling clips."""
    rng = np.random.default_rng(seed)
    e, n = errs["errors"].to_numpy(), errs["ref_chars"].to_numpy()
    idx = rng.integers(0, len(e), size=(n_boot, len(e)))
    cers = e[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.percentile(cers, 2.5)), float(np.percentile(cers, 97.5))


def paired_diff_ci(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000,
                   seed: int = 42) -> tuple[float, float, float]:
    """CER(a) - CER(b) on the same clips, with a 95% paired bootstrap interval."""
    if len(a) != len(b):
        raise ValueError("Paired comparison needs the same clips")
    rng = np.random.default_rng(seed)
    ea, eb, n = a["errors"].to_numpy(), b["errors"].to_numpy(), a["ref_chars"].to_numpy()
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    diffs = (ea[idx].sum(axis=1) - eb[idx].sum(axis=1)) / n[idx].sum(axis=1)
    return (float((ea.sum() - eb.sum()) / n.sum()),
            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
