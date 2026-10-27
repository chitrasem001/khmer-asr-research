"""Character alignment between a reference and a hypothesis, for error views.

Uses the same normalization and space-free comparison as the primary CER, so the counts shown
in the demo match the numbers reported in the thesis.
"""
import jiwer

from khmer_asr.preprocessing.text import normalize_khmer_text, remove_spaces


def align(reference: str, hypothesis: str) -> dict:
    """Return CER, S/I/D counts and contiguous segments of equal/substitute/insert/delete."""
    ref = remove_spaces(normalize_khmer_text(reference))
    hyp = remove_spaces(normalize_khmer_text(hypothesis))
    if not ref:
        raise ValueError("Reference is empty after normalization")
    out = jiwer.process_characters([ref], [hyp])
    segments = [
        {
            "type": chunk.type,
            "ref": ref[chunk.ref_start_idx : chunk.ref_end_idx],
            "hyp": hyp[chunk.hyp_start_idx : chunk.hyp_end_idx],
        }
        for chunk in out.alignments[0]
    ]
    n = out.hits + out.substitutions + out.deletions
    return {
        "cer": (out.substitutions + out.deletions + out.insertions) / n,
        "substitutions": out.substitutions,
        "insertions": out.insertions,
        "deletions": out.deletions,
        "reference_chars": n,
        "segments": segments,
    }
