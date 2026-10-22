"""Recognition metrics. Inputs are normalized before scoring."""
import jiwer

from khmer_asr.preprocessing.text import normalize_khmer_text, remove_spaces


def cer(
    references: list[str],
    hypotheses: list[str],
    normalize: bool = True,
    ignore_spaces: bool = True,
) -> float:
    """Corpus-level Character Error Rate.

    The primary metric ignores spaces (Khmer word spacing is optional and inconsistent);
    pass ignore_spaces=False for the secondary, space-sensitive CER.
    """
    if normalize:
        references = [normalize_khmer_text(r) for r in references]
        hypotheses = [normalize_khmer_text(h) for h in hypotheses]
    if ignore_spaces:
        references = [remove_spaces(r) for r in references]
        hypotheses = [remove_spaces(h) for h in hypotheses]
    return jiwer.cer(references, hypotheses)
