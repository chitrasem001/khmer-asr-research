"""Recognition metrics. Inputs are normalized before scoring."""
import jiwer

from khmer_asr.preprocessing.text import normalize_khmer_text


def cer(references: list[str], hypotheses: list[str], normalize: bool = True) -> float:
    """Corpus-level Character Error Rate."""
    if normalize:
        references = [normalize_khmer_text(r) for r in references]
        hypotheses = [normalize_khmer_text(h) for h in hypotheses]
    return jiwer.cer(references, hypotheses)
