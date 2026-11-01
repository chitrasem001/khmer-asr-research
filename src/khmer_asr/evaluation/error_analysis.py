"""Character-level error analysis on normalized, space-free text (same view as the primary CER).

Every reference character is counted as a hit, a substitution or a deletion; inserted hypothesis
characters are counted separately. Characters are grouped into Khmer script classes so errors can be
compared across consonants, vowel signs, subscripts (coeng), digits and Latin letters.
"""
from collections import Counter
from dataclasses import dataclass, field

import jiwer

from khmer_asr.preprocessing.text import normalize_khmer_text, remove_spaces

COENG = "្"  # Khmer sign that turns the next consonant into a subscript


def char_class(c: str) -> str:
    """Script class of one character, after normalization."""
    cp = ord(c)
    if 0x1780 <= cp <= 0x17A2:
        return "consonant"
    if 0x17A3 <= cp <= 0x17B5:
        return "independent vowel"
    if 0x17B6 <= cp <= 0x17C5:
        return "dependent vowel"
    if c == COENG:
        return "coeng (subscript)"
    if 0x17C6 <= cp <= 0x17DD:
        return "diacritic / sign"
    if 0x17E0 <= cp <= 0x17E9:
        return "digit"
    if "a" <= c <= "z":
        return "latin"
    return "other"


@dataclass
class CharErrors:
    """Counts over a corpus of aligned reference/hypothesis pairs."""

    hits: Counter = field(default_factory=Counter)
    deletions: Counter = field(default_factory=Counter)
    insertions: Counter = field(default_factory=Counter)
    substitutions: Counter = field(default_factory=Counter)  # (ref char, hyp char) -> count

    def reference_count(self, c: str) -> int:
        subs = sum(n for (r, _), n in self.substitutions.items() if r == c)
        return self.hits[c] + subs + self.deletions[c]

    def by_class(self) -> dict[str, dict[str, float]]:
        """Per class: reference characters, substitution/deletion/insertion counts and error rate.

        The rate is (S + D + I) / N for the class, where I counts inserted characters of that class,
        so the classes' errors add up to the corpus CER numerator.
        """
        out: dict[str, dict[str, float]] = {}

        def bucket(c: str) -> dict[str, float]:
            return out.setdefault(char_class(c), {"chars": 0, "sub": 0, "del": 0, "ins": 0})

        for c, n in self.hits.items():
            bucket(c)["chars"] += n
        for (r, _), n in self.substitutions.items():
            bucket(r)["chars"] += n
            bucket(r)["sub"] += n
        for c, n in self.deletions.items():
            bucket(c)["chars"] += n
            bucket(c)["del"] += n
        for c, n in self.insertions.items():
            bucket(c)["ins"] += n
        for v in out.values():
            v["error_rate"] = (v["sub"] + v["del"] + v["ins"]) / v["chars"] if v["chars"] else None
        return out


def char_errors(references: list[str], hypotheses: list[str]) -> CharErrors:
    """Align each pair on normalized, space-free text and count character-level outcomes."""
    errs = CharErrors()
    for ref_raw, hyp_raw in zip(references, hypotheses, strict=True):
        ref = remove_spaces(normalize_khmer_text(ref_raw))
        hyp = remove_spaces(normalize_khmer_text(hyp_raw or ""))
        if not ref:
            continue
        out = jiwer.process_characters([ref], [hyp])
        for chunk in out.alignments[0]:
            r = ref[chunk.ref_start_idx : chunk.ref_end_idx]
            h = hyp[chunk.hyp_start_idx : chunk.hyp_end_idx]
            if chunk.type == "equal":
                errs.hits.update(r)
            elif chunk.type == "substitute":
                errs.substitutions.update(zip(r, h, strict=True))
            elif chunk.type == "delete":
                errs.deletions.update(r)
            elif chunk.type == "insert":
                errs.insertions.update(h)
    return errs


# Spellings the corpus uses interchangeably for the same sound: subscript da/ta and the two forms
# of the independent vowel qoo (as in ឲ្យ / ឱ្យ). No model can tell them apart from audio.
SPELLING_VARIANTS = {COENG + "ដ": COENG + "ត", "ឲ": "ឱ"}


def merge_spelling_variants(text: str) -> str:
    """Normalize, then map each spelling variant to one form (sensitivity analysis only; the
    primary CER keeps protocol v1)."""
    text = normalize_khmer_text(text)
    for old, new in SPELLING_VARIANTS.items():
        text = text.replace(old, new)
    return text


def is_looping(reference: str, hypothesis: str, factor: float = 1.5) -> bool:
    """A hypothesis much longer than its reference: the decoder repeated itself."""
    ref = remove_spaces(normalize_khmer_text(reference))
    hyp = remove_spaces(normalize_khmer_text(hypothesis or ""))
    return len(hyp) > factor * max(len(ref), 1)
