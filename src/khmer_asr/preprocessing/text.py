"""Khmer transcript normalization.

The rules here define the evaluation protocol; record any change in docs/decisions.md.
"""
import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
# Zero-width space / non-joiner / joiner / BOM, commonly found in Khmer text.
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")


def normalize_khmer_text(
    text: str,
    unicode_form: str = "NFC",
    remove_zero_width: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Normalize a Khmer transcript for training and evaluation."""
    text = unicodedata.normalize(unicode_form, text)
    if remove_zero_width:
        text = _ZERO_WIDTH.sub("", text)
    if collapse_whitespace:
        text = _WHITESPACE.sub(" ", text)
    return text.strip()
