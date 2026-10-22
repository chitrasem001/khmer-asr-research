"""Khmer transcript normalization.

The rules here define the evaluation protocol; record any change in docs/decisions.md.
Rules (in order):
 1. Unicode NFC.
 2. Deprecated Khmer characters: U+17A3 -> U+17A2, U+17A4 -> U+17A2 U+17B6, U+17B4/U+17B5 removed.
 3. Zero-width characters removed (U+200B/C/D, U+FEFF).
 4. Legacy character order fixed so identical-looking text compares equal:
    coeng-ro before another subscript, vowel typed before a subscript, nikahit before u.
 5. Punctuation (Unicode category P*, incl. Khmer ។ ៕ ៖) replaced by a space, except % & # @;
    symbols such as $ + and the repetition sign ៗ are kept. All of these are read aloud.
 6. Arabic digits mapped to Khmer digits; Latin letters lowercased.
 7. All whitespace (newlines, no-break spaces) collapsed to one space and stripped.
"""
import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
# Zero-width space / non-joiner / joiner / BOM, commonly found in Khmer text.
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")
_DEPRECATED = str.maketrans({"\u17a3": "\u17a2", "\u17a4": "\u17a2\u17b6", "\u17b4": "", "\u17b5": ""})
_DIGITS = str.maketrans("0123456789", "\u17e0\u17e1\u17e2\u17e3\u17e4\u17e5\u17e6\u17e7\u17e8\u17e9")

_CONSONANT = "[\u1780-\u17a2]"
_SUBSCRIPT = f"\u17d2{_CONSONANT}"
_VOWEL = "[\u17b6-\u17c5]"
# Coeng-ro typed before another subscript (legacy order) is moved last.
_COENG_RO_FIRST = re.compile(f"(\u17d2\u179a)({_SUBSCRIPT})")
# A dependent vowel typed before a subscript is moved after it.
_VOWEL_BEFORE_SUBSCRIPT = re.compile(f"({_VOWEL})({_SUBSCRIPT})")
# Nikahit typed before vowel u is swapped (u comes first).
_NIKAHIT_BEFORE_U = re.compile("\u17c6\u17bb")
# Unicode classes these as punctuation, but they are read aloud.
_SPOKEN_PUNCTUATION = frozenset("%&#@")


def _fix_order(text: str) -> str:
    # Vowel first: moving a vowel can expose a legacy coeng-ro order.
    text = _VOWEL_BEFORE_SUBSCRIPT.sub(r"\2\1", text)
    text = _COENG_RO_FIRST.sub(r"\2\1", text)
    return _NIKAHIT_BEFORE_U.sub("\u17bb\u17c6", text)


def _strip_punctuation(text: str) -> str:
    return "".join(
        " " if unicodedata.category(ch).startswith("P") and ch not in _SPOKEN_PUNCTUATION else ch
        for ch in text
    )


def normalize_khmer_text(
    text: str,
    unicode_form: str = "NFC",
    remove_zero_width: bool = True,
    collapse_whitespace: bool = True,
    fix_order: bool = True,
    remove_punctuation: bool = True,
    map_digits: bool = True,
    lowercase_latin: bool = True,
) -> str:
    """Normalize a Khmer transcript for training and evaluation."""
    text = unicodedata.normalize(unicode_form, text).translate(_DEPRECATED)
    if remove_zero_width:
        text = _ZERO_WIDTH.sub("", text)
    if fix_order:
        text = _fix_order(text)
    if remove_punctuation:
        text = _strip_punctuation(text)
    if map_digits:
        text = text.translate(_DIGITS)
    if lowercase_latin:
        text = text.lower()
    if collapse_whitespace:
        text = _WHITESPACE.sub(" ", text)
    return text.strip()


def remove_spaces(text: str) -> str:
    """Drop all whitespace; Khmer spacing is inconsistent, so primary CER ignores it."""
    return _WHITESPACE.sub("", text)
