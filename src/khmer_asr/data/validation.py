"""Sanity checks for dataset examples and manifests."""
from collections.abc import Iterable


def is_valid_example(
    transcript: str | None,
    duration_s: float | None,
    min_duration_s: float = 0.5,
    max_duration_s: float = 30.0,
) -> bool:
    """Return True if an example has a non-empty transcript and an acceptable duration."""
    if not transcript or not transcript.strip():
        return False
    if duration_s is None:
        return False
    return min_duration_s <= duration_s <= max_duration_s


def find_split_overlap(*splits: Iterable[str]) -> set[str]:
    """Return IDs that appear in more than one split (data leakage check)."""
    seen: set[str] = set()
    overlap: set[str] = set()
    for split in splits:
        ids = set(split)
        overlap |= seen & ids
        seen |= ids
    return overlap
