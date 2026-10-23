"""Nested training subsets for data-scaling experiments.

Subsets are drawn by whole parquet file, not by clip: audio is stored per file, so a random clip
sample would download nearly the whole corpus. Files are shuffled once with a fixed seed and taken
in order until the target hours are reached, so a 10 h subset is contained in the 50 h subset.
"""
import numpy as np
import pandas as pd


def select_train_hours(train: pd.DataFrame, hours: float | None, seed: int) -> pd.DataFrame:
    """Return all of `train` (hours=None) or the first files of a seeded file order up to `hours`."""
    if hours is None:
        return train
    files = np.sort(train["source_file"].unique())
    order = files[np.random.default_rng(seed).permutation(len(files))]
    file_hours = train.groupby("source_file")["duration"].sum().reindex(order) / 3600
    keep = file_hours.index[(file_hours.cumsum() - file_hours) < hours]
    return train[train["source_file"].isin(set(keep))]
