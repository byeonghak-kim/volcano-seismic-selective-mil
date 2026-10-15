"""Leakage-controlled temporal and group split construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def build_temporal_group_split(
    events: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    validation_folds: int = 5,
    validation_fold: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign overlap components touching the cutoff to test as whole groups."""
    required = {"event_id", "event_time", "label", "group_id"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events is missing required columns: {missing}")
    frame = events.copy().reset_index(drop=True)
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="raise")
    boundary = pd.Timestamp(cutoff)
    summary = frame.groupby("group_id").agg(
        maximum_time=("event_time", "max"), label_count=("label", "nunique")
    )
    if int(summary["label_count"].max()) != 1:
        raise ValueError("Mixed-label overlap groups must be excluded before splitting")
    test_groups = set(summary.index[summary["maximum_time"] >= boundary])
    test_mask = frame["group_id"].isin(test_groups).to_numpy()
    pretest_index = np.flatnonzero(~test_mask)
    if len(pretest_index) == 0 or np.all(test_mask):
        raise ValueError("Both pre-cutoff and test groups are required")
    pretest = frame.iloc[pretest_index]
    splitter = StratifiedGroupKFold(
        n_splits=validation_folds, shuffle=True, random_state=seed
    )
    folds = list(
        splitter.split(
            np.zeros(len(pretest), dtype=np.int8),
            pretest["label"].to_numpy(),
            pretest["group_id"].to_numpy(),
        )
    )
    if not 0 <= validation_fold < len(folds):
        raise ValueError("validation_fold is outside the generated folds")
    train_local, validation_local = folds[validation_fold]
    frame["split"] = ""
    frame.loc[pretest_index[train_local], "split"] = "train"
    frame.loc[pretest_index[validation_local], "split"] = "validation"
    frame.loc[test_mask, "split"] = "test"
    if frame.groupby("group_id")["split"].nunique().max() != 1:
        raise RuntimeError("Overlap-group leakage detected")
    return frame


def split_indices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return integer indices for train, validation, and test rows."""
    return {
        split: np.flatnonzero(frame["split"].eq(split).to_numpy())
        for split in ("train", "validation", "test")
    }
