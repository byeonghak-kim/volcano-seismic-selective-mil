"""Leakage-control utilities for bags sharing physical minute records."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd


def overlap_group_ids(
    first_record: Sequence[Hashable],
    second_record: Sequence[Hashable],
) -> np.ndarray:
    """Connect bags that share either physical record identifier."""
    if len(first_record) != len(second_record):
        raise ValueError("Record identifier arrays must have equal length")
    n_events = len(first_record)
    parent = list(range(n_events))
    size = [1] * n_events

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    owner: dict[Hashable, int] = {}
    for event_index, records in enumerate(zip(first_record, second_record)):
        for record in records:
            if record in owner:
                union(event_index, owner[record])
            else:
                owner[record] = event_index
    roots = [find(index) for index in range(n_events)]
    root_to_group = {root: group for group, root in enumerate(sorted(set(roots)))}
    return np.asarray([root_to_group[root] for root in roots], dtype=np.int64)


def exclude_mixed_label_groups(
    frame: pd.DataFrame,
    group_column: str = "group_id",
    label_column: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return retained and excluded rows after mixed-label group removal."""
    label_count = frame.groupby(group_column)[label_column].nunique()
    mixed = set(label_count[label_count > 1].index)
    excluded = frame.loc[frame[group_column].isin(mixed)].copy()
    retained = frame.loc[~frame[group_column].isin(mixed)].copy()
    return retained, excluded

