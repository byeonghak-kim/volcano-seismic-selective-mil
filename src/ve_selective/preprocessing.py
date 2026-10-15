"""Keyed archive construction and the waveform preprocessing used by the study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.signal import iirfilter, resample, sosfilt, zpk2sos

from .grouping import exclude_mixed_label_groups, overlap_group_ids


EVENT_COLUMNS = (
    "event_id",
    "label",
    "event_time",
    "first_record_key",
    "second_record_key",
)
RECORD_COLUMNS = ("record_key", "array_path", "sampling_rate_hz")
CATALOGUE_LABEL_MAP = {
    "A": "A",
    "B": "B",
    "BH": "B",
    "BL": "B",
    "BP": "B",
    "BS": "B",
    "BT": "B",
    "T": "T",
    "T(c)": "T",
    "T(e)": "T",
    "T(s)": "T",
    "TK": "T",
    "TP": "T",
}


@dataclass(frozen=True)
class PreprocessingConfig:
    target_sampling_rate_hz: float = 100.0
    low_cut_hz: float = 0.5
    high_cut_hz: float = 30.0
    butterworth_order: int = 4
    samples_per_record: int = 6000


@dataclass
class ArchiveBundle:
    waveforms: np.ndarray
    labels: np.ndarray
    events: pd.DataFrame
    excluded_mixed_label_events: pd.DataFrame

    def save(self, output: Path) -> None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "waveforms.npy", self.waveforms, allow_pickle=False)
        np.save(output / "labels.npy", self.labels, allow_pickle=False)
        self.events.to_csv(output / "events.csv", index=False)
        self.excluded_mixed_label_events.to_csv(
            output / "excluded_mixed_label_events.csv", index=False
        )


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def collapse_catalogue_label(label: str) -> str:
    """Collapse the upstream catalogue subcodes to the study's A/B/T targets."""
    value = str(label).strip()
    if value not in CATALOGUE_LABEL_MAP:
        raise ValueError(f"Unknown catalogue label: {value!r}")
    return CATALOGUE_LABEL_MAP[value]


def preprocess_component(
    samples: np.ndarray,
    sampling_rate_hz: float,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> np.ndarray:
    """Demean, forward/reverse SOS filter, resample, and crop/right-pad one channel.

    Values are SAC samples in stored/native units. Instrument-response correction
    and cross-sensor unit conversion are intentionally outside this recipe.
    """
    x = np.asarray(samples)
    if x.ndim != 1 or x.size == 0 or np.iscomplexobj(x):
        raise ValueError("samples must be a non-empty real one-dimensional array")
    if not np.issubdtype(x.dtype, np.number) or not np.all(np.isfinite(x)):
        raise ValueError("samples must contain finite numeric values")
    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 2.0 * config.high_cut_hz:
        raise ValueError("sampling_rate_hz must be finite and exceed 60 Hz")
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    nyquist = 0.5 * fs
    z, p, k = iirfilter(
        config.butterworth_order,
        [config.low_cut_hz / nyquist, config.high_cut_hz / nyquist],
        btype="band",
        ftype="butter",
        output="zpk",
    )
    sos = zpk2sos(z, p, k)
    x = sosfilt(sos, x)
    x = sosfilt(sos, x[::-1])[::-1]
    if not np.isclose(fs, config.target_sampling_rate_hz):
        target_length = int(round(len(x) * config.target_sampling_rate_hz / fs))
        x = resample(x, target_length)
    x = np.asarray(x, dtype=np.float32)
    length = config.samples_per_record
    if len(x) >= length:
        return x[:length]
    return np.pad(x, (0, length - len(x))).astype(np.float32, copy=False)


def preprocess_record(
    components: np.ndarray,
    sampling_rate_hz: float,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> np.ndarray:
    """Preprocess one E/N/U-or-Z record represented as `(3, samples)`."""
    record = np.asarray(components)
    if record.ndim != 2 or record.shape[0] != 3:
        raise ValueError(f"Expected component order E,N,U/Z with shape (3,L), got {record.shape}")
    return np.stack(
        [preprocess_component(record[index], sampling_rate_hz, config) for index in range(3)]
    ).astype(np.float32)


def numpy_record_loader(records_root: Path) -> Callable[[pd.Series], np.ndarray]:
    """Create a safe loader for relative `.npy` paths listed in records.csv."""
    root = Path(records_root).resolve()

    def load(row: pd.Series) -> np.ndarray:
        relative = Path(str(row["array_path"]))
        if relative.is_absolute():
            raise ValueError("array_path values must be relative to records_root")
        path = (root / relative).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"array_path escapes records_root: {relative}")
        if path.suffix.lower() != ".npy":
            raise ValueError("The sanitized interface accepts only numeric .npy records")
        return np.load(path, allow_pickle=False)

    return load


def build_keyed_archive(
    event_manifest: pd.DataFrame,
    record_manifest: pd.DataFrame,
    record_loader: Callable[[pd.Series], np.ndarray],
    config: PreprocessingConfig = PreprocessingConfig(),
    classes: tuple[str, ...] = ("A", "B", "T"),
) -> ArchiveBundle:
    """Build W2 tensors by joining event rows to physical station-minute keys."""
    events = event_manifest.copy().reset_index(drop=True)
    records = record_manifest.copy().reset_index(drop=True)
    _require_columns(events, EVENT_COLUMNS, "event manifest")
    _require_columns(records, RECORD_COLUMNS, "record manifest")
    if events["event_id"].duplicated().any():
        raise ValueError("event_id must be unique")
    if records["record_key"].duplicated().any():
        raise ValueError("record_key must be unique")
    events["label"] = events["label"].map(collapse_catalogue_label)
    unknown = sorted(set(events["label"]) - set(classes))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    events["event_time"] = pd.to_datetime(events["event_time"], errors="raise")
    events["group_id"] = overlap_group_ids(
        events["first_record_key"], events["second_record_key"]
    )
    retained, excluded = exclude_mixed_label_groups(events)
    retained = retained.reset_index(drop=True)
    excluded = excluded.reset_index(drop=True)
    record_rows = records.set_index("record_key", drop=False)
    required_keys = set(retained["first_record_key"]) | set(retained["second_record_key"])
    missing_keys = sorted(required_keys - set(record_rows.index))
    if missing_keys:
        raise ValueError(f"Record keys missing from record manifest: {missing_keys[:5]}")
    cache: dict[str, np.ndarray] = {}

    def prepared(key: str) -> np.ndarray:
        if key not in cache:
            row = record_rows.loc[key]
            cache[key] = preprocess_record(
                record_loader(row), float(row["sampling_rate_hz"]), config
            )
        return cache[key]

    waveforms = np.stack(
        [
            np.stack([prepared(row.first_record_key), prepared(row.second_record_key)])
            for row in retained.itertuples(index=False)
        ]
    ).astype(np.float32)
    label_map = {label: index for index, label in enumerate(classes)}
    labels = retained["label"].map(label_map).to_numpy(np.int64)
    return ArchiveBundle(waveforms, labels, retained, excluded)
