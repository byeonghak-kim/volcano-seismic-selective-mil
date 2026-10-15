"""Executable raw-data-free fixture for archive, training, policy, and bootstrap."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from ve_selective.pipeline import run_research_pipeline
from ve_selective.preprocessing import build_keyed_archive, numpy_record_loader
from ve_selective.splitting import build_temporal_group_split


def create_fixture(output: Path, seed: int = 20260814) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    records_root = output / "records"
    records_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    time = np.arange(6000, dtype=np.float64) / 100.0
    event_rows = []
    record_rows = []
    frequencies = {"A": 8.0, "B": 4.0, "T": 2.0}
    event_id = 0
    for class_index, label in enumerate(("A", "B", "T")):
        for group_index in range(20):
            record_keys = []
            for record_index in range(2):
                key = f"synthetic_{label}_{group_index:02d}_{record_index}"
                record_keys.append(key)
                phase = np.arange(3)[:, None] * np.pi / 5.0
                if label == "T":
                    envelope = np.ones_like(time)
                else:
                    center = 15.0 + 12.0 * record_index
                    envelope = np.exp(-0.5 * ((time - center) / 1.5) ** 2)
                signal = 180.0 * envelope[None, :] * np.sin(
                    2.0 * np.pi * frequencies[label] * time[None, :] + phase
                )
                signal += rng.normal(0.0, 8.0, size=signal.shape)
                relative = Path(f"{key}.npy")
                np.save(records_root / relative, signal.astype(np.float32), allow_pickle=False)
                record_rows.append({"record_key": key, "array_path": relative.as_posix(), "sampling_rate_hz": 100.0})
            if group_index < 15:
                event_time = pd.Timestamp("2023-01-01") + pd.Timedelta(days=class_index * 80 + group_index)
            else:
                event_time = pd.Timestamp("2023-10-01") + pd.Timedelta(days=class_index * 10 + group_index - 15)
            event_rows.append({"event_id": event_id, "label": label, "event_time": event_time.isoformat(), "first_record_key": record_keys[0], "second_record_key": record_keys[1]})
            event_id += 1
    events = pd.DataFrame(event_rows)
    records = pd.DataFrame(record_rows)
    events.to_csv(output / "events_manifest.csv", index=False)
    records.to_csv(output / "records_manifest.csv", index=False)
    return events, records, records_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("demo_output/research_pipeline"))
    parser.add_argument("--config", type=Path, default=Path("configs/research_pipeline.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events, records, records_root = create_fixture(args.output)
    bundle = build_keyed_archive(events, records, numpy_record_loader(records_root))
    split = build_temporal_group_split(bundle.events, "2023-10-01", seed=42)
    split.to_csv(args.output / "split_manifest.csv", index=False)
    bundle.save(args.output / "archive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["training"]["seeds"] = [42]
    config["training"]["batch_size"] = 16
    config["policy"].update({
        "class_A_precision_constraint": 0.0,
        "target_coverage": 0.50,
        "coverage_tolerance": 0.40,
        "minimum_accepted_accuracy": 0.0,
        "minimum_accepted_macro_f1": 0.0,
        "coverage_targets": [0.50],
        "rate_grid": [0.25, 0.50, 0.75, 1.00],
    })
    config["resampling"]["repetitions"] = 20
    config["resampling"]["matched_count_temporal_test"] = 8
    summary = run_research_pipeline(
        bundle.waveforms,
        bundle.labels,
        split,
        args.output / "results",
        config,
        quick=True,
    )
    print(json.dumps({"status": summary["status"], "split_counts": summary["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
