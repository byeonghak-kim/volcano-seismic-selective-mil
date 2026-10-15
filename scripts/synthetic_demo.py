"""Run model-and-policy inference on deterministic preprocessed-input tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ve_selective.inference import SelectiveEnsemble


def synthetic_events(n_events: int = 6) -> np.ndarray:
    rng = np.random.default_rng(20260807)
    time = np.arange(6000, dtype=np.float32) / 100.0
    events = rng.normal(0.0, 12.0, size=(n_events, 2, 3, 6000)).astype(np.float32)
    frequencies = (2.0, 4.5, 8.0)
    for event_index in range(n_events):
        active_window = event_index % 2
        center = 8.0 + 8.0 * (event_index % 5)
        envelope = np.exp(-0.5 * ((time - center) / 1.5) ** 2)
        for component in range(3):
            signal = 350.0 * envelope * np.sin(
                2.0 * np.pi * frequencies[event_index % 3] * time
                + component * np.pi / 5.0
            )
            events[event_index, active_window, component] += signal.astype(np.float32)
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("demo_output/predictions.csv"))
    args = parser.parse_args()
    release_root = Path(__file__).resolve().parents[1]
    system = SelectiveEnsemble.from_release(release_root, device=args.device)
    result = system.predict(
        synthetic_events(),
        sampling_rate_hz=100.0,
        input_domain="preprocessed_waveform_tensor",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
