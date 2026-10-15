"""Verify frozen inference against an externally supplied private reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ve_selective.inference import SelectiveEnsemble


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray, rows_per_chunk: int = 32) -> str:
    """Hash C-order values without embedding a private array in the report."""
    values = np.asarray(array)
    digest = hashlib.sha256()
    for start in range(0, len(values), rows_per_chunk):
        chunk = np.ascontiguousarray(values[start : start + rows_per_chunk])
        digest.update(chunk.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the released frozen system with private final A/B/T "
            "probabilities without publishing event-level data."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Private preprocessed .npy tensor shaped (N, 2, 3, 6000)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Private CSV containing final post-adjustment p_A, p_B, and p_T columns",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probability-tolerance", type=float, default=2e-4)
    parser.add_argument(
        "--output", type=Path, default=Path("private_prediction_verification.json")
    )
    args = parser.parse_args()

    release_root = Path(__file__).resolve().parents[1]
    input_tensor = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if input_tensor.dtype != np.float32:
        raise ValueError("--input must store float32 preprocessed waveform values")
    if input_tensor.ndim != 4 or input_tensor.shape[1:] != (2, 3, 6000):
        raise ValueError(
            "--input must be shaped (N, 2, 3, 6000); "
            f"received {input_tensor.shape}"
        )

    reference = pd.read_csv(args.reference)
    required_columns = ["p_A", "p_B", "p_T"]
    missing = [name for name in required_columns if name not in reference]
    if missing:
        raise ValueError(f"--reference is missing columns: {missing}")
    expected = reference[required_columns].to_numpy(dtype=np.float64)
    if len(expected) != len(input_tensor):
        raise ValueError("--input and --reference must contain the same event count")
    if not np.all(np.isfinite(expected)) or np.any(expected < 0.0):
        raise ValueError("Reference probabilities must be finite and non-negative")
    if not np.allclose(expected.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Reference A/B/T probabilities must sum to one")

    system = SelectiveEnsemble.from_release(release_root, device=args.device)
    reproduced = system.predict(
        input_tensor,
        batch_size=args.batch_size,
        sampling_rate_hz=100.0,
        input_domain="preprocessed_waveform_tensor",
    )
    actual = reproduced[required_columns].to_numpy(dtype=np.float64)
    expected_prediction = expected.argmax(axis=1)
    actual_prediction = actual.argmax(axis=1)
    expected_accept = (
        expected.max(axis=1) >= system.thresholds[expected_prediction]
    )
    actual_accept = reproduced["decision"].eq("accept").to_numpy()
    maximum_difference = float(np.max(np.abs(actual - expected)))

    config_path = release_root / "configs" / "final_w2_temporal.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_hashes = {
        Path(relative_path).name: _sha256_file(release_root / relative_path)
        for relative_path in config["checkpoints"].values()
    }
    result = {
        "verification_scope": (
            "private-derived chronological-test reference; event-level inputs "
            "and reference probabilities are not distributed"
        ),
        "n_events": int(len(input_tensor)),
        "maximum_absolute_probability_difference": maximum_difference,
        "mean_absolute_probability_difference": float(
            np.mean(np.abs(actual - expected))
        ),
        "probability_tolerance": float(args.probability_tolerance),
        "prediction_agreement": float(
            np.mean(actual_prediction == expected_prediction)
        ),
        "accept_refer_agreement": float(np.mean(actual_accept == expected_accept)),
        "expected_coverage": float(np.mean(expected_accept)),
        "reproduced_coverage": float(np.mean(actual_accept)),
        "provenance": {
            "input_file_sha256": _sha256_file(args.input),
            "input_tensor_sha256": _sha256_array(input_tensor),
            "input_shape": list(input_tensor.shape),
            "input_dtype": str(input_tensor.dtype),
            "reference_file_sha256": _sha256_file(args.reference),
            "reference_probability_sha256": _sha256_array(expected),
            "reference_shape": list(expected.shape),
            "reference_dtype": str(expected.dtype),
            "release_config_sha256": _sha256_file(config_path),
            "frozen_checkpoint_sha256": checkpoint_hashes,
            "device": str(system.device),
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
        },
        "status": "pass"
        if (
            maximum_difference <= args.probability_tolerance
            and np.array_equal(actual_prediction, expected_prediction)
            and np.array_equal(actual_accept, expected_accept)
        )
        else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
