"""Train and evaluate the sanitized research workflow from a built archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ve_selective.pipeline import run_research_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/research_pipeline.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_research_pipeline(
        np.load(args.archive / "waveforms.npy", allow_pickle=False),
        np.load(args.archive / "labels.npy", allow_pickle=False),
        pd.read_csv(args.split_manifest),
        args.output,
        config,
        quick=args.quick,
    )
    print(json.dumps({"status": summary["status"], "quick_mode": summary["quick_mode"]}, indent=2))


if __name__ == "__main__":
    main()
