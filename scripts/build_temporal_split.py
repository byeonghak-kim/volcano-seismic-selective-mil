"""Create the overlap-group-aware temporal train/validation/test manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ve_selective.splitting import build_temporal_group_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build_temporal_group_split(pd.read_csv(args.events), args.cutoff, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
