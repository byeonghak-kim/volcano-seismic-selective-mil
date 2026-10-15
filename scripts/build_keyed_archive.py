"""Build sanitized W2 arrays from event/record manifests and relative `.npy` files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ve_selective.preprocessing import build_keyed_archive, numpy_record_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = build_keyed_archive(
        pd.read_csv(args.events),
        pd.read_csv(args.records),
        numpy_record_loader(args.records_root),
    )
    bundle.save(args.output)
    print(f"events={len(bundle.events)} excluded_mixed={len(bundle.excluded_mixed_label_events)}")


if __name__ == "__main__":
    main()
