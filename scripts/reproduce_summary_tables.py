"""Regenerate concise manuscript tables from supplied aggregate artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary = root / "artifacts" / "summary"
    args.output.mkdir(parents=True, exist_ok=True)

    primary = pd.read_csv(summary / "primary_policy_metrics_v36.csv")
    primary.to_csv(args.output / "table_primary_policy_v36.csv", index=False)

    comparison = pd.read_csv(summary / "selective_model_comparison_v37.csv")
    comparison.to_csv(
        args.output / "table_selective_model_comparison_v37.csv", index=False
    )

    policies = pd.read_csv(summary / "validation_locked_policies_v37.csv")
    policies.to_csv(args.output / "table_validation_locked_policies_v37.csv", index=False)

    bootstrap = pd.read_csv(summary / "primary_group_bootstrap_ci_v38.csv")
    bootstrap.to_csv(args.output / "table_primary_group_bootstrap_ci_v38.csv", index=False)

    adjustment = pd.read_csv(summary / "class_A_adjustment_ablation_v39.csv")
    adjustment.to_csv(
        args.output / "table_class_A_adjustment_ablation_v39.csv", index=False
    )
    print(primary.to_string(index=False))
    print(f"Saved tables to {args.output.resolve()}")


if __name__ == "__main__":
    main()
