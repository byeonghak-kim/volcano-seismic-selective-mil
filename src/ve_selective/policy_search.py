"""Validation-only probability adjustment and accept/refer policy search."""

from __future__ import annotations

from itertools import product

import numpy as np

from .evaluation import classification_metrics, evaluate_selective
from .selective import apply_class_probability_multiplier


def threshold_for_coverage(score: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(score, dtype=float))[::-1]
    if len(values) == 0:
        return float("inf")
    count = max(1, min(len(values), int(np.rint(float(coverage) * len(values)))))
    return float(values[count - 1])


def fit_global_threshold(probability: np.ndarray, target_coverage: float) -> float:
    return threshold_for_coverage(np.asarray(probability).max(axis=1), target_coverage)


def fit_class_probability_adjustment(
    validation_labels: np.ndarray,
    validation_probability: np.ndarray,
    class_index: int = 0,
    precision_floor: float = 0.98,
    multipliers: np.ndarray | None = None,
) -> dict:
    """Fit a constant class probability multiplier on validation rows only."""
    y = np.asarray(validation_labels, dtype=np.int64)
    p = np.asarray(validation_probability, dtype=np.float64)
    grid = np.geomspace(0.01, 2.0, 2001) if multipliers is None else np.asarray(multipliers)
    best = None
    for multiplier in grid:
        adjusted = apply_class_probability_multiplier(p, class_index, float(multiplier))
        metrics = classification_metrics(y, adjusted.argmax(axis=1), p.shape[1])
        if metrics["precision"][class_index] + 1e-12 < precision_floor:
            continue
        ranking = (metrics["macro_f1"], metrics["accuracy"], metrics["f1"][class_index], metrics["recall"][class_index])
        if best is None or ranking > best[0]:
            best = (ranking, float(multiplier), metrics)
    if best is None:
        raise RuntimeError("No class probability adjustment met the validation precision constraint")
    return {"class_index": class_index, "multiplier": best[1], "precision_constraint": precision_floor, "validation_metrics": best[2]}


def fit_class_specific_thresholds(
    validation_labels: np.ndarray,
    validation_probability: np.ndarray,
    target_coverage: float,
    coverage_tolerance: float = 0.02,
    rate_grid: np.ndarray | None = None,
) -> dict:
    """Shared validation search with identical candidate-rate grids per class."""
    y = np.asarray(validation_labels, dtype=np.int64)
    p = np.asarray(validation_probability, dtype=np.float64)
    prediction = p.argmax(axis=1)
    confidence = p.max(axis=1)
    rates = np.arange(0.30, 1.0001, 0.05) if rate_grid is None else np.asarray(rate_grid)
    cuts = {c: {float(rate): threshold_for_coverage(confidence[prediction == c], float(rate)) for rate in rates} for c in range(p.shape[1])}
    best = None
    for class_rates in product(rates, repeat=p.shape[1]):
        thresholds = np.asarray([cuts[c][float(class_rates[c])] for c in range(p.shape[1])])
        accepted = confidence >= thresholds[prediction]
        if abs(float(accepted.mean()) - target_coverage) > coverage_tolerance:
            continue
        result = evaluate_selective(y, p, accepted)
        ranking = (0.5 * (result["accepted_accuracy"] + result["accepted_macro_f1"]), float(np.nanmin(result["accepted_f1"])), -abs(result["coverage"] - target_coverage), -float(np.std(class_rates)))
        if best is None or ranking > best[0]:
            best = (ranking, np.asarray(class_rates), thresholds, result)
    if best is None:
        raise RuntimeError("No class-specific threshold policy met the coverage tolerance")
    return {"rates": best[1], "thresholds": best[2], "validation_metrics": best[3], "nominal_target_coverage": target_coverage}


def fit_performance_constrained_policy(
    validation_labels: np.ndarray,
    validation_probability: np.ndarray,
    minimum_accuracy: float = 0.96,
    minimum_macro_f1: float = 0.95,
    coverage_targets: np.ndarray | None = None,
    coverage_tolerance: float = 0.02,
    rate_grid: np.ndarray | None = None,
) -> dict:
    """Maximize validation coverage subject to accepted-performance constraints."""
    targets = np.arange(0.45, 0.8001, 0.05) if coverage_targets is None else np.asarray(coverage_targets)
    candidates = [fit_class_specific_thresholds(validation_labels, validation_probability, float(target), coverage_tolerance, rate_grid) for target in targets]
    feasible = [candidate for candidate in candidates if candidate["validation_metrics"]["accepted_accuracy"] >= minimum_accuracy and candidate["validation_metrics"]["accepted_macro_f1"] >= minimum_macro_f1]
    if not feasible:
        raise RuntimeError("No policy met the validation performance constraints")
    best = max(feasible, key=lambda item: (item["validation_metrics"]["coverage"], item["validation_metrics"]["accepted_accuracy"], item["validation_metrics"]["accepted_macro_f1"]))
    return {**best, "minimum_accuracy": minimum_accuracy, "minimum_macro_f1": minimum_macro_f1}


def apply_thresholds(probability: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    prediction = p.argmax(axis=1)
    return p.max(axis=1) >= np.asarray(thresholds)[prediction]
