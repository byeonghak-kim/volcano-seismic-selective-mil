"""Selective metrics, risk-coverage summaries, and resampling controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def classification_metrics(labels: np.ndarray, prediction: np.ndarray, n_classes: int = 3) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    if len(y) == 0:
        empty = np.full(n_classes, np.nan)
        return {"accuracy": np.nan, "macro_f1": np.nan, "precision": empty, "recall": empty.copy(), "f1": empty.copy(), "confusion": np.zeros((n_classes, n_classes), dtype=int)}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=np.arange(n_classes), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(n_classes), average="macro", zero_division=0)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": confusion_matrix(y, pred, labels=np.arange(n_classes)),
    }


def evaluate_selective(labels: np.ndarray, probability: np.ndarray, accepted: np.ndarray) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    accept = np.asarray(accepted, dtype=bool)
    prediction = p.argmax(axis=1)
    full = classification_metrics(y, prediction, p.shape[1])
    selected = classification_metrics(y[accept], prediction[accept], p.shape[1])
    errors = prediction != y
    class_coverage = np.asarray([_divide(np.sum(accept & (y == c)), np.sum(y == c)) for c in range(p.shape[1])])
    automatic_recall = np.asarray([_divide(np.sum(accept & (y == c) & (prediction == c)), np.sum(y == c)) for c in range(p.shape[1])])
    return {
        "coverage": float(accept.mean()),
        "accepted_accuracy": selected["accuracy"],
        "accepted_macro_f1": selected["macro_f1"],
        "accepted_precision": selected["precision"],
        "accepted_recall": selected["recall"],
        "accepted_f1": selected["f1"],
        "accepted_confusion": selected["confusion"],
        "full_accuracy": full["accuracy"],
        "full_macro_f1": full["macro_f1"],
        "full_precision": full["precision"],
        "full_recall": full["recall"],
        "full_f1": full["f1"],
        "full_confusion": full["confusion"],
        "class_coverage": class_coverage,
        "class_automatic_recall": automatic_recall,
        "error_capture": _divide(np.sum((~accept) & errors), np.sum(errors)),
    }


def risk_coverage_curve(labels: np.ndarray, probability: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    prediction = p.argmax(axis=1)
    order = np.argsort(-p.max(axis=1), kind="stable")
    errors = (prediction[order] != y[order]).astype(float)
    accepted_count = np.arange(1, len(y) + 1)
    coverage = accepted_count / len(y)
    cumulative_error = np.cumsum(errors)
    selective_risk = cumulative_error / accepted_count
    generalized_risk = cumulative_error / len(y)
    frame = pd.DataFrame({"accepted_count": accepted_count, "coverage": coverage, "selective_risk": selective_risk, "generalized_risk": generalized_risk})
    return frame, {
        "aurc": float(np.mean(selective_risk)),
        "augrc": float((getattr(np, "trapezoid", None) or np.trapz)(np.r_[0.0, generalized_risk], np.r_[0.0, coverage])),
    }


def exact_top_count(probability: np.ndarray, accepted_count: int) -> np.ndarray:
    """Label-free matched-count diagnostic based only on confidence rank."""
    p = np.asarray(probability, dtype=np.float64)
    count = int(np.clip(accepted_count, 0, len(p)))
    accepted = np.zeros(len(p), dtype=bool)
    accepted[np.argsort(-p.max(axis=1), kind="stable")[:count]] = True
    return accepted


def _selective_metric_bundle(
    labels: np.ndarray, probability: np.ndarray, accepted: np.ndarray
) -> dict[str, float]:
    """Return the complete manuscript-facing selective metric bundle."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    accept = np.asarray(accepted, dtype=bool)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("The manuscript metric bundle requires A/B/T probabilities")
    if len(y) != len(p) or len(accept) != len(y):
        raise ValueError("labels, probability, and accepted must have equal length")

    result = evaluate_selective(y, p, accept)
    errors = p.argmax(axis=1) != y
    accepted_n = int(np.sum(accept))
    referred_n = int(np.sum(~accept))
    accepted_errors = int(np.sum(accept & errors))
    referred_errors = int(np.sum((~accept) & errors))
    accepted_error_rate = _divide(accepted_errors, accepted_n)
    referral_error_rate = _divide(referred_errors, referred_n)

    bundle = {
        "coverage": result["coverage"],
        "accepted_accuracy": result["accepted_accuracy"],
        "accepted_macro_f1": result["accepted_macro_f1"],
        "generalized_risk": _divide(accepted_errors, len(y)),
        "error_capture": result["error_capture"],
        "referral_error_enrichment": _divide(
            referral_error_rate, accepted_error_rate
        ),
    }
    for class_index, class_name in enumerate(("A", "B", "T")):
        bundle.update(
            {
                f"{class_name}_accepted_precision": float(
                    result["accepted_precision"][class_index]
                ),
                f"{class_name}_accepted_recall": float(
                    result["accepted_recall"][class_index]
                ),
                f"{class_name}_accepted_f1": float(
                    result["accepted_f1"][class_index]
                ),
                f"{class_name}_coverage": float(
                    result["class_coverage"][class_index]
                ),
                f"{class_name}_automatic_recall": float(
                    result["class_automatic_recall"][class_index]
                ),
            }
        )
    bundle.update(
        {
            "full_accuracy": result["full_accuracy"],
            "full_macro_f1": result["full_macro_f1"],
        }
    )
    return bundle


def _random_referral_endpoint_bundle(
    labels: np.ndarray, probability: np.ndarray, accepted: np.ndarray
) -> dict[str, float]:
    """Return the nine endpoints used by the random-referral controls."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    accept = np.asarray(accepted, dtype=bool)
    prediction = p.argmax(axis=1)
    errors = prediction != y
    accepted_n = int(np.sum(accept))
    referred_n = int(np.sum(~accept))
    accepted_errors = int(np.sum(accept & errors))
    referred_errors = int(np.sum((~accept) & errors))
    accepted_error_rate = _divide(accepted_errors, accepted_n)
    referral_error_rate = _divide(referred_errors, referred_n)
    if accepted_n:
        confusion = np.bincount(
            3 * y[accept] + prediction[accept], minlength=9
        ).reshape(3, 3)
        true_positive = np.diag(confusion).astype(float)
        f1_denominator = (
            2.0 * true_positive
            + confusion.sum(axis=0)
            - true_positive
            + confusion.sum(axis=1)
            - true_positive
        )
        class_f1 = np.divide(
            2.0 * true_positive,
            f1_denominator,
            out=np.zeros(3, dtype=float),
            where=f1_denominator != 0.0,
        )
        accepted_macro_f1 = float(np.mean(class_f1))
    else:
        accepted_macro_f1 = float("nan")
    return {
        "coverage": float(np.mean(accept)),
        "accepted_accuracy": _divide(accepted_n - accepted_errors, accepted_n),
        "accepted_macro_f1": accepted_macro_f1,
        "error_capture": _divide(referred_errors, accepted_errors + referred_errors),
        "accepted_error_rate": accepted_error_rate,
        "referral_error_rate": referral_error_rate,
        "referral_error_enrichment": _divide(
            referral_error_rate, accepted_error_rate
        ),
        "accepted_errors": float(accepted_errors),
        "referred_errors": float(referred_errors),
    }


def group_bootstrap(
    labels: np.ndarray,
    probability: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    repetitions: int = 10_000,
    seed: int = 20260810,
) -> pd.DataFrame:
    """Overlap-group bootstrap for the complete S4 metric bundle."""
    y = np.asarray(labels, dtype=np.int64)
    group = np.asarray(groups)
    unique_group = np.unique(group)
    group_class = {}
    lookup = {}
    for value in unique_group:
        index = np.flatnonzero(group == value)
        classes = np.unique(y[index])
        if len(classes) != 1:
            raise ValueError("Bootstrap groups must be class-pure")
        group_class[value] = int(classes[0])
        lookup[value] = index
    groups_by_class = {c: np.asarray([g for g in unique_group if group_class[g] == c]) for c in np.unique(y)}
    observed = _selective_metric_bundle(y, probability, accepted)
    metrics = tuple(observed)
    samples = {metric: np.empty(repetitions, dtype=float) for metric in metrics}
    rng = np.random.default_rng(seed)
    for repetition in range(repetitions):
        index_parts = []
        for class_groups in groups_by_class.values():
            chosen = rng.choice(class_groups, len(class_groups), replace=True)
            index_parts.extend(lookup[value] for value in chosen)
        sampled = np.concatenate(index_parts)
        result = _selective_metric_bundle(
            y[sampled], probability[sampled], accepted[sampled]
        )
        for metric in metrics:
            samples[metric][repetition] = result[metric]
    rows = []
    for metric in metrics:
        finite = samples[metric][np.isfinite(samples[metric])]
        low = float(np.quantile(finite, 0.025)) if len(finite) else float("nan")
        high = float(np.quantile(finite, 0.975)) if len(finite) else float("nan")
        rows.append({"metric": metric, "estimate": observed[metric], "ci95_low": low, "ci95_high": high, "finite_repetitions": len(finite), "bootstrap_repetitions": repetitions, "bootstrap_unit": "overlap_group_stratified_by_true_class"})
    return pd.DataFrame(rows)


def random_referral_controls(
    labels: np.ndarray,
    probability: np.ndarray,
    accepted: np.ndarray,
    repetitions: int = 10_000,
    seed: int = 20260810,
) -> pd.DataFrame:
    """Reproduce the three S12 random-routing controls and nine endpoints."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    observed_mask = np.asarray(accepted, dtype=bool)
    prediction = p.argmax(axis=1)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Random-referral controls require A/B/T probabilities")
    if len(y) != len(p) or len(observed_mask) != len(y):
        raise ValueError("labels, probability, and accepted must have equal length")
    observed = _random_referral_endpoint_bundle(y, p, observed_mask)
    accepted_n = int(np.sum(observed_mask))
    schemes = {
        "uniform_random": (np.zeros(len(y), dtype=int), {0: accepted_n}),
        "true_class_count_matched": (
            y,
            {
                c: int(np.sum(observed_mask & (y == c)))
                for c in range(p.shape[1])
            },
        ),
        "predicted_class_count_matched": (
            prediction,
            {
                c: int(np.sum(observed_mask & (prediction == c)))
                for c in range(p.shape[1])
            },
        ),
    }
    metrics = tuple(observed)
    upper_tail_metrics = {
        "accepted_accuracy",
        "accepted_macro_f1",
        "error_capture",
        "referral_error_rate",
        "referral_error_enrichment",
        "referred_errors",
    }
    rng = np.random.default_rng(seed)
    rows = []
    for scheme, (strata, counts) in schemes.items():
        distribution = {metric: np.empty(repetitions) for metric in metrics}
        for repetition in range(repetitions):
            mask = np.zeros(len(y), dtype=bool)
            for stratum, count in counts.items():
                candidates = np.flatnonzero(strata == stratum)
                if count:
                    mask[rng.choice(candidates, count, replace=False)] = True
            result = _random_referral_endpoint_bundle(y, p, mask)
            for metric in metrics:
                distribution[metric][repetition] = result[metric]
        for metric in metrics:
            values = distribution[metric]
            finite = values[np.isfinite(values)]
            if metric in upper_tail_metrics:
                exceedances = int(np.sum(values >= observed[metric]))
            else:
                exceedances = int(np.sum(values <= observed[metric]))
            rows.append(
                {
                    "scheme": scheme,
                    "metric": metric,
                    "observed_confidence_policy": observed[metric],
                    "random_mean": float(np.mean(finite))
                    if len(finite)
                    else float("nan"),
                    "random_ci95_low": float(np.quantile(finite, 0.025))
                    if len(finite)
                    else float("nan"),
                    "random_ci95_high": float(np.quantile(finite, 0.975))
                    if len(finite)
                    else float("nan"),
                    "one_sided_monte_carlo_p": (exceedances + 1)
                    / (repetitions + 1),
                    "repetitions": repetitions,
                    "accepted_n": accepted_n,
                    "accepted_counts_by_stratum": str(counts),
                }
            )
    return pd.DataFrame(rows)
