"""Validation-only probability calibration utilities."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss


def softmax_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    score = np.asarray(logits, dtype=np.float64) / float(temperature)
    score -= score.max(axis=1, keepdims=True)
    probability = np.exp(score)
    return probability / probability.sum(axis=1, keepdims=True)


def fit_temperature(
    validation_labels: np.ndarray,
    validation_logits: np.ndarray,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """Minimize multiclass NLL using validation rows only."""
    y = np.asarray(validation_labels, dtype=np.int64)
    logits = np.asarray(validation_logits, dtype=np.float64)
    classes = np.arange(logits.shape[1])
    result = minimize_scalar(
        lambda value: log_loss(y, softmax_logits(logits, value), labels=classes),
        bounds=bounds,
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(result.x)


def calibrate_seed_ensemble(
    logits_by_seed: list[np.ndarray],
    labels: np.ndarray,
    validation_index: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """Fit one validation temperature per seed and average probabilities."""
    temperatures = [
        fit_temperature(labels[validation_index], logits[validation_index])
        for logits in logits_by_seed
    ]
    probability = np.mean(
        [softmax_logits(logits, temperature) for logits, temperature in zip(logits_by_seed, temperatures)],
        axis=0,
    )
    return probability / probability.sum(axis=1, keepdims=True), temperatures
