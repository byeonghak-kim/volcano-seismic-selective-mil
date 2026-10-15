"""Calibration and validation-fixed selective-decision utilities."""

from __future__ import annotations

import numpy as np


def apply_class_probability_multiplier(
    probability: np.ndarray,
    class_index: int,
    multiplier: float,
) -> np.ndarray:
    """Multiply one class probability and renormalize every event distribution."""
    corrected = np.asarray(probability, dtype=np.float64).copy()
    if corrected.ndim != 2:
        raise ValueError("probability must have shape (events, classes)")
    if not 0 <= int(class_index) < corrected.shape[1]:
        raise IndexError("class_index is outside the probability matrix")
    if not np.isfinite(multiplier) or float(multiplier) <= 0.0:
        raise ValueError("multiplier must be finite and positive")
    corrected[:, int(class_index)] *= float(multiplier)
    normalizer = corrected.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
        raise ValueError("probability rows must have a finite positive sum")
    return corrected / normalizer


def temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    logit = np.log(np.clip(probability, 1e-12, 1.0)) / float(temperature)
    logit -= logit.max(axis=1, keepdims=True)
    calibrated = np.exp(logit)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def confidence_score(probability: np.ndarray, kind: str = "msp") -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if kind == "msp":
        return probability.max(axis=1)
    if kind == "margin":
        ordered = np.sort(probability, axis=1)
        return ordered[:, -1] - ordered[:, -2]
    if kind == "negative_entropy":
        return np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1)
    raise ValueError(f"Unknown confidence score: {kind}")


def class_specific_acceptance(
    probability: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    accepted = confidence >= thresholds[prediction]
    return prediction, confidence, accepted
