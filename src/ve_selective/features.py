"""The study's 72-per-record and 144-per-event engineered feature recipe."""

from __future__ import annotations

import numpy as np


BANDS_HZ = ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 30.0))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64) - np.mean(left)
    right = np.asarray(right, dtype=np.float64) - np.mean(right)
    left_scale = np.sqrt(np.mean(left * left) + 1e-12)
    right_scale = np.sqrt(np.mean(right * right) + 1e-12)
    return float(np.mean(left * right) / (left_scale * right_scale + 1e-12))


def _time_features(values: np.ndarray) -> list[float]:
    x = np.nan_to_num(np.asarray(values, dtype=np.float64))
    mean = float(np.mean(x))
    standard_deviation = float(np.std(x) + 1e-12)
    standardized = (x - mean) / standard_deviation
    crossings = float(np.mean(np.signbit(x[:-1]) != np.signbit(x[1:])))
    return [
        mean,
        standard_deviation,
        float(np.mean(standardized**3)),
        float(np.mean(standardized**4) - 3.0),
        float(np.sqrt(np.mean(x**2))),
        float(np.max(np.abs(x))),
        float(np.percentile(x, 10)),
        float(np.percentile(x, 50)),
        float(np.percentile(x, 90)),
        crossings,
    ]


def _spectral_features(values: np.ndarray, sampling_rate_hz: float) -> list[float]:
    x = np.nan_to_num(np.asarray(values, dtype=np.float64))
    spectrum = np.fft.rfft(x)
    frequency = np.fft.rfftfreq(len(x), d=1.0 / sampling_rate_hz)
    power = np.abs(spectrum) ** 2 / max(len(x), 1)
    if power.size:
        power[0] = 0.0
    total = float(power.sum())
    if total < 1e-20:
        return [0.0] * 11
    probability = power / (total + 1e-12)
    centroid = float(np.sum(frequency * power) / total)
    bandwidth = float(np.sqrt(np.sum((frequency - centroid) ** 2 * power) / total))
    flatness = float(
        np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12)
    )
    entropy = float(
        -np.sum(probability * np.log(probability + 1e-12))
        / np.log(max(len(probability), 2))
    )
    cumulative = np.cumsum(power)
    rolloff_index = int(np.searchsorted(cumulative, 0.95 * total, side="left"))
    rolloff = float(frequency[min(rolloff_index, len(frequency) - 1)])
    dominant = float(frequency[int(np.argmax(power))])
    band_energy = [
        float(power[(frequency >= low) & (frequency < high)].sum() / total)
        for low, high in BANDS_HZ
    ]
    return [centroid, bandwidth, flatness, entropy, rolloff, dominant, *band_energy]


def _polarization_features(record: np.ndarray) -> list[float]:
    eigenvalues = np.linalg.eigvalsh(np.cov(record))
    eigenvalues = np.sort(eigenvalues)[::-1] + 1e-12
    first, second, third = (float(value) for value in eigenvalues[:3])
    return [
        first,
        second,
        third,
        float(1.0 - (second + third) / (2.0 * first)),
        float((second - third) / first),
        float(np.sqrt(third / first)),
    ]


def extract_record_features(
    record: np.ndarray, sampling_rate_hz: float = 100.0
) -> np.ndarray:
    """Return 72 features for one `(E,N,U/Z) x samples` record."""
    waveform = np.asarray(record, dtype=np.float64)
    if waveform.ndim != 2 or waveform.shape[0] != 3:
        raise ValueError(f"Expected (3, samples), received {waveform.shape}")
    features: list[float] = []
    for component in waveform:
        features.extend(_time_features(component))
        features.extend(_spectral_features(component, sampling_rate_hz))
    east, north, vertical = waveform
    features.extend(
        [
            _safe_correlation(east, north),
            _safe_correlation(east, vertical),
            _safe_correlation(north, vertical),
        ]
    )
    features.extend(_polarization_features(waveform))
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (72,):
        raise RuntimeError(f"Feature recipe produced {result.shape}, expected (72,)")
    return result


def extract_144_features(
    waveforms: np.ndarray,
    quality: np.ndarray,
    sampling_rate_hz: float = 100.0,
) -> np.ndarray:
    """Pool two 72-d record vectors by quality-weighted mean and maximum."""
    array = np.asarray(waveforms)
    q = np.asarray(quality, dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != 2 or array.shape[2] != 3:
        raise ValueError(f"Expected (N,2,3,L), received {array.shape}")
    if q.shape != array.shape[:2]:
        raise ValueError(f"quality must have shape {array.shape[:2]}, received {q.shape}")
    per_record = np.empty((len(array), 2, 72), dtype=np.float32)
    for event in range(len(array)):
        for record in range(2):
            per_record[event, record] = extract_record_features(
                array[event, record], sampling_rate_hz
            )
    weight = np.clip(q, 0.0, None)[..., None]
    weighted_mean = (per_record * weight).sum(axis=1) / np.maximum(
        weight.sum(axis=1), 1e-6
    )
    maximum = np.where(weight > 0.0, per_record, -np.inf).max(axis=1)
    no_weight = q.sum(axis=1) <= 1e-8
    weighted_mean[no_weight] = 0.0
    maximum[no_weight] = 0.0
    return np.concatenate([weighted_mean, maximum], axis=1).astype(np.float32)
