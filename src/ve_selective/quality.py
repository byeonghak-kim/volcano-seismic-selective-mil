"""Deterministic record-informativeness proxy used as a soft attention bias."""

from __future__ import annotations

import numpy as np


def window_quality_score(
    waveform: np.ndarray,
    sampling_rate: float = 100.0,
    edge_margin_seconds: float = 3.0,
    snr_threshold_db: float = 6.0,
    active_multiplier: float = 2.0,
    minimum_active_seconds: float = 0.20,
    frame_seconds: float = 0.50,
    hop_seconds: float = 0.10,
) -> tuple[float, float]:
    """Return a soft informativeness proxy and active duration for one record."""
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != 3:
        raise ValueError(f"Expected (3, samples), received {x.shape}")
    samples = x.shape[1]
    frame = max(1, int(frame_seconds * sampling_rate))
    hop = max(1, int(hop_seconds * sampling_rate))
    edge_margin = int(edge_margin_seconds * sampling_rate)
    minimum_active = int(minimum_active_seconds * sampling_rate)
    magnitude = np.sqrt(np.mean(x**2, axis=0) + 1e-12)
    n_frames = 1 + max(0, (samples - frame) // hop)
    if n_frames <= 0:
        return 0.0, 0.0
    envelope = np.empty(n_frames, dtype=np.float32)
    for index in range(n_frames):
        start = index * hop
        segment = magnitude[start : start + frame]
        envelope[index] = float(np.sqrt(np.mean(segment * segment) + 1e-12))
    baseline = float(np.median(envelope) + 1e-12)
    peak = float(np.max(envelope) + 1e-12)
    snr_db = float(20.0 * np.log10(peak / baseline))
    peak_frame = int(np.argmax(envelope))
    peak_sample = int(peak_frame * hop + frame // 2)
    active = envelope > baseline * active_multiplier
    if np.any(active):
        first = int(np.argmax(active))
        last = int(len(active) - 1 - np.argmax(active[::-1]))
        first_sample = first * hop
        last_sample = last * hop + frame
        active_samples = max(0, last_sample - first_sample)
    else:
        first_sample = last_sample = active_samples = 0
    active_seconds = float(active_samples / sampling_rate)
    penalty = 0.0
    if peak_sample < edge_margin or peak_sample > samples - edge_margin:
        penalty += 0.5
    if active_samples > 0 and (
        first_sample < edge_margin or last_sample > samples - edge_margin
    ):
        penalty += 0.5
    if active_samples < minimum_active:
        penalty += 0.5
    snr_component = np.clip((snr_db - snr_threshold_db) / 10.0, 0.0, 1.0)
    quality = float(np.clip(snr_component * np.exp(-penalty), 0.0, 1.0))
    return quality, active_seconds


def event_quality(waveforms: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Compute model-input quality for `(N, 2, 3, samples)` waveforms."""
    array = np.asarray(waveforms)
    if array.ndim != 4 or array.shape[1:3] != (2, 3):
        raise ValueError(f"Expected (N, 2, 3, samples), received {array.shape}")
    quality = np.empty((len(array), 2), dtype=np.float32)
    for event_index in range(len(array)):
        for window_index in range(2):
            quality[event_index, window_index] = window_quality_score(
                array[event_index, window_index]
            )[0]
    return np.clip(quality, floor, 1.0).astype(np.float32)
