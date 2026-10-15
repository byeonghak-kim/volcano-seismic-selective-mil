from __future__ import annotations

import numpy as np
import pandas as pd

from ve_selective.evaluation import exact_top_count, group_bootstrap, random_referral_controls, risk_coverage_curve
from ve_selective.features import extract_144_features, extract_record_features
from ve_selective.policy_search import apply_thresholds, fit_class_probability_adjustment, fit_class_specific_thresholds, fit_global_threshold
from ve_selective.preprocessing import PreprocessingConfig, build_keyed_archive, collapse_catalogue_label, preprocess_component
from ve_selective.splitting import build_temporal_group_split


def test_component_preprocessing_and_keyed_overlap_archive() -> None:
    time = np.arange(3000) / 100.0
    record = np.stack([np.sin(2 * np.pi * frequency * time) for frequency in (2.0, 3.0, 4.0)]).astype(np.float32)
    processed = preprocess_component(record[0] + 10.0, 100.0, PreprocessingConfig())
    assert processed.shape == (6000,)
    assert processed.dtype == np.float32
    events = pd.DataFrame(
        [
            {"event_id": "e0", "label": "A", "event_time": "2023-01-01", "first_record_key": "r0", "second_record_key": "r1"},
            {"event_id": "e1", "label": "A", "event_time": "2023-01-02", "first_record_key": "r1", "second_record_key": "r2"},
            {"event_id": "e2", "label": "B", "event_time": "2023-10-01", "first_record_key": "r3", "second_record_key": "r4"},
        ]
    )
    records = pd.DataFrame([{"record_key": f"r{i}", "array_path": f"r{i}.npy", "sampling_rate_hz": 100.0} for i in range(5)])
    bundle = build_keyed_archive(events, records, lambda row: record)
    assert bundle.waveforms.shape == (3, 2, 3, 6000)
    assert bundle.events.loc[0, "group_id"] == bundle.events.loc[1, "group_id"]
    assert bundle.events.loc[2, "group_id"] != bundle.events.loc[0, "group_id"]
    assert [collapse_catalogue_label(value) for value in ("BH", "BT", "T(c)", "TP")] == ["B", "B", "T", "T"]


def test_feature_dimension_and_raw_zero_quality_pooling() -> None:
    rng = np.random.default_rng(7)
    waveform = rng.normal(size=(1, 2, 3, 6000)).astype(np.float32)
    first = extract_record_features(waveform[0, 0])
    expected = np.concatenate([first, first])
    features = extract_144_features(waveform, np.asarray([[1.0, 0.0]], dtype=np.float32))
    assert features.shape == (1, 144)
    assert np.allclose(features[0], expected)


def test_temporal_group_split_keeps_overlap_components_intact() -> None:
    rows = []
    for class_index, label in enumerate(("A", "B", "T")):
        for group in range(10):
            rows.append({"event_id": f"{label}{group}", "event_time": "2023-10-01" if group >= 8 else f"2023-0{class_index + 1}-{group + 1:02d}", "label": label, "group_id": class_index * 100 + group})
    split = build_temporal_group_split(pd.DataFrame(rows), "2023-10-01", validation_folds=4)
    assert set(split["split"]) == {"train", "validation", "test"}
    assert split.groupby("group_id")["split"].nunique().max() == 1
    assert split.loc[split["split"].eq("test"), "event_time"].min() >= pd.Timestamp("2023-10-01")


def test_policy_risk_matched_count_and_resampling_paths() -> None:
    y = np.tile(np.arange(3), 12)
    probability = np.full((len(y), 3), 0.1)
    probability[np.arange(len(y)), y] = np.linspace(0.55, 0.90, len(y))
    probability /= probability.sum(axis=1, keepdims=True)
    adjustment = fit_class_probability_adjustment(y, probability, precision_floor=0.0, multipliers=np.asarray([0.5, 1.0]))
    assert adjustment["multiplier"] in {0.5, 1.0}
    global_threshold = fit_global_threshold(probability, 0.5)
    assert np.isfinite(global_threshold)
    policy = fit_class_specific_thresholds(y, probability, 0.5, 0.2, np.asarray([0.5, 1.0]))
    accepted = apply_thresholds(probability, policy["thresholds"])
    assert accepted.any()
    matched = exact_top_count(probability, int(accepted.sum()))
    assert matched.sum() == accepted.sum()
    curve, summary = risk_coverage_curve(y, probability)
    assert len(curve) == len(y) and set(summary) == {"aurc", "augrc"}

    resampling_probability = np.full((len(y), 3), 0.05)
    resampling_probability[np.arange(len(y)), y] = 0.90
    error_index = np.r_[0:6, 24:30]
    resampling_probability[error_index] = 0.05
    resampling_probability[error_index, (y[error_index] + 1) % 3] = 0.90
    resampling_accepted = np.zeros(len(y), dtype=bool)
    resampling_accepted[:18] = True
    groups = np.arange(len(y))
    bootstrap = group_bootstrap(
        y,
        resampling_probability,
        resampling_accepted,
        groups,
        repetitions=20,
    )
    expected_bootstrap_metrics = {
        "coverage",
        "accepted_accuracy",
        "accepted_macro_f1",
        "generalized_risk",
        "error_capture",
        "referral_error_enrichment",
        "full_accuracy",
        "full_macro_f1",
    }
    expected_bootstrap_metrics.update(
        f"{class_name}_{metric}"
        for class_name in ("A", "B", "T")
        for metric in (
            "accepted_precision",
            "accepted_recall",
            "accepted_f1",
            "coverage",
            "automatic_recall",
        )
    )
    assert len(bootstrap) == 23
    assert set(bootstrap["metric"]) == expected_bootstrap_metrics
    estimates = bootstrap.set_index("metric")["estimate"]
    assert np.isclose(estimates["generalized_risk"], 6 / 36)
    assert np.isclose(estimates["referral_error_enrichment"], 1.0)
    assert (bootstrap["bootstrap_repetitions"] == 20).all()
    assert (bootstrap["finite_repetitions"] <= 20).all()

    controls = random_referral_controls(
        y,
        resampling_probability,
        resampling_accepted,
        repetitions=20,
    )
    expected_schemes = {
        "uniform_random",
        "true_class_count_matched",
        "predicted_class_count_matched",
    }
    expected_endpoints = {
        "coverage",
        "accepted_accuracy",
        "accepted_macro_f1",
        "error_capture",
        "accepted_error_rate",
        "referral_error_rate",
        "referral_error_enrichment",
        "accepted_errors",
        "referred_errors",
    }
    assert len(controls) == 27
    assert set(controls["scheme"]) == expected_schemes
    assert set(controls["metric"]) == expected_endpoints
    assert (controls.groupby("scheme").size() == 9).all()
    assert (controls["accepted_n"] == 18).all()
    assert controls["accepted_counts_by_stratum"].notna().all()
    assert controls["one_sided_monte_carlo_p"].between(1 / 21, 1).all()
    assert np.allclose(
        controls["one_sided_monte_carlo_p"] * 21,
        np.round(controls["one_sided_monte_carlo_p"] * 21),
    )
    assert (
        controls.loc[controls["metric"].eq("coverage"), "one_sided_monte_carlo_p"]
        == 1.0
    ).all()
    observed = controls.set_index(["scheme", "metric"])[
        "observed_confidence_policy"
    ]
    for scheme in expected_schemes:
        assert np.isclose(observed[scheme, "accepted_errors"], 6)
        assert np.isclose(observed[scheme, "referred_errors"], 6)
        assert np.isclose(observed[scheme, "accepted_error_rate"], 1 / 3)
        assert np.isclose(observed[scheme, "referral_error_rate"], 1 / 3)
