"""Orchestration for the sanitized training, comparison, and policy workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import baseline_logits, fit_feature_baselines
from .calibration import fit_temperature, softmax_logits
from .evaluation import exact_top_count, group_bootstrap, random_referral_controls, risk_coverage_curve, evaluate_selective
from .features import extract_144_features
from .policy_search import apply_thresholds, fit_class_probability_adjustment, fit_class_specific_thresholds, fit_global_threshold, fit_performance_constrained_policy
from .quality import event_quality
from .selective import apply_class_probability_multiplier
from .splitting import split_indices
from .training import TrainingConfig, fit_cnn120, fit_w2_model


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_research_pipeline(
    waveforms: np.ndarray,
    labels: np.ndarray,
    split_manifest: pd.DataFrame,
    output_dir: Path,
    config: dict,
    quick: bool = False,
) -> dict:
    """Run W2/CNN/classical fitting and validation-locked selective analysis."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    indices = split_indices(split_manifest)
    raw_quality = event_quality(waveforms, floor=0.0)
    model_quality = np.clip(
        raw_quality, float(config["quality_floor"]), 1.0
    ).astype(np.float32)
    train_config = config["training"]
    seeds = list(train_config["seeds"])
    if quick:
        seeds = seeds[:1]
    w2_fits = []
    for seed in seeds:
        fit_config = TrainingConfig(
            seed=int(seed),
            epochs=1 if quick else int(train_config["epochs"]),
            batch_size=int(train_config["batch_size"]),
            learning_rate=float(train_config["learning_rate"]),
            device=str(train_config.get("device", "cpu")),
            max_batches_per_epoch=1 if quick else None,
        )
        w2_fits.append(
            fit_w2_model(
                waveforms,
                model_quality,
                labels,
                indices["train"],
                indices["validation"],
                fit_config,
                output / "models" / f"w2_seed{seed}",
            )
        )
    w2_probability = np.mean([fit.probability for fit in w2_fits], axis=0)
    w2_probability /= w2_probability.sum(axis=1, keepdims=True)

    policy_config = config["policy"]
    validation = indices["validation"]
    test = indices["test"]
    adjustment = fit_class_probability_adjustment(
        labels[validation],
        w2_probability[validation],
        class_index=0,
        precision_floor=float(policy_config["class_A_precision_constraint"]),
        multipliers=(np.geomspace(0.05, 2.0, 51) if quick else None),
    )
    adjusted = apply_class_probability_multiplier(
        w2_probability, 0, adjustment["multiplier"]
    )
    rate_grid = np.asarray(policy_config.get("rate_grid", np.arange(0.30, 1.0001, 0.05)))
    class_policy = fit_class_specific_thresholds(
        labels[validation],
        adjusted[validation],
        float(policy_config["target_coverage"]),
        float(policy_config["coverage_tolerance"]),
        rate_grid,
    )
    constrained = fit_performance_constrained_policy(
        labels[validation],
        adjusted[validation],
        float(policy_config["minimum_accepted_accuracy"]),
        float(policy_config["minimum_accepted_macro_f1"]),
        np.asarray(policy_config["coverage_targets"]),
        float(policy_config["coverage_tolerance"]),
        rate_grid,
    )
    global_threshold = fit_global_threshold(
        adjusted[validation], float(policy_config["target_coverage"])
    )
    accepted = apply_thresholds(adjusted, constrained["thresholds"])
    test_metrics = evaluate_selective(labels[test], adjusted[test], accepted[test])
    curve, curve_summary = risk_coverage_curve(labels[test], adjusted[test])
    curve.to_csv(output / "risk_coverage_curve.csv", index=False)
    matched = exact_top_count(adjusted[test], int(accepted[test].sum()))
    matched_metrics = evaluate_selective(labels[test], adjusted[test], matched)
    repetitions = int(config["resampling"]["repetitions"])
    bootstrap_results = group_bootstrap(
        labels[test],
        adjusted[test],
        accepted[test],
        split_manifest.iloc[test]["group_id"].to_numpy(),
        repetitions,
        int(config["resampling"]["seed"]),
    )
    bootstrap_results.to_csv(output / "group_bootstrap.csv", index=False)
    random_control_results = random_referral_controls(
        labels[test], adjusted[test], accepted[test], repetitions, int(config["resampling"]["seed"])
    )
    random_control_results.to_csv(output / "random_referral_controls.csv", index=False)

    features = extract_144_features(waveforms, raw_quality)
    baseline_seed = int(config.get("baseline_training", {}).get("seed", 42))
    baseline_fit = fit_feature_baselines(
        features, labels, indices["train"], validation, seed=baseline_seed, quick=quick
    )
    baseline_fit.validation_search.to_csv(output / "baseline_validation_search.csv", index=False)
    probability_sets = {
        "W2 MIL (class-A probability adjustment)": adjusted,
    }
    model_temperatures = {
        "W2 MIL (class-A probability adjustment)": [fit.temperature for fit in w2_fits]
    }
    for name, model in baseline_fit.models.items():
        logits = baseline_logits(model, features)
        temperature = fit_temperature(labels[validation], logits[validation])
        probability = softmax_logits(logits, temperature)
        display_name = f"{name} (144 features)"
        probability_sets[display_name] = probability
        model_temperatures[display_name] = [temperature]

    cnn_seed_config = list(config["cnn_training"]["seeds"])
    if quick:
        cnn_seed_config = cnn_seed_config[:1]
    for recipe in ("standard", "imbalance_adjusted"):
        recipe_config = config["cnn_training"]["recipes"][recipe]
        fits = []
        for seed in cnn_seed_config:
            cnn_config = TrainingConfig(
                seed=int(seed),
                epochs=1 if quick else int(config["cnn_training"]["epochs"]),
                batch_size=int(recipe_config["batch_size"]),
                learning_rate=float(recipe_config["learning_rate"]),
                weight_decay=float(recipe_config["weight_decay"]),
                device=str(train_config.get("device", "cpu")),
                max_batches_per_epoch=1 if quick else None,
                patience=int(recipe_config.get("early_stopping_patience") or 10**9),
            )
            fits.append(
                fit_cnn120(
                    waveforms,
                    labels,
                    indices["train"],
                    validation,
                    recipe,
                    cnn_config,
                    output / "models" / f"cnn120_{recipe}_seed{seed}",
                )
            )
        probability = np.mean([fit.probability for fit in fits], axis=0)
        probability /= probability.sum(axis=1, keepdims=True)
        display_name = (
            "120-s concatenation CNN (standard training)"
            if recipe == "standard"
            else "120-s concatenation CNN (imbalance-adjusted training)"
        )
        probability_sets[display_name] = probability
        model_temperatures[display_name] = [fit.temperature for fit in fits]

    comparison_rows = []
    policy_rows = []
    curve_rows = []
    matched_count = min(
        int(config["resampling"].get("matched_count_temporal_test", int(accepted[test].sum()))),
        len(test),
    )
    for name, probability in probability_sets.items():
        curve_frame, model_curve_summary = risk_coverage_curve(labels[test], probability[test])
        curve_frame.insert(0, "model", name)
        curve_rows.append(curve_frame)

        policies = []
        threshold = fit_global_threshold(
            probability[validation], float(policy_config["target_coverage"])
        )
        global_accepted = probability[test].max(axis=1) >= threshold
        policies.append(("validation-locked global MSP", global_accepted, {"threshold": threshold}))

        try:
            shared = fit_class_specific_thresholds(
                labels[validation],
                probability[validation],
                float(policy_config["target_coverage"]),
                float(policy_config["coverage_tolerance"]),
                rate_grid,
            )
        except RuntimeError:
            policy_rows.append({"model": name, "policy": "validation-locked class-specific", "status": "infeasible_on_validation", "test_labels_used_for_fitting": False})
        else:
            shared_accepted = apply_thresholds(probability[test], shared["thresholds"])
            policies.append(("validation-locked class-specific", shared_accepted, {"threshold_A": shared["thresholds"][0], "threshold_B": shared["thresholds"][1], "threshold_T": shared["thresholds"][2]}))

        matched_accepted = exact_top_count(probability[test], matched_count)
        policies.append((f"label-free exact {matched_count}/{len(test)} diagnostic", matched_accepted, {}))
        for policy_name, model_accepted, parameters in policies:
            metrics = evaluate_selective(labels[test], probability[test], model_accepted)
            row = {
                "model": name,
                "policy": policy_name,
                "coverage": metrics["coverage"],
                "accepted_accuracy": metrics["accepted_accuracy"],
                "accepted_macro_f1": metrics["accepted_macro_f1"],
                "error_capture": metrics["error_capture"],
                "full_accuracy": metrics["full_accuracy"],
                "full_macro_f1": metrics["full_macro_f1"],
                "aurc": model_curve_summary["aurc"],
                "augrc": model_curve_summary["augrc"],
            }
            for class_index, class_name in enumerate(("A", "B", "T")):
                row.update({
                    f"{class_name}_accepted_precision": metrics["accepted_precision"][class_index],
                    f"{class_name}_accepted_recall": metrics["accepted_recall"][class_index],
                    f"{class_name}_accepted_f1": metrics["accepted_f1"][class_index],
                    f"{class_name}_coverage": metrics["class_coverage"][class_index],
                    f"{class_name}_automatic_recall": metrics["class_automatic_recall"][class_index],
                })
            comparison_rows.append(row)
            policy_rows.append({"model": name, "policy": policy_name, **parameters, "test_labels_used_for_fitting": False})
    pd.DataFrame(comparison_rows).to_csv(output / "selective_model_comparison.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(output / "validation_locked_policies.csv", index=False)
    pd.concat(curve_rows, ignore_index=True).to_csv(output / "risk_coverage_curves_all_models.csv", index=False)

    summary = {
        "status": "completed",
        "quick_mode": quick,
        "split_counts": {name: int(len(index)) for name, index in indices.items()},
        "w2_seed_temperatures": [fit.temperature for fit in w2_fits],
        "class_A_probability_adjustment": adjustment,
        "global_validation_threshold": global_threshold,
        "class_specific_policy": class_policy,
        "performance_constrained_policy": constrained,
        "test_metrics": test_metrics,
        "matched_count_test_metrics": matched_metrics,
        "risk_curve_summary": curve_summary,
        "baseline_selected_parameters": baseline_fit.selected_parameters,
        "model_temperatures": model_temperatures,
        "comparison_model_count": len(probability_sets),
        "matched_count_diagnostic": matched_count,
        "bootstrap_metric_count": int(len(bootstrap_results)),
        "random_referral_control_row_count": int(len(random_control_results)),
        "random_referral_control_endpoints": sorted(
            random_control_results["metric"].unique().tolist()
        ),
        "test_labels_used_for_fitting": False,
    }
    (output / "pipeline_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary
