from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from ve_selective.grouping import exclude_mixed_label_groups, overlap_group_ids
from ve_selective.model import (
    InceptionEncoder1D,
    MILClassifier,
    adapt_legacy_checkpoint,
    build_final_model,
)
from ve_selective.selective import (
    apply_class_probability_multiplier,
    class_specific_acceptance,
    temperature_scale,
)
from ve_selective.inference import SelectiveEnsemble


def test_final_model_shape_and_parameter_count() -> None:
    model = build_final_model().eval()
    waveform = torch.zeros(2, 2, 3, 6000)
    feature = torch.empty(2, 0)
    quality = torch.ones(2, 2)
    with torch.inference_mode():
        output = model(waveform, feature, quality)
    assert output["logits"].shape == (2, 3)
    assert output["att"].shape == (2, 2)
    assert sum(parameter.numel() for parameter in model.parameters()) == 834_182
    assert "win_logits" not in output
    assert "bt_logits" not in output


def test_legacy_zero_feature_checkpoint_conversion_preserves_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    legacy = MILClassifier(
        encoder=InceptionEncoder1D(in_channels=3, embedding_dim=256),
        feature_dim=2,
        embedding_dim=256,
        n_classes=3,
        dropout=0.25,
        use_quality_attention=True,
        use_max_pooling=False,
        use_learned_attention_temperature=True,
    ).eval()
    state = torch.load(
        root / "models" / "w2_softq_seed17.pt",
        map_location="cpu",
        weights_only=True,
    )
    legacy.load_state_dict(state)
    effective = build_final_model().eval()
    effective.load_state_dict(adapt_legacy_checkpoint(state, effective))
    waveform = torch.zeros(1, 2, 3, 6000)
    quality = torch.ones(1, 2)
    with torch.inference_mode():
        legacy_output = legacy(waveform, torch.zeros(1, 2), quality)
        effective_output = effective(waveform, torch.empty(1, 0), quality)
    assert torch.equal(legacy_output["logits"], effective_output["logits"])
    assert torch.equal(legacy_output["att"], effective_output["att"])


def test_overlap_grouping_and_mixed_label_exclusion() -> None:
    group = overlap_group_ids(["r0", "r1", "r4"], ["r1", "r2", "r5"])
    assert group[0] == group[1]
    assert group[2] != group[0]
    frame = pd.DataFrame(
        {"group_id": group, "label": ["A", "T", "B"], "event": [0, 1, 2]}
    )
    retained, excluded = exclude_mixed_label_groups(frame)
    assert retained["event"].tolist() == [2]
    assert excluded["event"].tolist() == [0, 1]


def test_calibration_and_class_specific_decision() -> None:
    probability = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.65, 0.25]])
    calibrated = temperature_scale(probability, 1.0)
    prediction, confidence, accepted = class_specific_acceptance(
        calibrated, np.asarray([0.75, 0.70, 0.70])
    )
    assert prediction.tolist() == [0, 1]
    assert np.allclose(confidence, [0.8, 0.65])
    assert accepted.tolist() == [True, False]


def test_final_release_policy_parameters() -> None:
    root = Path(__file__).resolve().parents[1]
    import json
    config = json.loads((root / "configs" / "final_w2_temporal.json").read_text())
    assert config["feature_vector_dimension"] == 0
    assert config["legacy_checkpoint_zero_feature_columns"] == 2
    assert np.allclose(
        [config["temperatures"][str(seed)] for seed in [17, 42, 73]],
        [0.5952595303723589, 0.5098293538500619, 0.5186144919625266],
    )
    assert np.isclose(config["a_probability_multiplier"], 0.25464158411554416)
    assert np.isclose(config["chronological_test_realized_coverage"], 0.6933797909407665)
    assert np.allclose(
        [config["predicted_class_thresholds"][c] for c in ["A", "B", "T"]],
        [0.9519033419752547, 0.8790783923425981, 0.8972924214771505],
    )


def test_primary_probability_correction_precedes_class_threshold() -> None:
    probability = np.asarray(
        [
            [0.60, 0.30, 0.10],
            [0.98, 0.01, 0.01],
        ]
    )
    corrected = apply_class_probability_multiplier(
        probability,
        class_index=0,
        multiplier=0.25464158411554416,
    )
    assert np.allclose(corrected.sum(axis=1), 1.0)
    prediction, confidence, accepted = class_specific_acceptance(
        corrected,
        np.asarray([0.9519033419752547, 0.8790783923425981, 0.8972924214771505]),
    )
    assert prediction.tolist() == [1, 0]
    assert np.allclose(confidence, corrected.max(axis=1))
    assert accepted.tolist() == [False, False]


def test_predict_rejects_inputs_outside_preprocessed_tensor_contract() -> None:
    system = SelectiveEnsemble.__new__(SelectiveEnsemble)

    def assert_rejected(array, **kwargs) -> None:
        try:
            system.predict(array, **kwargs)
        except (TypeError, ValueError):
            return
        raise AssertionError("Input outside the preprocessed tensor contract was accepted")

    valid_shape = np.zeros((1, 2, 3, 6000), dtype=np.float32)
    contract = {
        "sampling_rate_hz": 100.0,
        "input_domain": "preprocessed_waveform_tensor",
    }
    assert_rejected(valid_shape)
    assert_rejected(valid_shape, input_domain="raw_waveform")
    assert_rejected(valid_shape, sampling_rate_hz=50.0, input_domain=contract["input_domain"])
    assert_rejected(valid_shape, batch_size=0, **contract)
    assert_rejected(np.zeros((1, 3, 6000), dtype=np.float32), **contract)
    assert_rejected(np.zeros((0, 2, 3, 6000), dtype=np.float32), **contract)
    nonfinite = valid_shape.copy()
    nonfinite[0, 0, 0, 0] = np.nan
    assert_rejected(nonfinite, **contract)
    assert_rejected(valid_shape.astype(np.complex64), **contract)


if __name__ == "__main__":
    test_final_model_shape_and_parameter_count()
    test_legacy_zero_feature_checkpoint_conversion_preserves_outputs()
    test_overlap_grouping_and_mixed_label_exclusion()
    test_calibration_and_class_specific_decision()
    test_final_release_policy_parameters()
    test_primary_probability_correction_precedes_class_threshold()
    test_predict_rejects_inputs_outside_preprocessed_tensor_contract()
    print("All direct smoke tests passed.")
