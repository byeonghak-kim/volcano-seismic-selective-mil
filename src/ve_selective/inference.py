"""Frozen ensemble inference for already-preprocessed waveform tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import adapt_legacy_checkpoint, build_final_model
from .quality import event_quality
from .selective import (
    apply_class_probability_multiplier,
    class_specific_acceptance,
    temperature_scale,
)


class SelectiveEnsemble:
    def __init__(
        self,
        models: list[torch.nn.Module],
        temperatures: list[float],
        mean: np.ndarray,
        standard_deviation: np.ndarray,
        log_prior: np.ndarray,
        thresholds: np.ndarray,
        classes: list[str],
        device: torch.device,
        quality_floor: float,
        a_probability_multiplier: float = 1.0,
    ) -> None:
        self.models = models
        self.temperatures = temperatures
        self.mean = np.asarray(mean, dtype=np.float32)[None, None, :, None]
        self.standard_deviation = np.asarray(
            standard_deviation, dtype=np.float32
        )[None, None, :, None]
        self.log_prior = np.asarray(log_prior, dtype=np.float32)
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.classes = np.asarray(classes)
        self.device = device
        self.quality_floor = float(quality_floor)
        self.a_probability_multiplier = float(a_probability_multiplier)

    @classmethod
    def from_release(
        cls, release_root: Path, device: str = "cpu"
    ) -> "SelectiveEnsemble":
        root = Path(release_root).resolve()
        config = json.loads(
            (root / "configs" / "final_w2_temporal.json").read_text(encoding="utf-8")
        )
        torch_device = torch.device(device)
        models = []
        temperatures = []
        for seed in config["seeds"]:
            model = build_final_model().to(torch_device)
            checkpoint = torch.load(
                root / config["checkpoints"][str(seed)],
                map_location=torch_device,
                weights_only=True,
            )
            checkpoint = adapt_legacy_checkpoint(checkpoint, model)
            model.load_state_dict(checkpoint)
            model.eval()
            models.append(model)
            temperatures.append(float(config["temperatures"][str(seed)]))
        thresholds = [
            config["predicted_class_thresholds"][class_name]
            for class_name in config["classes"]
        ]
        return cls(
            models=models,
            temperatures=temperatures,
            mean=config["normalization_mean"],
            standard_deviation=config["normalization_sd"],
            log_prior=config["log_prior"],
            thresholds=np.asarray(thresholds),
            classes=config["classes"],
            device=torch_device,
            quality_floor=config["quality_floor"],
            a_probability_multiplier=config.get("a_probability_multiplier", 1.0),
        )

    @torch.inference_mode()
    def predict(
        self,
        preprocessed_waveforms: np.ndarray,
        batch_size: int = 32,
        *,
        sampling_rate_hz: float = 100.0,
        input_domain: str | None = None,
    ) -> pd.DataFrame:
        """Classify preprocessed `(N, 2, 3, 6000)` tensors sampled at 100 Hz.

        The caller must apply the archive preprocessing recipe documented in
        the release README before calling this method. This method computes the
        frozen training-set normalization, model ensemble, and accept/refer
        policy; it does not demean, filter, resample, crop, or pad raw records.

        Component order is east, north, and vertical (E, N, U/Z). Set
        ``input_domain`` explicitly when integrating this API to guard against
        accidentally passing raw waveform arrays.
        """
        if input_domain != "preprocessed_waveform_tensor":
            raise ValueError(
                "predict accepts only input_domain='preprocessed_waveform_tensor'; "
                "raw waveform preprocessing is not implemented by this method"
            )
        try:
            sampling_rate = float(sampling_rate_hz)
        except (TypeError, ValueError) as error:
            raise ValueError("sampling_rate_hz must be 100.0") from error
        if not np.isfinite(sampling_rate) or not np.isclose(
            sampling_rate, 100.0, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                f"Expected preprocessed tensors at 100 Hz, received {sampling_rate_hz}"
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise ValueError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        source = np.asarray(preprocessed_waveforms)
        if not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source):
            raise TypeError("preprocessed_waveforms must contain real numeric values")
        array = np.asarray(source, dtype=np.float32)
        if array.ndim != 4 or array.shape[1:] != (2, 3, 6000):
            raise ValueError(
                "Expected preprocessed waveform tensors shaped "
                f"(N, 2, 3, 6000), received {array.shape}"
            )
        if len(array) == 0:
            raise ValueError("At least one preprocessed event is required")
        if not np.all(np.isfinite(array)):
            raise ValueError("preprocessed_waveforms must contain only finite values")
        quality = event_quality(array, floor=self.quality_floor)
        normalized = (array - self.mean) / (self.standard_deviation + 1e-8)
        probabilities = []
        attention_batches = []
        for model, temperature in zip(self.models, self.temperatures):
            model_probability = []
            model_attention = []
            for start in range(0, len(array), batch_size):
                end = min(start + batch_size, len(array))
                waveform = torch.from_numpy(normalized[start:end]).to(self.device)
                feature = torch.empty((end - start, 0), device=self.device)
                q = torch.from_numpy(quality[start:end]).to(self.device)
                output = model(waveform, feature, q)
                logits = output["logits"] + torch.as_tensor(
                    self.log_prior, dtype=output["logits"].dtype, device=self.device
                )[None, :]
                raw = torch.softmax(logits, dim=1).cpu().numpy()
                model_probability.append(temperature_scale(raw, temperature))
                model_attention.append(output["att"].cpu().numpy())
            probabilities.append(np.concatenate(model_probability))
            attention_batches.append(np.concatenate(model_attention))
        ensemble = np.mean(probabilities, axis=0)
        ensemble /= ensemble.sum(axis=1, keepdims=True)
        if self.a_probability_multiplier != 1.0:
            a_index = int(np.flatnonzero(self.classes == "A")[0])
            ensemble = apply_class_probability_multiplier(
                ensemble,
                class_index=a_index,
                multiplier=self.a_probability_multiplier,
            )
        prediction, confidence, accepted = class_specific_acceptance(
            ensemble, self.thresholds
        )
        attention = np.mean(attention_batches, axis=0)
        result = pd.DataFrame(
            {
                "prediction": self.classes[prediction],
                "confidence": confidence,
                "decision": np.where(accepted, "accept", "refer"),
                "quality_record_1": quality[:, 0],
                "quality_record_2": quality[:, 1],
                "attention_record_1": attention[:, 0],
                "attention_record_2": attention[:, 1],
            }
        )
        for class_index, class_name in enumerate(self.classes):
            result[f"p_{class_name}"] = ensemble[:, class_index]
        return result
