"""Code-faithful W2 MIL and 120-s CNN training recipes."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .calibration import fit_temperature, softmax_logits
from .model import build_final_model


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 35
    batch_size: int = 16
    learning_rate: float = 6e-4
    weight_decay: float = 1e-2
    focal_gamma: float = 2.0
    label_smoothing: float = 0.02
    device: str = "cpu"
    max_batches_per_epoch: int | None = None
    patience: int = 8


@dataclass
class NeuralFit:
    model: nn.Module
    logits: np.ndarray
    probability: np.ndarray
    temperature: float
    normalization_mean: np.ndarray
    normalization_sd: np.ndarray
    log_prior: np.ndarray | None
    history: pd.DataFrame
    best_epoch: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def component_normalization(waveforms: np.ndarray, train_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(waveforms)[np.asarray(train_index)]
    mean = train.mean(axis=(0, 1, 3)).astype(np.float32)
    standard_deviation = (train.std(axis=(0, 1, 3)) + 1e-8).astype(np.float32)
    return mean, standard_deviation


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
    label_smoothing: float = 0.02,
) -> torch.Tensor:
    classes = logits.shape[1]
    log_probability = F.log_softmax(logits, dim=1)
    probability = log_probability.exp()
    with torch.no_grad():
        label = torch.zeros_like(logits).scatter_(1, target[:, None], 1.0)
        label = label * (1.0 - label_smoothing) + label_smoothing / classes
    pt = (probability * label).sum(dim=1)
    loss = -((1.0 - pt) ** gamma) * (log_probability * label).sum(dim=1)
    if alpha is not None:
        loss = loss * alpha[target]
    return loss.mean()


def effective_number_weights(counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    count = np.asarray(counts, dtype=np.float64)
    weight = (1.0 - beta) / np.maximum(1.0 - np.power(beta, count), 1e-12)
    return (weight / weight.mean()).astype(np.float32)


class _W2Dataset(Dataset):
    def __init__(self, waveform, quality, labels, mean, standard_deviation):
        self.waveform = np.asarray(waveform, dtype=np.float32)
        self.quality = np.asarray(quality, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)[None, :, None]
        self.standard_deviation = np.asarray(standard_deviation, dtype=np.float32)[None, :, None]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        waveform = (self.waveform[index] - self.mean) / self.standard_deviation
        return torch.from_numpy(waveform), torch.from_numpy(self.quality[index]), torch.tensor(self.labels[index], dtype=torch.long)


class _CNN120Dataset(Dataset):
    def __init__(self, waveform, labels, mean, standard_deviation):
        self.waveform = np.asarray(waveform, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None]
        self.standard_deviation = np.asarray(standard_deviation, dtype=np.float32)[:, None]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        event = self.waveform[index]
        concatenated = np.concatenate([event[0], event[1]], axis=1)
        concatenated = (concatenated - self.mean) / self.standard_deviation
        return torch.from_numpy(concatenated), torch.tensor(self.labels[index], dtype=torch.long)


def _validation_score(labels: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    prediction = logits.argmax(axis=1)
    return (
        float(f1_score(labels, prediction, average="macro", zero_division=0)),
        float(accuracy_score(labels, prediction)),
    )


def _cnn_validation_score(labels: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    prediction = logits.argmax(axis=1)
    return (
        float(f1_score(labels, prediction, average="macro", zero_division=0)),
        float(accuracy_score(labels, prediction)),
    )


@torch.inference_mode()
def _infer_w2(model, loader, device, log_prior):
    model.eval()
    logits, labels = [], []
    for waveform, quality, target in loader:
        waveform = waveform.to(device)
        quality = quality.to(device)
        output = model(waveform, torch.empty((len(target), 0), device=device), quality)["logits"]
        output = output + torch.as_tensor(log_prior, dtype=output.dtype, device=device)[None, :]
        logits.append(output.cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def fit_w2_model(
    waveforms: np.ndarray,
    quality: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    config: TrainingConfig = TrainingConfig(),
    output_dir: Path | None = None,
) -> NeuralFit:
    """Fit the final W2 recipe and select a checkpoint by validation macro-F1."""
    set_seed(config.seed)
    device = torch.device(config.device)
    y = np.asarray(labels, dtype=np.int64)
    mean, standard_deviation = component_normalization(waveforms, train_index)
    counts = np.bincount(y[train_index], minlength=3).astype(np.float64)
    prior = counts / counts.sum()
    log_prior = np.log(prior + 1e-12).astype(np.float32)
    datasets = {
        "train": _W2Dataset(waveforms[train_index], quality[train_index], y[train_index], mean, standard_deviation),
        "validation": _W2Dataset(waveforms[validation_index], quality[validation_index], y[validation_index], mean, standard_deviation),
        "all": _W2Dataset(waveforms, quality, y, mean, standard_deviation),
    }
    sampler = WeightedRandomSampler(
        torch.ones(len(datasets["train"]), dtype=torch.double),
        num_samples=len(datasets["train"]),
        replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, sampler=sampler)
    validation_loader = DataLoader(datasets["validation"], batch_size=config.batch_size, shuffle=False)
    all_loader = DataLoader(datasets["all"], batch_size=config.batch_size, shuffle=False)
    model = build_final_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )
    best = None
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_index, (waveform, q, target) in enumerate(train_loader):
            if config.max_batches_per_epoch is not None and batch_index >= config.max_batches_per_epoch:
                break
            waveform, q, target = waveform.to(device), q.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveform, torch.empty((len(target), 0), device=device), q)["logits"]
            adjusted = logits + torch.as_tensor(log_prior, device=device)[None, :]
            loss = focal_loss(adjusted, target, gamma=config.focal_gamma, label_smoothing=config.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            seen += len(target)
        scheduler.step()
        validation_logits, validation_labels = _infer_w2(model, validation_loader, device, log_prior)
        score = _validation_score(validation_labels, validation_logits)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(seen, 1), "validation_macro_f1": score[0], "validation_accuracy": score[1]})
        if best is None or score > best[0]:
            best = (score, epoch, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    assert best is not None
    model.load_state_dict(best[2])
    logits, _ = _infer_w2(model, all_loader, device, log_prior)
    temperature = fit_temperature(y[validation_index], logits[validation_index])
    probability = softmax_logits(logits, temperature)
    result = NeuralFit(model, logits, probability, temperature, mean, standard_deviation, log_prior, pd.DataFrame(history), best[1])
    if output_dir is not None:
        _save_fit(result, config, Path(output_dir), "W2 MIL")
    return result


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))

    def forward(self, values):
        identity = self.skip(values)
        values = F.relu(self.bn1(self.conv1(values)), inplace=True)
        return F.relu(self.bn2(self.conv2(values)) + identity, inplace=True)


class ResNet1D120(nn.Module):
    """Direct 120-s concatenation comparator used for both CNN recipes."""
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(3, 24, 31, stride=8, padding=15, bias=False), nn.BatchNorm1d(24), nn.ReLU(inplace=True), nn.MaxPool1d(3, stride=2, padding=1))
        self.stage1 = BasicBlock1D(24, 24)
        self.stage2 = BasicBlock1D(24, 48, 2)
        self.stage3 = BasicBlock1D(48, 96, 2)
        self.stage4 = BasicBlock1D(96, 192, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(192, n_classes))

    def forward(self, values):
        values = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(values)))))
        return self.head(self.pool(values).squeeze(-1))


@torch.inference_mode()
def _infer_cnn(model, loader, device, log_prior):
    model.eval()
    logits, labels = [], []
    for waveform, target in loader:
        output = model(waveform.to(device))
        if log_prior is not None:
            output = output + torch.as_tensor(log_prior, device=device)[None, :]
        logits.append(output.cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def fit_cnn120(
    waveforms: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    recipe: str,
    config: TrainingConfig = TrainingConfig(batch_size=128, learning_rate=1e-3),
    output_dir: Path | None = None,
) -> NeuralFit:
    """Fit either `standard` or `imbalance_adjusted` 120-s CNN training."""
    if recipe not in {"standard", "imbalance_adjusted"}:
        raise ValueError("recipe must be 'standard' or 'imbalance_adjusted'")
    set_seed(config.seed)
    device = torch.device(config.device)
    y = np.asarray(labels, dtype=np.int64)
    mean, standard_deviation = component_normalization(waveforms, train_index)
    datasets = {
        "train": _CNN120Dataset(waveforms[train_index], y[train_index], mean, standard_deviation),
        "validation": _CNN120Dataset(waveforms[validation_index], y[validation_index], mean, standard_deviation),
        "all": _CNN120Dataset(waveforms, y, mean, standard_deviation),
    }
    counts = np.bincount(y[train_index], minlength=3)
    weights = torch.as_tensor(effective_number_weights(counts), device=device)
    prior = counts / counts.sum()
    log_prior = np.log(prior + 1e-12).astype(np.float32) if recipe == "imbalance_adjusted" else None
    if recipe == "imbalance_adjusted":
        sample_weight = 1.0 / np.maximum(counts, 1)
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weight[y[train_index]], dtype=torch.double), len(train_index), replacement=True, generator=torch.Generator().manual_seed(config.seed))
        train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed))
    validation_loader = DataLoader(datasets["validation"], batch_size=config.batch_size, shuffle=False)
    all_loader = DataLoader(datasets["all"], batch_size=config.batch_size, shuffle=False)
    model = ResNet1D120().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
        eta_min=(0.0 if recipe == "imbalance_adjusted" else config.learning_rate * 0.02),
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
    best = None
    history = []
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_index, (waveform, target) in enumerate(train_loader):
            if config.max_batches_per_epoch is not None and batch_index >= config.max_batches_per_epoch:
                break
            waveform, target = waveform.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveform)
            if recipe == "imbalance_adjusted":
                loss = focal_loss(logits + torch.as_tensor(log_prior, device=device)[None, :], target, gamma=2.0, alpha=weights, label_smoothing=config.label_smoothing)
            else:
                loss = criterion(logits, target)
            loss.backward()
            if recipe == "imbalance_adjusted":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            seen += len(target)
        scheduler.step()
        validation_logits, validation_labels = _infer_cnn(model, validation_loader, device, log_prior)
        score = _cnn_validation_score(validation_labels, validation_logits)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(seen, 1), "validation_macro_f1": score[0], "validation_accuracy": score[1]})
        if best is None or score > best[0]:
            best = (score, epoch, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
            stale_epochs = 0
        else:
            stale_epochs += 1
            if recipe == "standard" and stale_epochs >= config.patience:
                break
    assert best is not None
    model.load_state_dict(best[2])
    logits, _ = _infer_cnn(model, all_loader, device, log_prior)
    temperature = fit_temperature(y[validation_index], logits[validation_index])
    probability = softmax_logits(logits, temperature)
    result = NeuralFit(model, logits, probability, temperature, mean, standard_deviation, log_prior, pd.DataFrame(history), best[1])
    if output_dir is not None:
        _save_fit(result, config, Path(output_dir), f"120-s concatenation CNN ({recipe.replace('_', '-')} training)")
    return result


def _save_fit(fit: NeuralFit, config: TrainingConfig, output: Path, model_name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save(fit.model.state_dict(), output / "model_state.pt")
    np.save(output / "logits.npy", fit.logits, allow_pickle=False)
    np.save(output / "probability.npy", fit.probability, allow_pickle=False)
    fit.history.to_csv(output / "training_history.csv", index=False)
    metadata = {"model": model_name, "training_config": asdict(config), "best_epoch": fit.best_epoch, "temperature": fit.temperature, "normalization_mean": fit.normalization_mean.tolist(), "normalization_sd": fit.normalization_sd.tolist(), "log_prior": None if fit.log_prior is None else fit.log_prior.tolist(), "checkpoint_selection": "validation macro-F1, then validation accuracy", "calibration": "validation NLL temperature fitting"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
