"""Frozen neural architecture used by the final W2 MIL system."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.b1 = nn.Conv1d(in_channels, out_channels, 1)
        self.b3 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.b5 = nn.Conv1d(in_channels, out_channels, 5, padding=2)
        self.pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, 1),
        )
        self.bn = nn.BatchNorm1d(out_channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.cat(
            [self.b1(x), self.b3(x), self.b5(x), self.pool(x)], dim=1
        )
        return F.relu(self.bn(output), inplace=True)


class InceptionEncoder1D(nn.Module):
    def __init__(self, in_channels: int = 3, embedding_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.inc1 = InceptionBlock1D(64, 64)
        self.down1 = nn.MaxPool1d(2)
        self.inc2 = InceptionBlock1D(256, 64)
        self.down2 = nn.MaxPool1d(2)
        self.inc3 = InceptionBlock1D(256, 64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(self.inc1(x))
        x = self.down2(self.inc2(x))
        x = self.inc3(x)
        return self.proj(self.gap(x).squeeze(-1))


class GatedAttnPoolWeighted(nn.Module):
    """Gated pooling with a learned soft record-informativeness bias."""
    def __init__(
        self,
        embedding_dim: int = 256,
        use_quality: bool = True,
        use_learned_temperature: bool = True,
    ) -> None:
        super().__init__()
        self.V = nn.Linear(embedding_dim, embedding_dim)
        self.U = nn.Linear(embedding_dim, embedding_dim)
        self.w = nn.Linear(embedding_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.use_quality = bool(use_quality)
        self.use_learned_temperature = bool(use_learned_temperature)

    def forward(
        self, embedding: torch.Tensor, quality: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_logit = self.w(
            torch.tanh(self.V(embedding)) * torch.sigmoid(self.U(embedding))
        ).squeeze(-1)
        if self.use_quality:
            safe_quality = torch.clamp(quality, min=1e-6, max=1.0)
            attention_logit = attention_logit + self.alpha * torch.log(safe_quality)
        temperature = (
            torch.clamp(self.temp, 0.2, 5.0)
            if self.use_learned_temperature
            else 1.0
        )
        attention = torch.softmax(attention_logit / temperature, dim=1)
        representation = (attention.unsqueeze(-1) * embedding).sum(dim=1)
        return representation, attention


class MILClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        embedding_dim: int = 256,
        n_classes: int = 3,
        dropout: float = 0.25,
        use_quality_attention: bool = True,
        use_max_pooling: bool = False,
        use_learned_attention_temperature: bool = True,
        use_window_head: bool = True,
        use_bt_head: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = int(feature_dim)
        self.pool = GatedAttnPoolWeighted(
            embedding_dim=embedding_dim,
            use_quality=use_quality_attention,
            use_learned_temperature=use_learned_attention_temperature,
        )
        self.use_max_pooling = bool(use_max_pooling)
        self.win_head = (
            nn.Linear(embedding_dim, n_classes) if use_window_head else None
        )
        fusion_input = (
            2 * embedding_dim if self.use_max_pooling else embedding_dim
        ) + self.feature_dim
        self.fuse = nn.Sequential(
            nn.Linear(fusion_input, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(256, n_classes)
        self.bt_head = nn.Linear(256, 2) if use_bt_head else None

    def forward(
        self, waveform: torch.Tensor, feature: torch.Tensor, quality: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch, windows, components, samples = waveform.shape
        embedding = self.encoder(
            waveform.reshape(batch * windows, components, samples)
        ).reshape(batch, windows, -1)
        attention_representation, attention = self.pool(embedding, quality)
        if self.use_max_pooling:
            negative_infinity = torch.finfo(embedding.dtype).min
            masked = torch.where(
                quality.unsqueeze(-1) > 0.0, embedding, negative_infinity
            )
            maximum = masked.max(dim=1).values
            maximum = torch.where(
                torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
            )
            representation = torch.cat([attention_representation, maximum], dim=1)
        else:
            representation = attention_representation
        fusion_input = (
            representation
            if self.feature_dim == 0
            else torch.cat([representation, feature], dim=1)
        )
        fused = self.fuse(fusion_input)
        output = {
            "logits": self.cls_head(fused),
            "att": attention,
        }
        if self.bt_head is not None:
            output["bt_logits"] = self.bt_head(fused)
        if self.win_head is not None:
            output["win_logits"] = self.win_head(embedding)
        return output


def build_final_model() -> MILClassifier:
    """Construct the effective frozen inference architecture."""
    return MILClassifier(
        encoder=InceptionEncoder1D(in_channels=3, embedding_dim=256),
        feature_dim=0,
        embedding_dim=256,
        n_classes=3,
        dropout=0.25,
        use_quality_attention=True,
        use_max_pooling=False,
        use_learned_attention_temperature=True,
        use_window_head=False,
        use_bt_head=False,
    )


def adapt_legacy_checkpoint(
    state_dict: dict[str, torch.Tensor],
    model: MILClassifier,
) -> dict[str, torch.Tensor]:
    """Drop two historical all-zero feature columns without changing logits."""
    adapted = dict(state_dict)
    key = "fuse.0.weight"
    if key not in adapted:
        raise KeyError(f"Checkpoint is missing {key}")
    checkpoint_weight = adapted[key]
    expected_width = model.fuse[0].in_features
    if checkpoint_weight.shape[1] == expected_width:
        return adapted
    if checkpoint_weight.shape[1] == expected_width + 2:
        adapted[key] = checkpoint_weight[:, :expected_width].contiguous()
    elif checkpoint_weight.shape[1] != expected_width:
        raise ValueError(
            f"Unsupported {key} width {checkpoint_weight.shape[1]}; "
            f"expected {expected_width} or legacy width {expected_width + 2}"
        )
    model_keys = set(model.state_dict())
    for inactive_key in (
        "win_head.weight",
        "win_head.bias",
        "bt_head.weight",
        "bt_head.bias",
    ):
        if inactive_key not in model_keys:
            adapted.pop(inactive_key, None)
    return adapted
