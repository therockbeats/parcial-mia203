"""
Pérdidas multietiqueta ponderadas por elemento.

WeightedBCE    entropía cruzada binaria con ``pos_weight`` por hallazgo.
WeightedFocal  pérdida focal (Lin et al., 2017): (1 - p_t)^gamma * BCE.

En ambas, ``w`` (estudio x hallazgo) pondera cada término; ``w = 0`` lo excluye.
La pérdida se normaliza por la suma de pesos.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCE(nn.Module):
    """BCE con pesos por elemento y ``pos_weight`` opcional."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor([]))

    def forward(self, logits, y, w):
        pw = self.pos_weight if self.pos_weight.numel() else None
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction="none")
        return (loss * w).sum() / w.sum().clamp(min=1.0)


class WeightedFocal(nn.Module):
    """Pérdida focal con pesos por elemento y ``pos_weight`` opcional."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor([]))

    def forward(self, logits, y, w):
        pw = self.pos_weight if self.pos_weight.numel() else None
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * y + (1 - p) * (1 - y)
        loss = (1 - p_t) ** self.gamma * bce
        return (loss * w).sum() / w.sum().clamp(min=1.0)


def compute_pos_weight(y, w, max_value: float = 10.0) -> torch.Tensor:
    """``pos_weight_k`` = negativos válidos / positivos válidos, acotado a [1, max_value]."""
    y, w = torch.as_tensor(y), torch.as_tensor(w)
    valid = w > 0
    pos = ((y == 1) & valid).sum(0).float()
    neg = ((y == 0) & valid).sum(0).float()
    return (neg / pos.clamp(min=1.0)).clamp(1.0, max_value)


def build_loss(cfg: dict, y_train, w_train) -> nn.Module:
    """Pérdida según ``train.loss`` y ``train.pos_weight``."""
    t = cfg["train"]
    pw = compute_pos_weight(y_train, w_train, t.get("pos_weight_max", 10.0)) if t.get("pos_weight") == "auto" else None
    if t.get("loss", "bce") == "focal":
        return WeightedFocal(float(t.get("focal_gamma", 2.0)), pw)
    return WeightedBCE(pw)
