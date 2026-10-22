"""
Funciones de pérdida multietiqueta con pesos por elemento.

BCE (entropía cruzada binaria), una por hallazgo:
    L = -[ y log p + (1-y) log(1-p) ],  con p = sigmoide(logit)
  * pos_weight multiplica el término de los positivos: compensa hallazgos raros
    (si un hallazgo tiene 1 positivo por cada 20 negativos, pos_weight ~ 20).

Pérdida focal (Lin et al., 2017):
    FL = -(1 - p_t)^gamma * log(p_t)
  * Reduce el peso de los ejemplos fáciles (p_t cercano a 1) y concentra el
    aprendizaje en los difíciles. Útil con desbalance fuerte (fractura, quiste de Baker).

En ambas, w (peso por estudio y hallazgo) permite ignorar etiquetas inciertas (w=0)
o confiar menos en las "no mencionadas" (w=0.5).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor([]))

    def forward(self, logits, y, w):
        pw = self.pos_weight if self.pos_weight.numel() else None
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction="none")
        return (loss * w).sum() / w.sum().clamp(min=1.0)


class WeightedFocal(nn.Module):
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
    """pos_weight_k = (# negativos válidos) / (# positivos válidos), recortado a max_value."""
    y, w = torch.as_tensor(y), torch.as_tensor(w)
    valid = w > 0
    pos = ((y == 1) & valid).sum(0).float()
    neg = ((y == 0) & valid).sum(0).float()
    return (neg / pos.clamp(min=1.0)).clamp(1.0, max_value)


def build_loss(cfg: dict, y_train, w_train) -> nn.Module:
    t = cfg["train"]
    pw = compute_pos_weight(y_train, w_train, t.get("pos_weight_max", 10.0)) if t.get("pos_weight") == "auto" else None
    if t.get("loss", "bce") == "focal":
        return WeightedFocal(float(t.get("focal_gamma", 2.0)), pw)
    return WeightedBCE(pw)
