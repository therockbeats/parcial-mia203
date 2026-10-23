"""Pérdidas enmascaradas para segmentación binaria: BCE, Dice y su combinación."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    """Entropía cruzada binaria promediada sobre los píxeles válidos (``w`` = 1)."""
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction="none")
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def masked_dice(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Pérdida Dice suave calculada sobre todos los píxeles válidos del lote."""
    p = torch.sigmoid(logits) * w
    t = y * w
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + eps) / (p.sum() + t.sum() + eps)


class SegLoss(nn.Module):
    """``bce``, ``dice`` o ``bce_dice`` (suma de ambas)."""

    def __init__(self, kind: str = "bce_dice", pos_weight: float = 1.0):
        super().__init__()
        if kind not in ("bce", "dice", "bce_dice"):
            raise ValueError(f"Pérdida desconocida: {kind}")
        self.kind, self.pos_weight = kind, pos_weight

    def forward(self, logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.kind == "bce":
            return masked_bce(logits, y, w, self.pos_weight)
        if self.kind == "dice":
            return masked_dice(logits, y, w)
        return masked_bce(logits, y, w, self.pos_weight) + masked_dice(logits, y, w)
