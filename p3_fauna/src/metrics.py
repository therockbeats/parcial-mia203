"""Emparejamiento de cajas y métricas a nivel de imagen."""
from __future__ import annotations

import numpy as np
import pandas as pd


def iou_xyxy(a: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU entre una caja (x1, y1, x2, y2) y un conjunto de cajas."""
    if len(boxes) == 0:
        return np.zeros(0)
    ix1, iy1 = np.maximum(a[0], boxes[:, 0]), np.maximum(a[1], boxes[:, 1])
    ix2, iy2 = np.minimum(a[2], boxes[:, 2]), np.minimum(a[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(a) + area(boxes) - inter + 1e-9)


def box_hits(gt: np.ndarray, pred_xyxy: np.ndarray, pred_cls: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Para cada caja real (clase, x1, y1, x2, y2): ¿hay una predicción de la misma especie con IoU >= umbral?"""
    hits = np.zeros(len(gt), dtype=bool)
    for i, g in enumerate(gt):
        same = pred_cls == int(g[0])
        hits[i] = bool((iou_xyxy(g[1:], pred_xyxy[same]) >= iou_thr).any())
    return hits


def presence_scores(table: pd.DataFrame) -> dict:
    """
    Métricas de presencia por imagen (¿hay algún animal?) a partir de las columnas
    ``hay_animal`` (referencia) y ``detecta_animal`` (modelo con deploy_conf).
    """
    y, p = table.hay_animal.astype(bool), table.detecta_animal.astype(bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"imagenes": len(table), "vacias": int((~y).sum()),
            "precision_presencia": prec, "recall_presencia": rec,
            "f1_presencia": 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan"),
            "fp_en_vacias": fp / (fp + tn) if fp + tn else float("nan"),
            "especie_correcta": float(table.loc[y, "especie_correcta"].mean()) if y.any() else float("nan")}
