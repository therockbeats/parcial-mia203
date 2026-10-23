"""Métricas de la clase agua: matriz de confusión, IoU, F1 y comparaciones por bootstrap."""
from __future__ import annotations

import numpy as np


def confusion(pred: np.ndarray, label: np.ndarray, valid: np.ndarray) -> dict:
    """Conteos TP, FP, FN y TN sobre los píxeles válidos."""
    p, t, v = pred.astype(bool), label == 1, valid.astype(bool)
    return {"tp": int((p & t & v).sum()), "fp": int((p & ~t & v).sum()),
            "fn": int((~p & t & v).sum()), "tn": int((~p & ~t & v).sum())}


def scores(tp: float, fp: float, fn: float, tn: float = 0.0) -> dict:
    """IoU, F1, precisión, sensibilidad, omisión, comisión y exactitud de la clase agua."""
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    tot = tp + fp + fn + tn
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec,
            "omision": 1 - rec if rec == rec else float("nan"),
            "comision": 1 - prec if prec == prec else float("nan"),
            "exactitud": (tp + tn) / tot if tot > 0 else float("nan")}


def chip_table_scores(df) -> dict:
    """Métricas agregadas (píxeles de todos los chips) y IoU medio por chip."""
    s = scores(df.tp.sum(), df.fp.sum(), df.fn.sum(), df.tn.sum())
    union = df.tp + df.fp + df.fn
    s["iou_medio_chip"] = float((df.tp[union > 0] / union[union > 0]).mean()) if (union > 0).any() else float("nan")
    return s


def bootstrap_iou(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """IC 95 % del IoU agregado remuestreando chips."""
    rng = np.random.default_rng(seed)
    k = len(tp)
    vals = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        d = tp[i].sum() + fp[i].sum() + fn[i].sum()
        if d > 0:
            vals.append(tp[i].sum() / d)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def paired_bootstrap_iou(a: dict, b: dict, n: int = 2000, seed: int = 0) -> tuple[float, float, float, float]:
    """
    Diferencia de IoU agregado (a - b) sobre los mismos chips.

    ``a`` y ``b`` contienen arreglos ``tp``, ``fp`` y ``fn`` por chip, en el mismo orden.
    Devuelve la diferencia observada, su IC 95 % y el valor p bilateral.
    """
    def iou(d, i):
        den = d["tp"][i].sum() + d["fp"][i].sum() + d["fn"][i].sum()
        return d["tp"][i].sum() / den if den > 0 else np.nan

    k = len(a["tp"])
    full = np.arange(k)
    obs = iou(a, full) - iou(b, full)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        d.append(iou(a, i) - iou(b, i))
    d = np.asarray(d)
    d = d[~np.isnan(d)]
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(p)


def best_threshold(probs: np.ndarray, labels: np.ndarray, valid: np.ndarray,
                   grid: np.ndarray | None = None) -> float:
    """Umbral que maximiza el IoU agregado."""
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    v = valid.astype(bool)
    p, t = probs[v], labels[v] == 1
    best, thr = -1.0, 0.5
    for g in grid:
        q = p >= g
        tp, fp, fn = (q & t).sum(), (q & ~t).sum(), (~q & t).sum()
        iou = tp / max(tp + fp + fn, 1)
        if iou > best:
            best, thr = iou, float(g)
    return thr
