"""
Métricas de clasificación multietiqueta.

AUC-ROC por hallazgo y macro (promedio simple sobre hallazgos, métrica oficial), precisión
promedio, y precisión, sensibilidad y F1 a un umbral por hallazgo. Los umbrales se eligen en
validación maximizando F1 y se aplican sin cambios al conjunto de evaluación.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


# Calcula el AUC-ROC únicamente cuando existen
# ejemplos positivos y negativos en la muestra.

def safe_auc(y, p) -> float:
    """AUC-ROC; NaN si solo hay una clase."""
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def best_f1_thresholds(y: np.ndarray, p: np.ndarray, grid=None) -> np.ndarray:
    """Umbral por hallazgo que maximiza F1 en una grilla (0.5 si no hay positivos)."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    thr = np.full(y.shape[1], 0.5)
    for k in range(y.shape[1]):
        if y[:, k].sum() == 0:
            continue
        f1s = [f1_score(y[:, k], (p[:, k] >= t).astype(int), zero_division=0) for t in grid]
        thr[k] = grid[int(np.argmax(f1s))]
    return thr


def evaluate(y: np.ndarray, p: np.ndarray, names: list[str], thresholds: np.ndarray | None = None,
             mask: np.ndarray | None = None) -> pd.DataFrame:
    """AUC, AP, precisión, sensibilidad y F1 por hallazgo, con una fila MACRO."""
    thresholds = np.full(len(names), 0.5) if thresholds is None else thresholds
    rows = []
    for k, n in enumerate(names):
        sel = np.ones(len(y), bool) if mask is None else mask[:, k] > 0
        yk, pk = y[sel, k].astype(int), p[sel, k]
        if sel.sum() == 0:
            rows.append({"hallazgo": n, "n": 0, "positivos": 0, "auc": float("nan"), "ap": float("nan"),
                         "umbral": float(thresholds[k]), "precision": float("nan"), "recall": float("nan"),
                         "f1": float("nan")})
            continue
        pred = (pk >= thresholds[k]).astype(int)
        rows.append({
            "hallazgo": n, "n": int(sel.sum()), "positivos": int(yk.sum()),
            "auc": safe_auc(yk, pk),
            "ap": float(average_precision_score(yk, pk)) if yk.sum() > 0 else float("nan"),
            "umbral": float(thresholds[k]),
            "precision": float(precision_score(yk, pred, zero_division=0)),
            "recall": float(recall_score(yk, pred, zero_division=0)),
            "f1": float(f1_score(yk, pred, zero_division=0)),
        })
    df = pd.DataFrame(rows)
    macro = df[["auc", "ap", "precision", "recall", "f1"]].mean(skipna=True)
    df.loc[len(df)] = {"hallazgo": "MACRO", "n": int(len(y)), "positivos": int(df["positivos"].sum()),
                       "auc": macro["auc"], "ap": macro["ap"], "umbral": float("nan"),
                       "precision": macro["precision"], "recall": macro["recall"], "f1": macro["f1"]}
    return df


def macro_auc(y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Promedio de las AUC por hallazgo, ignorando las no definidas."""
    aucs = []
    for k in range(y.shape[1]):
        sel = np.ones(len(y), bool) if mask is None else mask[:, k] > 0
        aucs.append(safe_auc(y[sel, k], p[sel, k]))
    return float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float("nan")


def bootstrap_macro_auc(y: np.ndarray, p: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """IC 95 % del AUC macro por bootstrap de estudios."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v = macro_auc(y[idx], p[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
