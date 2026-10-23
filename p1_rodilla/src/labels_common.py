"""Conversión de estados de etiqueta en objetivos y pesos, y acuerdo con las etiquetas doradas."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels_rules import NEG, NM, POS, UNC


def to_targets(codes: pd.DataFrame, names: list[str], lab_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Objetivos ``y`` y pesos ``w`` a partir de los estados (POS, NEG, UNC, NM).

    Los pesos se toman de la sección ``labels`` del config; ``w = 0`` excluye la etiqueta.
    """
    c = codes[names].to_numpy()
    y = np.zeros_like(c, dtype=np.float32)
    w = np.zeros_like(c, dtype=np.float32)
    y[c == POS], w[c == POS] = 1.0, lab_cfg.get("weight_positive", 1.0)
    y[c == NEG], w[c == NEG] = 0.0, lab_cfg.get("weight_negative", 1.0)
    y[c == UNC], w[c == UNC] = float(lab_cfg.get("uncertain_as", 1)), lab_cfg.get("weight_uncertain", 0.0)
    if lab_cfg.get("unmentioned_policy", "negative") == "negative":
        y[c == NM], w[c == NM] = 0.0, lab_cfg.get("weight_unmentioned", 0.5)
    else:
        y[c == NM], w[c == NM] = 0.0, 0.0
    return y, w


def agreement_report(weak: pd.DataFrame, gold: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """
    Acuerdo entre etiquetas débiles y doradas por hallazgo.

    La etiqueta débil se binariza como positiva si es POS o UNC. Se reportan acuerdo,
    kappa de Cohen, sensibilidad, especificidad y conteos de estados no mencionados e
    inciertos, más una fila PROMEDIO. El ruido estimado es 1 - acuerdo.
    """
    from sklearn.metrics import cohen_kappa_score

    m = gold.merge(weak, on="study_id", suffixes=("_gold", "_weak"))
    rows = []
    for n in names:
        g = m[f"{n}_gold"].astype(int).to_numpy()
        code = m[f"{n}_weak"].to_numpy()
        p = np.isin(code, [POS, UNC]).astype(int)
        tp, tn = int(((p == 1) & (g == 1)).sum()), int(((p == 0) & (g == 0)).sum())
        fp, fn = int(((p == 1) & (g == 0)).sum()), int(((p == 0) & (g == 1)).sum())
        kappa = cohen_kappa_score(g, p) if len(set(g)) > 1 or len(set(p)) > 1 else float("nan")
        rows.append({
            "hallazgo": n, "n": len(m), "positivos_dorados": int(g.sum()),
            "acuerdo": (tp + tn) / max(len(m), 1),
            "kappa": kappa,
            "sensibilidad": tp / max(tp + fn, 1) if (tp + fn) else float("nan"),
            "especificidad": tn / max(tn + fp, 1) if (tn + fp) else float("nan"),
            "no_mencionado": int((code == NM).sum()),
            "incierto": int((code == UNC).sum()),
        })
    df = pd.DataFrame(rows)
    macro = df[["acuerdo", "kappa", "sensibilidad", "especificidad"]].mean(numeric_only=True)
    df.loc[len(df)] = {"hallazgo": "PROMEDIO", "n": len(m), "positivos_dorados": int(df["positivos_dorados"].sum()),
                       "acuerdo": macro["acuerdo"], "kappa": macro["kappa"], "sensibilidad": macro["sensibilidad"],
                       "especificidad": macro["especificidad"], "no_mencionado": int(df["no_mencionado"].sum()),
                       "incierto": int(df["incierto"].sum())}
    return df
