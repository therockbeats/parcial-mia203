"""
Etapa 8. Evaluación de un experimento en el conjunto dorado.

Umbral por hallazgo elegido en validación (máximo F1); en el conjunto dorado se calculan
AUC-ROC por hallazgo y macro (con IC 95 % por bootstrap), precisión promedio, y precisión,
sensibilidad y F1 con esos umbrales.

Salidas en <runs_dir>/<exp>/:
    metrics_gold.csv, metrics_val.csv, preds_gold.csv, summary.json, roc_gold.png

Uso:
    python -m p1_rodilla.scripts.s08_evaluar --exp B_resnet50_bce
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from common.config import REPO_ROOT, load_config
from common.utils import get_device, get_logger
from p1_rodilla.src.engine import load_split, make_dataset, make_loader, predict
from p1_rodilla.src.metrics import best_f1_thresholds, bootstrap_macro_auc, evaluate
from p1_rodilla.src.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    # Configuración efectiva guardada por el experimento
    base = load_config(args.config)
    exp_dir = Path(base["data"]["runs_dir"]) / args.exp
    cfg = load_config(exp_dir / "config.yaml", args.set)
    log = get_logger("s08", exp_dir / "eval.log")
    names = cfg["labels"]["names"]
    device = get_device(cfg["train"].get("device", "auto"))

    model = build_model(cfg).to(device)
    ckpt = torch.load(exp_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    log.info(f"Pesos de la época {ckpt['epoch']} (AUC val {ckpt['val_macro_auc']:.4f})")

    split = load_split(cfg)
    cfg["train"]["num_workers"] = min(int(cfg["train"].get("num_workers", 4)), 4)
    va_df, go_df = split[split.split == "val"], split[split.split == "gold"]
    p_va, y_va, w_va = predict(model, make_loader(cfg, make_dataset(cfg, va_df, False), False), device)
    p_go, y_go, _ = predict(model, make_loader(cfg, make_dataset(cfg, go_df, False), False), device)

    # Umbrales en validación (etiquetas con peso > 0)
    y_va_thr = np.where(w_va > 0, y_va, 0)
    thr = best_f1_thresholds(y_va_thr, p_va)
    m_val = evaluate(y_va, p_va, names, thr, mask=w_va)
    m_gold = evaluate(y_go, p_go, names, thr)
    lo, hi = bootstrap_macro_auc(y_go, p_go, int(cfg["eval"].get("bootstrap", 1000)), cfg.get("seed", 0))

    m_val.to_csv(exp_dir / "metrics_val.csv", index=False)
    m_gold.to_csv(exp_dir / "metrics_gold.csv", index=False)
    pd.DataFrame(p_go, columns=names).assign(study_id=go_df.study_id.values).to_csv(exp_dir / "preds_gold.csv", index=False)
    macro = m_gold[m_gold.hallazgo == "MACRO"].iloc[0]
    summary = {"exp": args.exp, "backbone": cfg["model"]["backbone"], "loss": cfg["train"]["loss"],
               "labels_source": cfg["labels"]["source"], "unmentioned_policy": cfg["labels"]["unmentioned_policy"],
               "gold_macro_auc": float(macro["auc"]), "gold_macro_auc_ic95": [lo, hi],
               "gold_macro_f1": float(macro["f1"]), "val_macro_auc": float(m_val[m_val.hallazgo == "MACRO"]["auc"].iloc[0]),
               "n_gold": int(len(go_df))}
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("\nMétricas en el conjunto dorado:\n" + m_gold.round(3).to_string(index=False))
    log.info(f"AUC macro dorado = {macro['auc']:.4f}  (IC95 % bootstrap: {lo:.3f} a {hi:.3f})")

    # Curvas ROC por hallazgo
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for k, (n, ax) in enumerate(zip(names, axes.ravel())):
        if len(np.unique(y_go[:, k])) == 2:
            fpr, tpr, _ = roc_curve(y_go[:, k], p_go[:, k])
            ax.plot(fpr, tpr, lw=2)
            ax.set_title(f"{n}  AUC={m_gold.loc[k, 'auc']:.2f}", fontsize=10)
        else:
            ax.set_title(f"{n}  (sin positivos y negativos)", fontsize=9)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("1 - especificidad")
        ax.set_ylabel("sensibilidad")
    fig.suptitle(f"Curvas ROC en el conjunto dorado: {args.exp}")
    fig.tight_layout()
    fig.savefig(exp_dir / "roc_gold.png", dpi=150)
    log.info(f"Resultados en {exp_dir}")


if __name__ == "__main__":
    main()
