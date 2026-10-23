"""
Etapa 10. Estudio de ablación.

Entrena (etapa 7) y evalúa (etapa 8) los experimentos de ``ablation.experiments``; cada uno
modifica un factor respecto de la referencia B (ResNet-50, BCE):
    A  codificador desde cero          D  ResNet-18
    C  pérdida focal                   E  hallazgos no mencionados excluidos de la pérdida
    F  sobremuestreo de hallazgos poco frecuentes

Salidas:
    p1_rodilla/reports/tablas/ablacion.csv
    p1_rodilla/reports/figuras/ablacion_auc.png

Uso:
    python -m p1_rodilla.scripts.s10_ablacion [--gpu 1]
    python -m p1_rodilla.scripts.s10_ablacion --only A_scratch_bce C_resnet50_focal
    python -m p1_rodilla.scripts.s10_ablacion --solo-tabla
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger


def run(cmd: list[str], env: dict) -> None:
    """Ejecuta un subproceso desde la raíz del repositorio."""
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--solo-tabla", action="store_true")
    ap.add_argument("--gpu", default=None, help="índice de la GPU (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s10")
    exps = cfg["ablation"]["experiments"]
    if args.only:
        exps = [e for e in exps if e["name"] in args.only]
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not args.solo_tabla:
        for e in exps:
            sets = sum([["--set", s] for s in e["set"]], [])
            extra = ["--smoke"] if args.smoke else []
            run([sys.executable, "-m", "p1_rodilla.scripts.s07_entrenar", "--exp", e["name"], "--config", args.config, *sets, *extra], env)
            run([sys.executable, "-m", "p1_rodilla.scripts.s08_evaluar", "--exp", e["name"], "--config", args.config], env)

    rows = []
    for e in cfg["ablation"]["experiments"]:
        p = Path(cfg["data"]["runs_dir"]) / e["name"] / "summary.json"
        if p.exists():
            s = json.loads(p.read_text())
            s["cambios"] = "; ".join(e["set"])
            mg = pd.read_csv(p.parent / "metrics_gold.csv").set_index("hallazgo")["auc"]
            s.update({f"auc_{k}": v for k, v in mg.items() if k != "MACRO"})
            rows.append(s)
    if not rows:
        log.warning("Sin resultados de experimentos.")
        return
    df = pd.DataFrame(rows)
    tab = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas") / "ablacion.csv"
    df.to_csv(tab, index=False)
    log.info("\n" + df[["exp", "backbone", "loss", "unmentioned_policy", "val_macro_auc", "gold_macro_auc", "gold_macro_f1"]]
             .round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    lo = [s - l for s, (l, h) in zip(df.gold_macro_auc, df.gold_macro_auc_ic95)]
    hi = [h - s for s, (l, h) in zip(df.gold_macro_auc, df.gold_macro_auc_ic95)]
    ax.barh(df.exp, df.gold_macro_auc, xerr=[lo, hi], color="#3b6ea5", capsize=4)
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("AUC-ROC macro en el conjunto dorado (IC 95 %)")
    ax.set_xlim(0.4, 1.0)
    fig.tight_layout()
    fig.savefig(ensure_dir(Path(cfg["data"]["reports_dir"]) / "figuras") / "ablacion_auc.png", dpi=150)
    log.info(f"Tabla: {tab}")


if __name__ == "__main__":
    main()
