"""
Etapa 6. Desbalance de clases.

1. Prevalencia: cajas por especie en entrenamiento y prueba, y tasa de imágenes vacías.
2. AP por especie frente al número de cajas de entrenamiento (escala logarítmica) y correlación
   de Spearman, en cámaras vistas y no vistas.
3. Efecto de las imágenes vacías como fondo (``C_sin_vacias`` frente a la referencia: falsos
   positivos en vacías y mAP) y, si se ejecutó el extra ``D_con_rfs``, del Repeat Factor
   Sampling (diferencia de AP por especie).

Salidas en p3_fauna/reports/:
    figuras/prevalencia.png, figuras/ap_vs_frecuencia.png, tablas/prevalencia_por_particion.csv,
    tablas/ap_por_especie_vs_frecuencia.csv, tablas/efecto_vacias.csv, tablas/efecto_rfs_por_especie.csv

Uso:
    python -m p3_fauna.scripts.s06_desbalance --exp B_yolov8s
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p3_fauna.src.experimentos import best_experiment, exp_dir, read_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    ap.add_argument("--exp", default=None, help="por defecto, el mejor experimento de la ablación")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s06")
    exp = args.exp or best_experiment(cfg)[0] or cfg["ablation"]["reference"]
    yolo = Path(cfg["data"]["yolo_dir"])
    figs = ensure_dir(Path(cfg["data"]["reports_dir"]) / "figuras")
    tables = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")

    # 1. Prevalencia
    boxes = pd.read_csv(yolo / "boxes_meta.csv")
    imgs = pd.read_csv(yolo / "images_meta.csv")
    prev = boxes.groupby(["especie", "particion"]).size().unstack(fill_value=0)
    prev = prev.sort_values("train", ascending=False) if "train" in prev else prev
    pct = (prev / prev.sum() * 100).round(2).add_suffix("_pct")
    pd.concat([prev, pct], axis=1).to_csv(tables / "prevalencia_por_particion.csv")
    vac = imgs.assign(vacia=imgs.n_boxes == 0).groupby("particion").vacia.mean().mul(100).round(1)
    log.info("Imágenes vacías (%):\n" + vac.to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = [c for c in ["train", "cis_test", "trans_test"] if c in prev]
    prev[cols].plot.bar(ax=ax, logy=True, width=0.8)
    ax.set_ylabel("cajas (escala log)")
    ax.set_xlabel("")
    ax.set_title("Cajas por especie y partición")
    fig.tight_layout()
    fig.savefig(figs / "prevalencia.png", dpi=150)
    plt.close(fig)

    # 2. AP frente a frecuencia
    pe = exp_dir(cfg, exp) / "por_especie.csv"
    if pe.exists():
        sp = pd.read_csv(pe, index_col=0)
        out, fig = [], plt.figure(figsize=(6.5, 4.5))
        ax = fig.gca()
        for split, mk in [("cis_test", "o"), ("trans_test", "s")]:
            col = f"AP50_95_{split}"
            if col not in sp:
                continue
            d = sp.dropna(subset=[col])
            d = d[d.cajas_train > 0]
            ax.scatter(d.cajas_train, d[col], marker=mk, label=split)
            for name, r in d.iterrows():
                ax.annotate(name, (r.cajas_train, r[col]), fontsize=7, alpha=0.7)
            rho = pd.Series(np.log(d.cajas_train)).corr(d[col], method="spearman") if len(d) > 2 else np.nan
            out.append({"particion": split, "especies": len(d), "spearman_log_n_vs_AP50_95": rho})
        ax.set_xscale("log")
        ax.set_xlabel("cajas de entrenamiento (escala log)")
        ax.set_ylabel("AP@0.5:0.95")
        ax.set_title(f"AP por especie frente a su frecuencia ({exp})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs / "ap_vs_frecuencia.png", dpi=150)
        plt.close(fig)
        sp.to_csv(tables / "ap_por_especie_vs_frecuencia.csv")
        corr = pd.DataFrame(out)
        corr.to_csv(tables / "correlacion_ap_frecuencia.csv", index=False)
        log.info("\n" + corr.round(3).to_string(index=False))

    # 3. Efecto de las imágenes vacías y del Repeat Factor Sampling
    ref = cfg["ablation"]["reference"]
    rows = []
    for name in (ref, "C_sin_vacias"):
        s = read_summary(cfg, name)
        if s:
            rows.append({"experimento": name, **{k: s.get(k) for k in
                         ("map50_95_cis_test", "map50_95_trans_test", "fp_vacias_cis_test", "fp_vacias_trans_test",
                          "recall_presencia_trans_test")}})
    if len(rows) == 2:
        pd.DataFrame(rows).to_csv(tables / "efecto_vacias.csv", index=False)
        log.info("Efecto de las vacías:\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    a, b = exp_dir(cfg, ref) / "por_especie.csv", exp_dir(cfg, "D_con_rfs") / "por_especie.csv"
    if a.exists() and b.exists():
        ra, rb = pd.read_csv(a, index_col=0), pd.read_csv(b, index_col=0)
        cols = [c for c in ("AP50_95_cis_test", "AP50_95_trans_test") if c in ra and c in rb]
        eff = ra[["cajas_train"]].join((rb[cols] - ra[cols]).add_prefix("delta_rfs_")).sort_values("cajas_train")
        eff.to_csv(tables / "efecto_rfs_por_especie.csv")
        log.info("Ganancia de AP por RFS (con RFS - referencia), de menos a más frecuente:\n" + eff.round(3).to_string())


if __name__ == "__main__":
    main()
