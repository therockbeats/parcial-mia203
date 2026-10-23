"""
Etapa 11. Análisis de fallos característicos del SAR.

Clasifica los píxeles mal predichos por el ensamble de la referencia según su retrodispersión y
su contexto:

    borde            error a menos de 2 píxeles del contorno de la referencia (píxeles mixtos,
                     desajuste geométrico entre imagen y etiqueta)
    fp_oscuro        falso positivo con VV y VH bajos: superficies lisas o secas que reflejan la
                     señal lejos del sensor (arena, asfalto, suelo desnudo seco) o sombra del radar
    fp_otro          resto de falsos positivos (suelo húmedo, vegetación baja, bordes difusos)
    fn_brillante     falso negativo con VV alto: doble rebote en zonas urbanas o vegetación
                     inundadas, o agua rugosa por viento
    fn_otro          resto de falsos negativos (agua somera o con retrodispersión intermedia)

En los chips dorados se describe cada categoría con índices de Sentinel-2 (NDVI, MNDWI) y con
el agua permanente de JRC; en Lima, con la cobertura del suelo y la pendiente. Para cada categoría
se generan ejemplos visuales con los chips de mayor número de píxeles de esa categoría.

Salidas:
    p2_inundaciones/reports/tablas/fallos_categorias.csv, fallos_descriptores.csv
    p2_inundaciones/reports/figuras/fallos_<categoria>.png

Uso:
    python -m p2_inundaciones.scripts.s11_fallos_sar [--ref GW_bcedice] [--splits test bolivia lima]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import ndimage

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p2_inundaciones.src.auxiliares import jrc, lima_aux, lulc_group, s1_db, s2_indices, slope_deg
from p2_inundaciones.src.ensemble import ensemble_split, ensemble_tables, labels_for, run_dirs

REF_CMAP = ListedColormap(["#bdbdbd", "#ffffff", "#2166ac"])
CATS = ["borde", "fp_oscuro", "fp_otro", "fn_brillante", "fn_otro"]
COLORS = {"borde": [1.0, 1.0, 0.0], "fp_oscuro": [1.0, 0.0, 0.0], "fp_otro": [1.0, 0.55, 0.0],
          "fn_brillante": [0.0, 0.9, 1.0], "fn_otro": [0.2, 0.3, 1.0]}
TITLES = {"borde": "Errores de borde", "fp_oscuro": "Falsos positivos en superficies oscuras",
          "fp_otro": "Otros falsos positivos", "fn_brillante": "Falsos negativos con retrodispersión alta",
          "fn_otro": "Otros falsos negativos"}


def categorize(db: np.ndarray, pred: np.ndarray, lab: np.ndarray, valid: np.ndarray, fc: dict) -> np.ndarray:
    """Categoría por píxel (índice en CATS, -1 si no es error)."""
    water = lab == 1
    edge = water ^ ndimage.binary_erosion(water, iterations=2)
    edge |= ndimage.binary_dilation(water, iterations=2) & ~water
    fp = pred & ~water & valid
    fn = ~pred & water & valid
    vv, vh = db[0], db[1]
    cat = np.full(lab.shape, -1, np.int8)
    cat[(fp | fn) & edge] = 0
    rest = (fp | fn) & ~edge
    dark = (vv < fc["dark_vv"]) & (vh < fc["dark_vh"])
    cat[rest & fp & dark] = 1
    cat[rest & fp & ~dark] = 2
    cat[rest & fn & (vv > fc["bright_vv"])] = 3
    cat[rest & fn & ~(vv > fc["bright_vv"])] = 4
    return cat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--ref", default=None)
    ap.add_argument("--splits", nargs="*", default=["test", "bolivia", "lima"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s11")
    fc = cfg["failures"]
    ref = args.ref or cfg["ablation"]["reference"]
    dirs = run_dirs(cfg, ref, [int(s) for s in cfg["ablation"]["seeds"]])
    if not dirs:
        raise FileNotFoundError(f"Sin corridas evaluadas de {ref}")
    thr = ensemble_tables(cfg, dirs, [])[0]
    tab_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")
    fig_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "figuras")

    counts, desc, examples = [], [], {c: [] for c in CATS}
    for split in args.splits:
        ids, probs = ensemble_split(dirs, split)
        if ids is None:
            continue
        lab, valid = labels_for(cfg, ids, split)
        tot = {c: 0 for c in CATS}
        acc = {c: {"ndvi": [], "mndwi": [], "jrc": [], "lulc": [], "pendiente": []} for c in CATS}
        for i, cid in enumerate(ids):
            db = s1_db(cfg, cid)
            pred = probs[i] >= thr
            cat = categorize(db, pred, lab[i], valid[i], fc)
            s2 = s2_indices(cfg, cid) if split != "lima" else None
            jw = jrc(cfg, cid) if split != "lima" else None
            lu = lima_aux(cfg, cid, "lulc") if split == "lima" else None
            dem = lima_aux(cfg, cid, "dem") if split == "lima" else None
            sl = slope_deg(dem) if dem is not None else None
            for k, c in enumerate(CATS):
                m = cat == k
                n = int(m.sum())
                tot[c] += n
                if n == 0:
                    continue
                examples[c].append((n, split, cid))
                if s2 is not None:
                    acc[c]["ndvi"].append(s2["ndvi"][m])
                    acc[c]["mndwi"].append(s2["mndwi"][m])
                if jw is not None:
                    acc[c]["jrc"].append(jw[m] == 1)
                if lu is not None:
                    acc[c]["lulc"].append(lulc_group(lu[m]))
                if sl is not None:
                    acc[c]["pendiente"].append(sl[m])
        errors = sum(tot.values())
        for c in CATS:
            counts.append({"particion": split, "categoria": c, "pixeles": tot[c], "pct_errores": 100 * tot[c] / max(errors, 1)})
            r = {"particion": split, "categoria": c}
            if acc[c]["ndvi"]:
                nd, mw = np.concatenate(acc[c]["ndvi"]), np.concatenate(acc[c]["mndwi"])
                r.update({"ndvi_mediana": float(np.nanmedian(nd)), "mndwi_mediana": float(np.nanmedian(mw)),
                          "pct_mndwi_pos": 100 * float(np.nanmean(mw > 0)), "pct_ndvi_mayor_03": 100 * float(np.nanmean(nd > 0.3))})
            if acc[c]["jrc"]:
                r["pct_agua_permanente"] = 100 * float(np.concatenate(acc[c]["jrc"]).mean())
            if acc[c]["lulc"]:
                g = pd.Series(np.concatenate(acc[c]["lulc"])).value_counts(normalize=True) * 100
                r.update({f"pct_{k}": v for k, v in g.items()})
            if acc[c]["pendiente"]:
                r["pendiente_mediana"] = float(np.nanmedian(np.concatenate(acc[c]["pendiente"])))
            desc.append(r)
    ct, ds = pd.DataFrame(counts), pd.DataFrame(desc)
    ct.to_csv(tab_dir / "fallos_categorias.csv", index=False)
    ds.to_csv(tab_dir / "fallos_descriptores.csv", index=False)
    log.info("\n" + ct.pivot(index="categoria", columns="particion", values="pct_errores").round(1).to_string())

    nex = int(fc.get("n_examples", 4))
    for k, c in enumerate(CATS):
        top = sorted(examples[c], reverse=True)[:nex]
        if not top:
            continue
        fig, axes = plt.subplots(len(top), 5, figsize=(13, 2.7 * len(top)), squeeze=False)
        for row, (n, split, cid) in enumerate(top):
            ids, probs = ensemble_split(dirs, split)
            i = ids.index(cid)
            lab, valid = labels_for(cfg, [cid], split)
            db = s1_db(cfg, cid)
            pred = probs[i] >= thr
            cat = categorize(db, pred, lab[0], valid[0], fc)
            s2 = s2_indices(cfg, cid) if split != "lima" else None
            axes[row, 0].imshow(db[0], cmap="gray", vmin=-25, vmax=0)
            axes[row, 0].set_title(f"{cid} ({split})\nVV", fontsize=8)
            axes[row, 1].imshow(db[1], cmap="gray", vmin=-32, vmax=-8)
            axes[row, 1].set_title("VH", fontsize=8)
            if s2 is not None:
                axes[row, 2].imshow(s2["rgb"])
                axes[row, 2].set_title("Sentinel-2 RGB", fontsize=8)
            else:
                dem = lima_aux(cfg, cid, "dem")
                if dem is not None:
                    axes[row, 2].imshow(dem, cmap="terrain")
                    axes[row, 2].set_title("Elevación SRTM", fontsize=8)
            show = np.where(valid[0], lab[0], -1)
            axes[row, 3].imshow(show, cmap=REF_CMAP, vmin=-1, vmax=1, interpolation="nearest")
            axes[row, 3].set_title("referencia (azul: agua; gris: sin etiqueta)", fontsize=8)
            ov = np.zeros(cat.shape + (4,))
            ov[pred & valid[0]] = [0.6, 0.6, 0.6, 0.5]
            for kk, cc in enumerate(CATS):
                ov[cat == kk] = COLORS[cc] + [0.95 if kk == k else 0.35]
            axes[row, 4].imshow(db[0], cmap="gray", vmin=-25, vmax=0)
            axes[row, 4].imshow(ov)
            axes[row, 4].set_title(f"predicción y errores ({n:,} px)", fontsize=8)
            for a in axes[row]:
                a.axis("off")
        fig.suptitle(TITLES[c], fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fallos_{c}.png", dpi=120)
        plt.close(fig)
    log.info(f"Guardado {tab_dir / 'fallos_categorias.csv'} y figuras fallos_*.png")


if __name__ == "__main__":
    main()
