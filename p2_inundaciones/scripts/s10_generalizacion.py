"""
Etapa 10. Generalización geográfica: Sen1Floods11 frente a Bolivia y Lima.

1. Para cada configuración (ensamble de semillas) compara el IoU, F1, precisión y sensibilidad en
   la prueba en distribución, en Bolivia y en Lima, y calcula la brecha.
2. Con el ensamble de la referencia, cuantifica en Lima las hipótesis sobre la brecha:
   a) sensibilidad según el origen del polígono de inundación (cauce o área inundada);
   b) falsos positivos y sensibilidad por cobertura del suelo (urbano, arena y roca, etc.);
   c) falsos positivos según la pendiente del terreno y el ángulo de incidencia local (sombra y
      escorzo del radar), y efecto de enmascarar el relieve como en los servicios operativos;
   d) distribución de la retrodispersión del agua y de la no agua en Sen1Floods11 y en Lima;
   e) ángulo de incidencia de la escena de Lima;
   f) cambio de retrodispersión entre la escena previa y la posterior dentro de los polígonos;
   g) línea base de Otsu sobre VH en Lima.

Salidas:
    p2_inundaciones/reports/tablas/generalizacion.csv, lima_hipotesis_*.csv
    p2_inundaciones/reports/figuras/lima_mapa_errores.png, retrodispersion_s1f_vs_lima.png

Uso:
    python -m p2_inundaciones.scripts.s10_generalizacion [--ref GW_bcedice]
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
from p2_inundaciones.src.auxiliares import local_incidence_deg, lima_aux, lulc_group, s1_db, slope_deg
from p2_inundaciones.src.ensemble import ensemble_split, ensemble_tables, labels_for, run_dirs
from p2_inundaciones.src.io_s1f import load_chips, otsu_threshold
from p2_inundaciones.src.metrics import bootstrap_iou, chip_table_scores, confusion, scores


def rate_table(groups: np.ndarray, pred: np.ndarray, lab: np.ndarray, valid: np.ndarray, name: str) -> pd.DataFrame:
    rows = []
    for gname in pd.unique(groups[valid]):
        m = valid & (groups == gname)
        c = confusion(pred[m], lab[m], np.ones(m.sum(), bool))
        rows.append({name: gname, "pixeles": int(m.sum()), "agua_ref_pct": 100 * (c["tp"] + c["fn"]) / max(m.sum(), 1),
                     "sensibilidad": c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else np.nan,
                     "tasa_fp": c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else np.nan,
                     "fp_pct_del_total": c["fp"]})
    df = pd.DataFrame(rows)
    if len(df):
        df["fp_pct_del_total"] = 100 * df.fp_pct_del_total / max(df.fp_pct_del_total.sum(), 1)
    return df.sort_values("pixeles", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s10")
    ab = cfg["ablation"]
    ref = args.ref or ab["reference"]
    seeds = [int(s) for s in ab["seeds"]]
    tab_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")
    fig_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "figuras")
    n = int(cfg["eval"]["bootstrap"])

    rows = []
    for e in ab["experiments"]:
        dirs = run_dirs(cfg, e["name"], seeds)
        if not dirs:
            continue
        thr, tabs = ensemble_tables(cfg, dirs, ["test", "bolivia", "lima"])
        r = {"configuracion": e["name"], "umbral": thr}
        for split, t in tabs.items():
            s = chip_table_scores(t)
            lo, hi = bootstrap_iou(t.tp.values, t.fp.values, t.fn.values, n, 0)
            r.update({f"{split}_iou": s["iou"], f"{split}_ic95": f"{lo:.3f}-{hi:.3f}", f"{split}_f1": s["f1"],
                      f"{split}_precision": s["precision"], f"{split}_recall": s["recall"]})
        for split in ("bolivia", "lima"):
            if f"{split}_iou" in r and "test_iou" in r:
                r[f"brecha_{split}"] = r["test_iou"] - r[f"{split}_iou"]
        rows.append(r)
    gen = pd.DataFrame(rows)
    gen.to_csv(tab_dir / "generalizacion.csv", index=False)
    if len(gen):
        log.info("\n" + gen[[c for c in gen.columns if c.endswith(("_iou", "brecha_bolivia", "brecha_lima")) or c == "configuracion"]].round(3).to_string(index=False))

    dirs = run_dirs(cfg, ref, seeds)
    ids, probs = ensemble_split(dirs, "lima") if dirs else (None, None)
    if ids is None:
        log.warning("Sin predicciones de Lima de la referencia; se omiten las hipótesis")
        return
    thr = ensemble_tables(cfg, dirs, [])[0]
    lab, valid = labels_for(cfg, ids, "lima")
    pred = probs >= thr
    chips = load_chips(cfg).set_index("chip_id")

    fuente = np.stack([lima_aux(cfg, c, "fuente") if lima_aux(cfg, c, "fuente") is not None else np.zeros(lab.shape[1:]) for c in ids])
    codes = pd.read_csv(tab_dir / "lima_codigos.csv") if (tab_dir / "lima_codigos.csv").exists() else pd.DataFrame()
    names = {0: "fuera de polígono"}
    for r in codes.itertuples():
        if str(r.codigo).isdigit():
            names[int(r.codigo)] = str(r.origen)
    lookup = np.array([names.get(i, str(i)) for i in range(int(fuente.max()) + 1)], dtype=object)
    src = lookup[fuente.astype(int)]
    t_src = rate_table(src, pred, lab, valid, "origen")
    t_src.to_csv(tab_dir / "lima_hipotesis_origen.csv", index=False)

    lulc = [lima_aux(cfg, c, "lulc") for c in ids]
    if all(x is not None for x in lulc):
        t_lulc = rate_table(lulc_group(np.stack(lulc)), pred, lab, valid, "cobertura")
        t_lulc.to_csv(tab_dir / "lima_hipotesis_cobertura.csv", index=False)
        log.info("\nPor cobertura del suelo:\n" + t_lulc.round(4).to_string(index=False))
    dem = [lima_aux(cfg, c, "dem") for c in ids]
    ang = [lima_aux(cfg, c, "angle") for c in ids]
    sl = loc = None
    if all(x is not None for x in dem):
        px_y = float(cfg["lima"]["pixel_deg"]) * 111320.0
        px_x = px_y * np.cos(np.radians(np.mean(cfg["lima"]["bbox"][1::2])))
        sl = np.stack([slope_deg(d, px_y, px_x) for d in dem])
        cls = np.where(sl < 5, "0-5", np.where(sl < 15, "5-15", np.where(sl < 30, "15-30", ">30")))
        t_sl = rate_table(cls.astype(object), pred, lab, valid, "pendiente_grados")
        t_sl.to_csv(tab_dir / "lima_hipotesis_pendiente.csv", index=False)
        log.info("\nPor pendiente:\n" + t_sl.round(4).to_string(index=False))
        if all(a is not None for a in ang):
            az = float(cfg["lima"]["sensor_azimuth_deg"])
            loc = np.stack([local_incidence_deg(d, a, az, px_y, px_x) for d, a in zip(dem, ang)])
            bins = [(-1, 30, "<30 (orientada al radar)"), (30, 40, "30-40"), (40, 50, "40-50 (casi plano)"),
                    (50, 60, "50-60"), (60, 90, "60-90"), (90, 181, ">=90 (de espaldas al radar)")]
            cls = np.full(loc.shape, "", dtype=object)
            for lo_, hi_, nm in bins:
                cls[(loc >= lo_) & (loc < hi_)] = nm
            t_loc = rate_table(cls, pred, lab, valid, "incidencia_local_grados")
            t_loc["orden"] = t_loc.incidencia_local_grados.map({nm: i for i, (_, _, nm) in enumerate(bins)})
            t_loc = t_loc.sort_values("orden").drop(columns="orden")
            t_loc.to_csv(tab_dir / "lima_hipotesis_incidencia_local.csv", index=False)
            log.info("\nPor ángulo de incidencia local:\n" + t_loc.round(4).to_string(index=False))

    db = np.stack([s1_db(cfg, c) for c in ids])
    rng = np.random.default_rng(0)
    test_ids, _ = ensemble_split(dirs, "test")
    tl, tv = labels_for(cfg, test_ids, "test")
    tdb = np.stack([s1_db(cfg, c) for c in test_ids])
    q = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for b, bname in enumerate(["VV", "VH"]):
        for name, arr, l, v, ls in (("Sen1Floods11", tdb, tl, tv, "-"), ("Lima", db, lab, valid, "--")):
            for cls_v, cls_n, col in ((1, "agua", "#2166ac"), (0, "no agua", "#b2182b")):
                vals = arr[:, b][v & (l == cls_v)]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                vals = rng.choice(vals, min(vals.size, 200000), replace=False)
                q.append({"banda": bname, "conjunto": name, "clase": cls_n, "pixeles": int((v & (l == cls_v)).sum()),
                          "p10": np.percentile(vals, 10), "mediana": np.median(vals), "p90": np.percentile(vals, 90)})
                axes[b].hist(vals, bins=120, range=(-35, 5), density=True, histtype="step", color=col, ls=ls,
                             label=f"{name}, {cls_n}")
        axes[b].set_xlabel(f"{bname} (dB)")
        axes[b].set_ylabel("densidad")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "retrodispersion_s1f_vs_lima.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(q).to_csv(tab_dir / "lima_hipotesis_retrodispersion.csv", index=False)

    extra = {}
    if all(a is not None for a in ang):
        a = np.stack(ang)[valid]
        a = a[np.isfinite(a)]
        extra.update({"angulo_incidencia_min": float(a.min()), "angulo_incidencia_media": float(a.mean()),
                      "angulo_incidencia_max": float(a.max())})
    pre = [lima_aux(cfg, c, "pre") for c in ids]
    if all(p is not None for p in pre):
        d = db - np.stack(pre)
        for code, nm in names.items():
            m = valid & (fuente == code)
            if m.any():
                for b, bname in enumerate(["VV", "VH"]):
                    extra[f"cambio_{bname}_dB_{nm}"] = float(np.nanmedian(d[:, b][m]))
    vh_valid = db[:, 1][valid]
    thr_otsu = otsu_threshold(vh_valid[np.isfinite(vh_valid)])
    otsu_pred = db[:, 1] < thr_otsu
    c = confusion(otsu_pred, lab, valid & np.isfinite(db[:, 1]))
    extra.update({"otsu_vh_umbral_db": thr_otsu, **{f"otsu_{k}": v for k, v in scores(**c).items()}})
    for code, nm in names.items():
        m = valid & (fuente == code) & (lab == 1)
        if m.any():
            extra[f"frac_bajo_otsu_{nm}"] = float((db[:, 1][m] < thr_otsu).mean())
    if sl is not None:
        masks = {"sin máscara": np.zeros_like(valid), "pendiente > 15°": sl > 15}
        if loc is not None:
            masks["pendiente > 15° o incidencia local > 60°"] = (sl > 15) | (loc > 60)
        mrows = []
        for mname, m in masks.items():
            for metodo, pr in (("U-Net", pred), ("Otsu VH", otsu_pred)):
                cm = confusion(pr & ~m, lab, valid & np.isfinite(db[:, 1]))
                mrows.append({"mascara": mname, "metodo": metodo, "area_excluida_pct": 100 * float(m[valid].mean()),
                              "agua_ref_excluida_pct": 100 * float(m[valid & (lab == 1)].mean()),
                              **scores(**cm)})
        t_m = pd.DataFrame(mrows)
        t_m.to_csv(tab_dir / "lima_mascara_terreno.csv", index=False)
        log.info("\nCon máscara de relieve:\n" + t_m.round(4).to_string(index=False))
    c = confusion(pred, lab, valid)
    extra.update({"referencia": ref, "umbral": thr, **{f"modelo_{k}": v for k, v in scores(**c).items()}})
    pd.DataFrame([extra]).T.rename(columns={0: "valor"}).to_csv(tab_dir / "lima_hipotesis_resumen.csv")
    log.info("\n" + t_src.round(4).to_string(index=False))
    log.info("\n" + pd.Series(extra).to_string())

    rows_, cols_ = chips.loc[ids, "row"].astype(int).values, chips.loc[ids, "col"].astype(int).values
    step = 4
    k = lab.shape[1] // step
    nr, nc = rows_.max() + 1, cols_.max() + 1
    bg = np.full((nr * k, nc * k), np.nan, np.float32)
    ov = np.zeros((nr * k, nc * k, 4))
    for i, (r, cc) in enumerate(zip(rows_, cols_)):
        sl_ = (slice(r * k, (r + 1) * k), slice(cc * k, (cc + 1) * k))
        bg[sl_] = db[i, 0, ::step, ::step]
        p, l, v = pred[i, ::step, ::step], lab[i, ::step, ::step], valid[i, ::step, ::step]
        o = np.zeros((k, k, 4))
        o[v & p & (l == 1)] = [0.1, 0.8, 0.1, 0.9]
        o[v & p & (l == 0)] = [1.0, 0.1, 0.1, 0.9]
        o[v & ~p & (l == 1)] = [0.1, 0.4, 1.0, 0.9]
        o[~v] = [1, 1, 1, 0.25]
        ov[sl_] = o
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(bg, cmap="gray", vmin=-25, vmax=0)
    ax.imshow(ov)
    ax.set_title(f"Lima, {cfg['lima']['scenes']['post']}: verdaderos positivos (verde), falsos positivos (rojo), "
                 f"falsos negativos (azul)", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "lima_mapa_errores.png", dpi=150)
    plt.close(fig)
    log.info(f"Guardado {tab_dir / 'generalizacion.csv'}")


if __name__ == "__main__":
    main()
