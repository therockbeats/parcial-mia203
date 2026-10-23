"""
Etapa 9. Estudio de ablación con varias semillas.

Entrena y evalúa (etapas 7 y 8) cada experimento de ``ablation.experiments`` con cada semilla
de ``ablation.seeds``; las corridas ya evaluadas se omiten. Los experimentos con
``train.init_from`` se inicializan con la corrida de la misma semilla del experimento indicado.

Para cada configuración reporta la media y la desviación estándar del IoU entre semillas y el
IoU del ensamble de semillas (promedio de probabilidades, umbral elegido en validación) con su
IC 95 %, y compara cada ensamble con el de su referencia (``ref`` del experimento o, si no lo
indica, ``ablation.reference``) mediante bootstrap pareado de chips.

Salidas:
    p2_inundaciones/reports/tablas/ablacion_corridas.csv, ablacion_resumen.csv
    p2_inundaciones/reports/figuras/ablacion_iou.png

Uso:
    python -m p2_inundaciones.scripts.s09_ablacion [--only G_bcedice GW_bcedice] [--seeds 42] [--solo-tabla]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p2_inundaciones.src.ensemble import ensemble_tables, run_dirs, run_metrics, run_name
from p2_inundaciones.src.metrics import bootstrap_iou, chip_table_scores, paired_bootstrap_iou


def run(cmd: list[str], env: dict) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def train_and_eval(cfg: dict, args, exps: list[dict], seeds: list[int], log) -> None:
    base = int(cfg.get("seed", 42))
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    runs = Path(cfg["data"]["runs_dir"])
    for seed in seeds:
        for e in exps:
            name = run_name(e["name"], seed, base)
            if (runs / name / "summary.json").exists():
                log.info(f"{name} ya evaluado; se omite")
                continue
            sets = []
            for s in e["set"]:
                if s.startswith("train.init_from="):
                    s = f"train.init_from={run_name(s.split('=', 1)[1], seed, base)}"
                sets += ["--set", s]
            sets += ["--set", f"seed={seed}"] + sum([["--set", s] for s in args.extra], [])
            try:
                run([sys.executable, "-m", "p2_inundaciones.scripts.s07_entrenar", "--exp", name, "--config", args.config, *sets], env)
                run([sys.executable, "-m", "p2_inundaciones.scripts.s08_evaluar", "--exp", name, "--config", args.config], env)
            except subprocess.CalledProcessError as err:
                log.error(f"{name} falló (código {err.returncode}); se continúa")


def summarize(cfg: dict, exps: list[dict], seeds: list[int], reference: str, splits: list[str], prefix: str, log) -> pd.DataFrame:
    tab_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")
    fig_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "figuras")
    per_run, ens = [], {}
    for e in exps:
        dirs = run_dirs(cfg, e["name"], seeds)
        if not dirs:
            continue
        for d in dirs:
            per_run.append({"configuracion": e["name"], **run_metrics(d)})
        ens[e["name"]] = ensemble_tables(cfg, dirs, splits)
    if not per_run:
        log.warning("Sin corridas evaluadas.")
        return pd.DataFrame()
    pr = pd.DataFrame(per_run)
    pr.to_csv(tab_dir / f"{prefix}_corridas.csv", index=False)
    n = int(cfg["eval"]["bootstrap"])
    rows = []
    refs = {e["name"]: e.get("ref", reference) for e in exps}
    for name, (thr, tabs) in ens.items():
        g = pr[pr.configuracion == name]
        ref_name = refs.get(name, reference)
        r = {"configuracion": name, "referencia": ref_name if ref_name != name else "", "semillas": len(g),
             "umbral_ensamble": thr}
        for split in splits:
            col = f"iou_{split}"
            if col in g:
                r[f"{split}_media"] = g[col].mean()
                r[f"{split}_de"] = g[col].std(ddof=1) if len(g) > 1 else np.nan
            if split in tabs:
                t = tabs[split]
                s = chip_table_scores(t)
                lo, hi = bootstrap_iou(t.tp.values, t.fp.values, t.fn.values, n, 0)
                r.update({f"{split}_iou_ens": s["iou"], f"{split}_ic95_inf": lo, f"{split}_ic95_sup": hi,
                          f"{split}_f1_ens": s["f1"], f"{split}_precision_ens": s["precision"],
                          f"{split}_recall_ens": s["recall"]})
                if ref_name in ens and name != ref_name and split in ens[ref_name][1]:
                    ref = ens[ref_name][1][split]
                    a = t.set_index("chip_id").loc[ref.chip_id]
                    d, dl, dh, p = paired_bootstrap_iou({k: a[k].values for k in ("tp", "fp", "fn")},
                                                        {k: ref[k].values for k in ("tp", "fp", "fn")}, n, 0)
                    r.update({f"{split}_delta_ref": d, f"{split}_delta_inf": dl, f"{split}_delta_sup": dh, f"{split}_p": p})
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(tab_dir / f"{prefix}_resumen.csv", index=False)

    show = [s for s in splits if f"{s}_iou_ens" in summ]
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 0.45 * len(summ) + 1.6), sharey=True, squeeze=False)
    for ax, split in zip(axes[0], show):
        for i, r in summ.iterrows():
            v = pr.loc[pr.configuracion == r.configuracion, f"iou_{split}"] if f"iou_{split}" in pr else []
            ax.scatter(v, [i] * len(v), color="#4d4d4d", s=14, alpha=0.7)
            ax.plot([r[f"{split}_ic95_inf"], r[f"{split}_ic95_sup"]], [i, i], color="#b2182b", lw=1)
            ax.scatter(r[f"{split}_iou_ens"], i, marker="D", color="#b2182b", s=26)
        ax.set_title({"val": "validación", "test": "prueba", "bolivia": "Bolivia", "lima": "Lima"}.get(split, split))
        ax.set_xlabel("IoU agua")
        ax.grid(alpha=0.3)
    axes[0][0].set_yticks(range(len(summ)), summ.configuracion)
    axes[0][0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_iou.png", dpi=150)
    plt.close(fig)
    cols = ["configuracion", "referencia", "semillas"] + [c for c in summ.columns if c.endswith(("_iou_ens", "_media", "_de", "_delta_ref", "_p"))]
    log.info("\n" + summ[cols].round(4).to_string(index=False))
    log.info(f"Guardado {tab_dir / f'{prefix}_resumen.csv'}")
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--solo-tabla", action="store_true")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="overrides añadidos a todas las corridas")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s09")
    ab = cfg["ablation"]
    seeds = args.seeds or [int(s) for s in ab["seeds"]]
    exps = [e for e in ab["experiments"] if not args.only or e["name"] in args.only]
    if not args.solo_tabla:
        train_and_eval(cfg, args, exps, seeds, log)
    summarize(cfg, ab["experiments"], [int(s) for s in ab["seeds"]], ab["reference"],
              ["val"] + list(cfg["eval"]["splits"]), "ablacion", log)


if __name__ == "__main__":
    main()
