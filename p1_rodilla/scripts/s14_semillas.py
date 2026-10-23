"""
Etapa 14. Variabilidad por semilla y ensamble de semillas.

Para cada experimento de ``ablation.experiments`` reúne la corrida original (semilla del
config) y las corridas adicionales ``<exp>_s<k>``. Reporta, por configuración, la media y la
desviación estándar del AUC macro en el conjunto dorado y en validación, y el AUC macro del
ensamble (promedio de las probabilidades de todas las semillas) con su IC 95 % por bootstrap.
Los ensambles se comparan con el de la referencia mediante bootstrap pareado.

Con ``--entrenar`` se entrenan y evalúan primero (etapas 7 y 8) las semillas que falten de los
experimentos indicados en ``--only`` (todos si se omite). El resumen incluye siempre todos los
experimentos con corridas evaluadas.

Salidas:
    p1_rodilla/reports/tablas/semillas_corridas.csv
    p1_rodilla/reports/tablas/semillas_resumen.csv
    p1_rodilla/reports/figuras/semillas_auc.png

Uso:
    python -m p1_rodilla.scripts.s14_semillas
    python -m p1_rodilla.scripts.s14_semillas --entrenar --semillas 1 2 \\
        --only B_resnet50_bce D_resnet18_bce E_resnet50_ignorar_no_mencionados
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p1_rodilla.src.data_io import load_gold
from p1_rodilla.src.metrics import bootstrap_macro_auc, macro_auc


def run(cmd: list[str], env: dict) -> None:
    """Ejecuta un subproceso desde la raíz del repositorio."""
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def find_runs(runs: Path, name: str) -> list[Path]:
    """Corrida original y corridas ``<name>_s<k>`` con resultados de evaluación."""
    pat = re.compile(rf"^{re.escape(name)}(_s\d+)?$")
    dirs = [d for d in runs.iterdir() if d.is_dir() and pat.match(d.name)] if runs.exists() else []
    return sorted(d for d in dirs if (d / "summary.json").exists() and (d / "preds_gold.csv").exists())


def run_seed(d: Path) -> int | None:
    """Semilla registrada en la configuración efectiva de la corrida."""
    f = d / "config.yaml"
    if not f.exists():
        return None
    return yaml.safe_load(f.read_text(encoding="utf-8")).get("seed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--ref", default="B_resnet50_bce")
    ap.add_argument("--only", nargs="*", default=None, help="experimentos a entrenar con --entrenar")
    ap.add_argument("--entrenar", action="store_true", help="entrenar y evaluar las semillas que falten")
    ap.add_argument("--semillas", nargs="*", type=int, default=[1, 2], help="semillas adicionales")
    ap.add_argument("--gpu", default=None, help="índice de la GPU (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s14")
    names = cfg["labels"]["names"]
    runs = Path(cfg["data"]["runs_dir"])
    exps = cfg["ablation"]["experiments"]

    if args.entrenar:
        env = dict(os.environ)
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        for e in [e for e in exps if not args.only or e["name"] in args.only]:
            sets = sum([["--set", s] for s in e["set"]], [])
            for s in args.semillas:
                exp = f"{e['name']}_s{s}"
                if (runs / exp / "summary.json").exists():
                    log.info(f"{exp} ya evaluado; se omite")
                    continue
                run([sys.executable, "-m", "p1_rodilla.scripts.s07_entrenar", "--exp", exp, "--config", args.config,
                     *sets, "--set", f"seed={s}"], env)
                run([sys.executable, "-m", "p1_rodilla.scripts.s08_evaluar", "--exp", exp, "--config", args.config], env)

    gold = load_gold(cfg).set_index("study_id")
    y = gold[names].to_numpy()

    rows, ens = [], {}
    for e in exps:
        dirs = find_runs(runs, e["name"])
        if not dirs:
            continue
        probs = []
        for d in dirs:
            s = json.loads((d / "summary.json").read_text())
            p = pd.read_csv(d / "preds_gold.csv", dtype={"study_id": str}).set_index("study_id")
            probs.append(p.loc[gold.index, names].to_numpy())
            rows.append({"configuracion": e["name"], "corrida": d.name, "semilla": run_seed(d),
                         "auc_macro_val": s["val_macro_auc"], "auc_macro_dorado": s["gold_macro_auc"]})
        ens[e["name"]] = np.mean(probs, axis=0)
    if not rows:
        raise FileNotFoundError("No hay corridas evaluadas (ejecutar s10 o s14 --entrenar)")
    per_run = pd.DataFrame(rows)

    ref = args.ref if args.ref in ens else next(iter(ens))
    rng = np.random.default_rng(cfg.get("seed", 42))
    deltas = {k: [] for k in ens}
    for _ in range(args.n):
        idx = rng.integers(0, len(y), len(y))
        a_ref = macro_auc(y[idx], ens[ref][idx])
        for k, p in ens.items():
            deltas[k].append(macro_auc(y[idx], p[idx]) - a_ref)

    summary = []
    for k, g in per_run.groupby("configuracion", sort=False):
        lo, hi = bootstrap_macro_auc(y, ens[k], args.n, cfg.get("seed", 42))
        d = np.asarray(deltas[k])
        d = d[~np.isnan(d)]
        summary.append({
            "configuracion": k,
            "n_semillas": len(g),
            "auc_dorado_media": g.auc_macro_dorado.mean(),
            "auc_dorado_de": g.auc_macro_dorado.std(ddof=1) if len(g) > 1 else np.nan,
            "auc_dorado_min": g.auc_macro_dorado.min(),
            "auc_dorado_max": g.auc_macro_dorado.max(),
            "auc_val_media": g.auc_macro_val.mean(),
            "auc_val_de": g.auc_macro_val.std(ddof=1) if len(g) > 1 else np.nan,
            "auc_dorado_ensamble": macro_auc(y, ens[k]),
            "ensamble_ic95_inf": lo,
            "ensamble_ic95_sup": hi,
            "delta_ensamble_vs_ref": macro_auc(y, ens[k]) - macro_auc(y, ens[ref]),
            "delta_ic95_inf": np.percentile(d, 2.5),
            "delta_ic95_sup": np.percentile(d, 97.5),
            "p_bilateral": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if k != ref else np.nan,
        })
    summary = pd.DataFrame(summary)

    tab_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")
    fig_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "figuras")
    per_run.to_csv(tab_dir / "semillas_corridas.csv", index=False)
    summary.to_csv(tab_dir / "semillas_resumen.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(summary) + 1.5))
    for i, r in summary.iterrows():
        v = per_run.loc[per_run.configuracion == r.configuracion, "auc_macro_dorado"]
        ax.scatter(v, [i] * len(v), color="tab:blue", alpha=0.7, label="semilla" if i == 0 else None)
        ax.scatter(r.auc_dorado_ensamble, i, marker="D", color="tab:red", label="ensamble" if i == 0 else None)
        ax.plot([r.ensamble_ic95_inf, r.ensamble_ic95_sup], [i, i], color="tab:red", lw=1)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary.configuracion)
    ax.invert_yaxis()
    ax.axvline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("AUC macro en el conjunto dorado")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "semillas_auc.png", dpi=150)

    log.info("\nCorridas:\n" + per_run.round(4).to_string(index=False))
    cols = ["configuracion", "n_semillas", "auc_dorado_media", "auc_dorado_de", "auc_val_media",
            "auc_dorado_ensamble", "ensamble_ic95_inf", "ensamble_ic95_sup",
            "delta_ensamble_vs_ref", "delta_ic95_inf", "delta_ic95_sup", "p_bilateral"]
    log.info(f"\nResumen por configuración (ensambles comparados con {ref}):\n"
             + summary[cols].round(4).to_string(index=False))
    log.info(f"Guardado {tab_dir / 'semillas_resumen.csv'}")


if __name__ == "__main__":
    main()
