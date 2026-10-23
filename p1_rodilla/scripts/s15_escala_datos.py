"""
Etapa 15. Escala de datos con etiquetas débiles.

Evalúa el efecto del número de estudios de entrenamiento (con etiquetas débiles) sobre el
AUC macro en el conjunto dorado. Los subconjuntos de entrenamiento son anidados: cada uno
contiene al anterior, y el más pequeño es el entrenamiento de ``split.csv``. La validación y
el conjunto dorado son los de ``split.csv`` en todos los tamaños.

1. Construye las particiones ``<interim_dir>/escala/split_n<N>.csv`` (``scale.train_sizes``).
2. Preprocesa los volúmenes que falten (etapa 6).
3. Entrena y evalúa (etapas 7 y 8) cada combinación de ``scale.experiments``, tamaño y
   ``scale.seeds``; las corridas ya evaluadas se omiten y un fallo no detiene el resto. El
   orden es por semilla, de modo que una ejecución interrumpida conserve curvas completas.
4. Resume por configuración y tamaño: media y DE entre semillas, ensamble de semillas con
   IC 95 %, y comparaciones pareadas del ensamble frente al tamaño base y frente a la
   referencia (``--ref``) con el mismo tamaño.

Las corridas del tamaño base son las de la ablación (``<exp>`` y ``<exp>_s<k>``); las demás se
nombran ``<exp>_n<N>`` y ``<exp>_n<N>_s<k>``.

Salidas:
    <interim_dir>/escala/split_n<N>.csv
    <runs_dir>/escala.log
    p1_rodilla/reports/tablas/escala_corridas.csv
    p1_rodilla/reports/tablas/escala_resumen.csv
    p1_rodilla/reports/tablas/escala_por_hallazgo.csv
    p1_rodilla/reports/figuras/escala_auc.png

Uso:
    python -m p1_rodilla.scripts.s15_escala_datos
    python -m p1_rodilla.scripts.s15_escala_datos --solo-tabla
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p1_rodilla.scripts.s05_seleccionar_estudios import load_weak, pick_series
from p1_rodilla.src.data_io import load_gold
from p1_rodilla.src.engine import load_split
from p1_rodilla.src.metrics import bootstrap_macro_auc, macro_auc, safe_auc


def build_splits(cfg: dict, log) -> dict[int, str]:
    """Particiones anidadas por tamaño de entrenamiento: {N: ruta relativa a interim_dir}."""
    names = cfg["labels"]["names"]
    sel = cfg["selection"]
    interim = Path(cfg["data"]["interim_dir"])
    base = pd.read_csv(interim / "split.csv", dtype={"study_id": str, "series_id": str})
    base_train = base[base.split == "train"]
    fixed = base[base.split != "train"]

    meta = pd.read_parquet(interim / "series_meta.parquet")
    meta["study_id"], meta["series_id"] = meta["study_id"].astype(str), meta["series_id"].astype(str)
    chosen = pick_series(meta, sel["series_priority"], sel["min_slices"])
    exclude = set(fixed.study_id) | set(load_gold(cfg).study_id) | set(base_train.study_id)
    rest = chosen[~chosen.study_id.isin(exclude)].merge(load_weak(cfg, interim), on="study_id")
    rest = rest.sample(frac=1.0, random_state=cfg.get("seed", 42)).reset_index(drop=True)
    rest["split"] = "train"
    rest = rest[base.columns]

    n_base, n_max = len(base_train), len(base_train) + len(rest)
    sizes = {n_max if str(s).lower() == "todos" else int(s) for s in cfg["scale"]["train_sizes"]}
    sizes = sorted(s for s in sizes if n_base < s <= n_max)
    log.info(f"Entrenamiento base: {n_base} | disponibles: {n_max} | tamaños adicionales: {sizes}")

    out = {n_base: "split.csv"}
    ensure_dir(interim / "escala")
    for n in sizes:
        split = pd.concat([base_train, rest.iloc[: n - n_base], fixed], ignore_index=True)
        rel = f"escala/split_n{n}.csv"
        split.to_csv(interim / rel, index=False)
        out[n] = rel
    return out


def run_name(exp: str, n: int, n_base: int, seed: int, base_seed: int) -> str:
    """Nombre de la corrida para una configuración, un tamaño y una semilla."""
    name = exp if n == n_base else f"{exp}_n{n}"
    return name if seed == base_seed else f"{name}_s{seed}"


def paired_delta(y: np.ndarray, p: np.ndarray, q: np.ndarray, n: int, seed: int) -> tuple[float, float, float, float]:
    """Diferencia de AUC macro (p - q) con IC 95 % y p bilateral por bootstrap pareado."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        d.append(macro_auc(y[idx], p[idx]) - macro_auc(y[idx], q[idx]))
    d = np.asarray(d)
    d = d[~np.isnan(d)]
    pval = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return macro_auc(y, p) - macro_auc(y, q), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), pval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--ref", default="B_resnet50_bce")
    ap.add_argument("--solo-tabla", action="store_true", help="solo resumir las corridas existentes")
    ap.add_argument("--sin-preprocesar", action="store_true", help="omitir la etapa 6")
    ap.add_argument("--gpu", default=None, help="índice de la GPU (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    runs = Path(cfg["data"]["runs_dir"])
    log = get_logger("s15", runs / "escala.log")
    names = cfg["labels"]["names"]
    sc = cfg["scale"]
    base_seed = int(cfg.get("seed", 42))
    ablation = {e["name"]: e["set"] for e in cfg["ablation"]["experiments"]}
    exps = [e for e in sc["experiments"] if e in ablation]
    for e in set(sc["experiments"]) - set(exps):
        log.warning(f"{e} no está en ablation.experiments; se omite")
    seeds = [int(s) for s in sc["seeds"]]

    splits = build_splits(cfg, log)
    n_base = min(splits)

    if not args.solo_tabla:
        env = dict(os.environ)
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if not args.sin_preprocesar:
            cmd = [sys.executable, "-m", "p1_rodilla.scripts.s06_preprocesar", "--config", args.config,
                   "--set", f"data.split_csv={splits[max(splits)]}"]
            log.info(">> " + " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

        plan = [(s, n, e) for s in seeds for n in sorted(splits) for e in exps]
        todo = [(s, n, e) for s, n, e in plan
                if not (runs / run_name(e, n, n_base, s, base_seed) / "summary.json").exists()]
        log.info(f"Corridas planificadas: {len(plan)} | pendientes: {len(todo)}")
        t_start = time.time()
        for i, (s, n, e) in enumerate(todo, 1):
            name = run_name(e, n, n_base, s, base_seed)
            sets = sum([["--set", x] for x in ablation[e]], []) + \
                ["--set", f"data.split_csv={splits[n]}", "--set", f"seed={s}"]
            t0 = time.time()
            log.info(f"[{i}/{len(todo)}] {name}")
            try:
                subprocess.run([sys.executable, "-m", "p1_rodilla.scripts.s07_entrenar", "--exp", name,
                                "--config", args.config, *sets], check=True, cwd=REPO_ROOT, env=env)
                subprocess.run([sys.executable, "-m", "p1_rodilla.scripts.s08_evaluar", "--exp", name,
                                "--config", args.config], check=True, cwd=REPO_ROOT, env=env)
                log.info(f"[{i}/{len(todo)}] {name} terminado en {(time.time() - t0) / 60:.1f} min "
                         f"(acumulado {(time.time() - t_start) / 3600:.2f} h)")
            except subprocess.CalledProcessError as err:
                log.error(f"[{i}/{len(todo)}] {name} falló (código {err.returncode}); se continúa")

    # ---------- Resumen ----------
    gold = load_gold(cfg).set_index("study_id")
    y = gold[names].to_numpy()
    n_train = {}
    for n, rel in splits.items():
        c = copy.deepcopy(cfg)
        c["data"]["split_csv"] = rel
        n_train[n] = int((load_split(c).split == "train").sum())

    rows, ens = [], {}
    for e in exps:
        for n in sorted(splits):
            probs = []
            for s in seeds:
                d = runs / run_name(e, n, n_base, s, base_seed)
                if not (d / "summary.json").exists():
                    continue
                sm = json.loads((d / "summary.json").read_text())
                p = pd.read_csv(d / "preds_gold.csv", dtype={"study_id": str}).set_index("study_id")
                probs.append(p.loc[gold.index, names].to_numpy())
                rows.append({"configuracion": e, "n_entrenamiento": n_train[n], "semilla": s, "corrida": d.name,
                             "auc_macro_val": sm["val_macro_auc"], "auc_macro_dorado": sm["gold_macro_auc"]})
            if probs:
                ens[(e, n)] = (len(probs), np.mean(probs, axis=0))
    if not rows:
        log.warning("Sin corridas evaluadas.")
        return
    per_run = pd.DataFrame(rows)

    summary, by_finding = [], []
    for (e, n), (k, p) in ens.items():
        g = per_run[(per_run.configuracion == e) & (per_run.n_entrenamiento == n_train[n])].auc_macro_dorado
        lo, hi = bootstrap_macro_auc(y, p, args.n, base_seed)
        r = {"configuracion": e, "n_entrenamiento": n_train[n], "n_semillas": k,
             "auc_dorado_media": g.mean(), "auc_dorado_de": g.std(ddof=1) if k > 1 else np.nan,
             "auc_dorado_ensamble": macro_auc(y, p), "ensamble_ic95_inf": lo, "ensamble_ic95_sup": hi}
        if n != n_base and (e, n_base) in ens:
            d, dl, dh, pv = paired_delta(y, p, ens[(e, n_base)][1], args.n, base_seed)
            r.update({"delta_vs_base": d, "delta_vs_base_ic95_inf": dl, "delta_vs_base_ic95_sup": dh,
                      "p_vs_base": pv})
        if e != args.ref and (args.ref, n) in ens:
            d, dl, dh, pv = paired_delta(y, p, ens[(args.ref, n)][1], args.n, base_seed)
            r.update({"delta_vs_ref": d, "delta_vs_ref_ic95_inf": dl, "delta_vs_ref_ic95_sup": dh,
                      "p_vs_ref": pv})
        summary.append(r)
        by_finding.append({"configuracion": e, "n_entrenamiento": n_train[n],
                           **{h: safe_auc(y[:, j], p[:, j]) for j, h in enumerate(names)}})
    summary = pd.DataFrame(summary).sort_values(["configuracion", "n_entrenamiento"])
    by_finding = pd.DataFrame(by_finding).sort_values(["configuracion", "n_entrenamiento"])

    tab_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")
    fig_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "figuras")
    per_run.to_csv(tab_dir / "escala_corridas.csv", index=False)
    summary.to_csv(tab_dir / "escala_resumen.csv", index=False)
    by_finding.to_csv(tab_dir / "escala_por_hallazgo.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = list(summary.groupby("configuracion", sort=False))
    for i, (c, (e, g)) in enumerate(zip(plt.cm.tab10.colors, groups)):
        shift = 1 + 0.03 * (i - (len(groups) - 1) / 2)          # desplazamiento horizontal para separar barras
        pr = per_run[per_run.configuracion == e]
        ax.scatter(pr.n_entrenamiento * shift, pr.auc_macro_dorado, color=c, alpha=0.4, s=18)
        ax.errorbar(g.n_entrenamiento * shift, g.auc_dorado_ensamble,
                    yerr=[g.auc_dorado_ensamble - g.ensamble_ic95_inf, g.ensamble_ic95_sup - g.auc_dorado_ensamble],
                    color=c, marker="o", capsize=3, label=f"{e} (ensamble)")
    ax.set_xscale("log")
    ticks = sorted(per_run.n_entrenamiento.unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("estudios de entrenamiento con etiquetas débiles (escala logarítmica)")
    ax.set_ylabel("AUC macro en el conjunto dorado")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "escala_auc.png", dpi=150)

    cols = [c for c in ["configuracion", "n_entrenamiento", "n_semillas", "auc_dorado_media", "auc_dorado_de",
                        "auc_dorado_ensamble", "ensamble_ic95_inf", "ensamble_ic95_sup", "delta_vs_base",
                        "p_vs_base", "delta_vs_ref", "p_vs_ref"] if c in summary.columns]
    log.info("\nResumen por configuración y tamaño:\n" + summary[cols].round(4).to_string(index=False))
    log.info(f"Guardado {tab_dir / 'escala_resumen.csv'}")


if __name__ == "__main__":
    main()
