"""
PASO 13. Comparación PAREADA de los experimentos de la ablación en el conjunto dorado.

Por qué pareada: todos los experimentos se evalúan sobre los MISMOS 58 estudios. Comparar
intervalos de confianza separados es conservador (ignora que los errores de dos modelos sobre
el mismo estudio están correlacionados). El bootstrap pareado remuestrea estudios una sola vez
por iteración y calcula, sobre esa misma muestra, la diferencia de AUC macro entre cada
experimento y el de referencia:

    delta_b = AUC_macro(exp, muestra_b) - AUC_macro(ref, muestra_b),   b = 1..n

El IC 95 % de delta son los percentiles 2.5 y 97.5. Si el intervalo no contiene 0, la diferencia
es significativa al 5 %. p_bilateral = 2 * min(P(delta <= 0), P(delta >= 0)).

Salida:  p1_rodilla/reports/tablas/comparacion_pareada.csv

Uso:
    python -m p1_rodilla.scripts.s13_comparacion_pareada
    (opcional) --ref B_resnet50_bce   experimento de referencia
               --n 2000               remuestreos bootstrap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p1_rodilla.src.data_io import load_gold
from p1_rodilla.src.metrics import macro_auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--ref", default="B_resnet50_bce")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s13")
    names = cfg["labels"]["names"]
    runs = Path(cfg["data"]["runs_dir"])

    gold = load_gold(cfg).set_index("study_id")
    preds = {}
    for e in cfg["ablation"]["experiments"]:
        f = runs / e["name"] / "preds_gold.csv"
        if f.exists():
            p = pd.read_csv(f, dtype={"study_id": str}).set_index("study_id")
            preds[e["name"]] = p.loc[gold.index, names].to_numpy()
    if not preds:
        raise FileNotFoundError("No hay preds_gold.csv de ningún experimento (ejecutar s08 o s10)")
    if args.ref not in preds:
        log.warning(f"Sin predicciones de {args.ref}; se usa {next(iter(preds))} como referencia")
        args.ref = next(iter(preds))
    y = gold[names].to_numpy()
    log.info(f"{len(preds)} experimentos, {len(y)} estudios dorados, {args.n} remuestreos")

    rng = np.random.default_rng(cfg.get("seed", 42))
    deltas = {k: [] for k in preds}
    for _ in range(args.n):
        idx = rng.integers(0, len(y), len(y))             # misma muestra para todos los modelos
        ref = macro_auc(y[idx], preds[args.ref][idx])
        for k, p in preds.items():
            deltas[k].append(macro_auc(y[idx], p[idx]) - ref)

    rows = []
    for k, d in deltas.items():
        d = np.asarray(d)
        d = d[~np.isnan(d)]
        rows.append({
            "experimento": k,
            "auc_macro_dorado": macro_auc(y, preds[k]),
            "delta_vs_ref": macro_auc(y, preds[k]) - macro_auc(y, preds[args.ref]),
            "delta_ic95_inf": np.percentile(d, 2.5),
            "delta_ic95_sup": np.percentile(d, 97.5),
            "p_bilateral": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if k != args.ref else np.nan,
        })
    tab = pd.DataFrame(rows).sort_values("auc_macro_dorado", ascending=False)
    out = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas") / "comparacion_pareada.csv"
    tab.to_csv(out, index=False)
    log.info(f"\nDiferencia de AUC macro frente a {args.ref} (bootstrap pareado):\n" + tab.round(4).to_string(index=False))
    log.info(f"Guardado {out}")


if __name__ == "__main__":
    main()
