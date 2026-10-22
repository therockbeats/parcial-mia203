"""
PASO 3. Etiquetas débiles con REGLAS multilingües (entregable 1 del enunciado).

1. Lee todos los reportes (texto libre) de los estudios de entrenamiento.
2. Detecta el idioma de cada reporte (para el análisis de sesgo lingüístico).
3. Aplica las reglas de p1_rodilla/src/labels_rules.py -> código por hallazgo:
       1 positivo, 0 negativo, 2 incierto, -1 no mencionado
4. Compara contra los estudios con etiqueta dorada: acuerdo, kappa, sensibilidad,
   especificidad y tasa de ruido estimada (1 - acuerdo), global y por idioma.
5. Particiona los estudios dorados en DESARROLLO y RESERVA (labels.gold_dev_fraction).
   El análisis de errores para refinar las reglas usa solo DESARROLLO; el acuerdo en
   RESERVA es la estimación no sesgada de la calidad de cada versión de las reglas.

Salidas:
    <interim_dir>/labels_rules.parquet
    p1_rodilla/reports/tablas/acuerdo_reglas_vs_dorado.csv
    p1_rodilla/reports/tablas/acuerdo_reglas_por_idioma.csv
    p1_rodilla/reports/tablas/idiomas.csv
    p1_rodilla/reports/tablas/acuerdo_reglas_historial.csv   (una fila por versión de reglas)
    <interim_dir>/gold_particion.csv

Uso:  python -m p1_rodilla.scripts.s03_etiquetas_reglas --version v1
"""
from __future__ import annotations

from pathlib import Path

import argparse

import pandas as pd
from tqdm import tqdm

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p1_rodilla.src.data_io import load_gold, load_reports
import numpy as np

from p1_rodilla.src.labels_common import agreement_report
from p1_rodilla.src.labels_rules import LABELS, STATUS_NAME, detect_language, extract_labels


def gold_partition(gold: pd.DataFrame, cfg: dict, log) -> pd.DataFrame:
    """
    Divide los estudios dorados en 'desarrollo' y 'reserva' de forma determinista (semilla del
    config). La partición se guarda y se reutiliza para que no cambie entre versiones de reglas.
    """
    path = ensure_dir(cfg["data"]["interim_dir"]) / "gold_particion.csv"
    if path.exists():
        part = pd.read_csv(path, dtype={"study_id": str})
        if set(part.study_id) == set(gold.study_id):
            return part
    ids = sorted(gold.study_id)
    rng = np.random.default_rng(cfg.get("seed", 42))
    rng.shuffle(ids)
    n_dev = int(round(float(cfg["labels"].get("gold_dev_fraction", 0.5)) * len(ids)))
    part = pd.DataFrame({"study_id": ids, "particion": ["desarrollo"] * n_dev + ["reserva"] * (len(ids) - n_dev)})
    part.to_csv(path, index=False)
    log.info(f"Partición de dorados creada: {n_dev} desarrollo / {len(ids) - n_dev} reserva -> {path}")
    return part


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--version", default="v1", help="etiqueta de la versión de las reglas (historial)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s03")
    names = cfg["labels"]["names"]
    assert names == LABELS, "labels.names debe coincidir con LABELS de labels_rules.py (mismo orden)"

    reports = load_reports(cfg)
    log.info(f"{len(reports):,} reportes cargados")

    rows = []
    for sid, text in tqdm(zip(reports.study_id, reports.report), total=len(reports), desc="reglas"):
        lab = extract_labels(text)
        lab["study_id"] = sid
        lab["language"] = detect_language(text)
        lab["report_chars"] = len(text)
        rows.append(lab)
    weak = pd.DataFrame(rows)[["study_id", "language", "report_chars"] + names]

    out = ensure_dir(cfg["data"]["interim_dir"]) / "labels_rules.parquet"
    weak.to_parquet(out, index=False)
    log.info(f"Guardado {out}")

    tab_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")
    # Distribución de estados por hallazgo (cuántos positivos/negativos/... encontró la regla)
    dist = pd.DataFrame({n: weak[n].map(STATUS_NAME).value_counts() for n in names}).T.fillna(0).astype(int)
    dist.to_csv(tab_dir / "distribucion_etiquetas_reglas.csv")
    log.info("\nDistribución de etiquetas débiles:\n" + dist.to_string())
    weak["language"].value_counts().rename_axis("idioma").reset_index(name="reportes").to_csv(tab_dir / "idiomas.csv", index=False)

    # Acuerdo con etiquetas doradas
    try:
        gold = load_gold(cfg)
    except KeyError as e:
        log.warning(f"No pude cargar las etiquetas doradas: {e}")
        return
    log.info(f"{len(gold)} estudios con etiqueta dorada")
    agr = agreement_report(weak, gold, names)
    agr.to_csv(tab_dir / "acuerdo_reglas_vs_dorado.csv", index=False)
    log.info("\nAcuerdo reglas vs dorado:\n" + agr.round(3).to_string(index=False))

    # Partición fija de los dorados: DESARROLLO (análisis de errores) y RESERVA (estimación final)
    part = gold_partition(gold, cfg, log)
    by_part = {}
    for name in ("desarrollo", "reserva"):
        ids = part.loc[part.particion == name, "study_id"]
        a = agreement_report(weak, gold[gold.study_id.isin(ids)], names)
        a.to_csv(tab_dir / f"acuerdo_reglas_{name}.csv", index=False)
        by_part[name] = a.set_index("hallazgo")
    comp = pd.DataFrame({f"kappa_{k}": v["kappa"] for k, v in by_part.items()})
    comp["kappa_total"] = agr.set_index("hallazgo")["kappa"]
    log.info("\nKappa por partición de los dorados:\n" + comp.round(3).to_string())

    hist_path = tab_dir / "acuerdo_reglas_historial.csv"
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    row = {"version": args.version}
    for name, a in by_part.items():
        row[f"kappa_{name}"] = a.loc["PROMEDIO", "kappa"]
        row[f"acuerdo_{name}"] = a.loc["PROMEDIO", "acuerdo"]
        for n in names:
            row[f"{n}_kappa_{name}"] = a.loc[n, "kappa"]
    hist = pd.concat([hist[hist.get("version", pd.Series(dtype=str)) != args.version] if len(hist) else hist,
                      pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(hist_path, index=False)
    log.info("\nHistorial de versiones (kappa promedio):\n"
             + hist[["version", "kappa_desarrollo", "kappa_reserva", "acuerdo_desarrollo", "acuerdo_reserva"]].round(3).to_string(index=False))

    # Acuerdo por idioma (para la discusión de sesgo geográfico/lingüístico)
    m = gold.merge(weak[["study_id", "language"]], on="study_id")
    per_lang = []
    for lang, g in m.groupby("language"):
        a = agreement_report(weak, g[["study_id"] + names], names)
        row = a[a.hallazgo == "PROMEDIO"].iloc[0].to_dict()
        row["idioma"] = lang
        per_lang.append(row)
    if per_lang:
        pl = pd.DataFrame(per_lang)[["idioma", "n", "acuerdo", "kappa", "sensibilidad", "especificidad"]]
        pl.to_csv(tab_dir / "acuerdo_reglas_por_idioma.csv", index=False)
        log.info("\nAcuerdo por idioma (pocos estudios por idioma: interpretar con cautela):\n" + pl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
