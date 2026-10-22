"""
PASO 3. Etiquetas débiles con REGLAS multilingües (entregable 1 del enunciado).

1. Lee todos los reportes (texto libre) de los estudios de entrenamiento.
2. Detecta el idioma de cada reporte (para el análisis de sesgo lingüístico).
3. Aplica las reglas de p1_rodilla/src/labels_rules.py -> código por hallazgo:
       1 positivo, 0 negativo, 2 incierto, -1 no mencionado
4. Compara contra los estudios con etiqueta dorada: acuerdo, kappa, sensibilidad,
   especificidad y tasa de ruido estimada (1 - acuerdo), global y por idioma.

Salidas:
    <interim_dir>/labels_rules.parquet
    p1_rodilla/reports/tablas/acuerdo_reglas_vs_dorado.csv
    p1_rodilla/reports/tablas/acuerdo_reglas_por_idioma.csv
    p1_rodilla/reports/tablas/idiomas.csv

Uso:  python -m p1_rodilla.scripts.s03_etiquetas_reglas
"""
from __future__ import annotations

from pathlib import Path

import argparse

import pandas as pd
from tqdm import tqdm

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p1_rodilla.src.data_io import load_gold, load_reports
from p1_rodilla.src.labels_common import agreement_report
from p1_rodilla.src.labels_rules import LABELS, STATUS_NAME, detect_language, extract_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
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
