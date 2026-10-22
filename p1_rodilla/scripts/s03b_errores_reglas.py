"""
PASO 3b. Análisis de errores de las reglas de etiquetado sobre los dorados de DESARROLLO.

Para cada hallazgo lista los falsos positivos (regla = positivo/incierto, dorado = 0) y los
falsos negativos (regla = negativo/no mencionado, dorado = 1), con las cláusulas del reporte
que originaron la decisión de la regla. Al final incluye el texto completo de cada reporte
con algún error. Opcionalmente añade una muestra de reportes NO dorados de un idioma, para
estudiar su vocabulario.

Los estudios de RESERVA no se exportan: su acuerdo debe seguir siendo una estimación
independiente de la calidad de las reglas.

La salida contiene texto de los reportes (datos de la competencia), por eso se escribe en
<interim_dir> (fuera del repositorio) y no se versiona.

Salida:  <interim_dir>/errores_reglas_desarrollo.md

Uso:
    python -m p1_rodilla.scripts.s03b_errores_reglas
    (opcional) --idioma bg --n 15     añade 15 reportes no dorados en búlgaro
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p1_rodilla.src.data_io import load_gold, load_reports
from p1_rodilla.src.labels_rules import LABELS, NEG, POS, STATUS_NAME, UNC, extract_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--idioma", default=None, help="código de idioma para la muestra de reportes no dorados")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s03b")
    interim = Path(cfg["data"]["interim_dir"])

    part_path = interim / "gold_particion.csv"
    if not part_path.exists():
        raise FileNotFoundError("Falta gold_particion.csv: ejecutar antes s03_etiquetas_reglas")
    part = pd.read_csv(part_path, dtype={"study_id": str})
    dev_ids = set(part.loc[part.particion == "desarrollo", "study_id"])

    gold = load_gold(cfg)
    gold = gold[gold.study_id.isin(dev_ids)].set_index("study_id")
    reports = load_reports(cfg).set_index("study_id")
    langs = pd.read_parquet(interim / "labels_rules.parquet")[["study_id", "language"]].set_index("study_id")["language"]

    # Reaplicar las reglas con captura de evidencia
    rule, evid = {}, {}
    for sid in gold.index:
        ev: dict[str, list[str]] = {}
        rule[sid] = extract_labels(reports.loc[sid, "report"], evidence=ev)
        evid[sid] = ev

    lines = ["# Errores de las reglas en los dorados de desarrollo", "",
             f"Estudios de desarrollo: {len(gold)}. Cada estudio se identifica por los últimos 8 caracteres "
             "de su StudyInstanceUID.", ""]
    with_errors: set[str] = set()
    summary = []
    for lab in LABELS:
        fp, fn = [], []
        for sid in gold.index:
            g, r = int(gold.loc[sid, lab]), rule[sid][lab]
            pred = int(r in (POS, UNC))
            if pred == 1 and g == 0:
                fp.append(sid)
            elif pred == 0 and g == 1:
                fn.append(sid)
        summary.append({"hallazgo": lab, "falsos_positivos": len(fp), "falsos_negativos": len(fn)})
        lines += [f"## {lab}  (FP = {len(fp)}, FN = {len(fn)})", ""]
        for kind, ids in (("Falso positivo", fp), ("Falso negativo", fn)):
            for sid in ids:
                with_errors.add(sid)
                lines.append(f"- **{kind}** · `{sid[-8:]}` · idioma `{langs.get(sid, '?')}` · "
                             f"regla: {STATUS_NAME[rule[sid][lab]]}")
                for e in evid[sid].get(lab, []) or ["(el hallazgo no se encontró en ninguna cláusula)"]:
                    lines.append(f"    - {e}")
        lines.append("")

    lines += ["# Reportes completos con algún error", ""]
    for sid in sorted(with_errors):
        lines += [f"## `{sid[-8:]}` · idioma `{langs.get(sid, '?')}`", "",
                  "Dorado positivo: " + ", ".join(l for l in LABELS if int(gold.loc[sid, l]) == 1), "",
                  "```", reports.loc[sid, "report"].strip(), "```", ""]

    if args.idioma:
        pool = langs[(langs == args.idioma) & ~langs.index.isin(part.study_id)].index
        sample = pd.Index(pool).to_series().sample(min(args.n, len(pool)), random_state=cfg.get("seed", 42))
        lines += [f"# Muestra de reportes no dorados en `{args.idioma}` ({len(sample)} de {len(pool)})", ""]
        for sid in sample:
            lab = extract_labels(reports.loc[sid, "report"])
            pos = [k for k, v in lab.items() if v == POS]
            neg = [k for k, v in lab.items() if v == NEG]
            lines += [f"## `{sid[-8:]}`", "", f"Regla positivo: {', '.join(pos) or '-'} | negativo: {', '.join(neg) or '-'}",
                      "", "```", reports.loc[sid, "report"].strip(), "```", ""]

    out = ensure_dir(interim) / "errores_reglas_desarrollo.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("\n" + pd.DataFrame(summary).to_string(index=False))
    log.info(f"{len(with_errors)} reportes con al menos un error. Guardado {out}")


if __name__ == "__main__":
    main()
