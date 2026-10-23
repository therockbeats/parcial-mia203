"""
Etapa 2. Metadatos de todas las series.

Lee en paralelo la cabecera del corte central de cada serie (sin píxeles) y obtiene plano,
supresión grasa, ponderación, número de cortes, dimensiones, espesor, fabricante, campo
magnético e institución. Si existe la tabla oficial de series, su plano, supresión grasa y
sensibilidad a líquido prevalecen sobre los valores deducidos de la cabecera, que se conservan
como ``plane_dicom`` y ``fat_sat_dicom`` para medir su concordancia.

Salidas:
    <interim_dir>/series_meta.parquet
    p1_rodilla/reports/tablas/series_resumen.csv
    p1_rodilla/reports/tablas/concordancia_dicom_vs_oficial.csv

Uso:
    python -m p1_rodilla.scripts.s02_metadatos_series [--workers 24] [--limit 50]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common.config import REPO_ROOT, load_config
from common.utils import Timer, ensure_dir, get_logger
from p1_rodilla.src.dicom_utils import read_series_header


def _one(args):
    """Metadatos de una serie (ejecutado en un proceso del pool)."""
    study_id, series_dir, ext = args
    row = {"study_id": study_id, "series_id": Path(series_dir).name}
    try:
        row.update(read_series_header(Path(series_dir), ext))
    except Exception as e:
        row["error"] = str(e)[:200]
    return row


def merge_official_series(meta: pd.DataFrame, cfg: dict, log) -> pd.DataFrame:
    """
    Une la tabla oficial de series y armoniza ``plane``, ``fat_sat`` y ``fluid_sensitive``.

    Prevalece la anotación oficial; los valores deducidos de la cabecera DICOM (o, para la
    sensibilidad a líquido, la ponderación PD/T2) solo completan valores faltantes.
    """
    sch = cfg["schema"]
    meta["plane_dicom"], meta["fat_sat_dicom"] = meta.get("plane"), meta.get("fat_sat")
    meta["fluid_sensitive"] = meta.get("weighting", pd.Series(index=meta.index, dtype=object)).isin(["PD", "T2"])
    p = Path(cfg["data"]["raw_dir"]) / sch["series_csv"] if sch.get("series_csv") else None
    if p is None or not p.exists():
        log.info("Sin tabla oficial de series: se usan solo las cabeceras DICOM")
        return meta

    s = pd.read_csv(p, low_memory=False).rename(columns={sch["study_col"]: "study_id", sch["series_col"]: "series_id"})
    s["study_id"], s["series_id"] = s["study_id"].astype(str), s["series_id"].astype(str)
    meta = meta.merge(s, on=["study_id", "series_id"], how="left", suffixes=("", "_oficial"))
    log.info(f"Unida la tabla oficial {p.name}: {s.shape[0]:,} filas; "
             f"{meta[sch['series_plane_col']].notna().sum():,} de {len(meta):,} series con anotación")

    off_plane = meta[sch["series_plane_col"]].astype("string").str.lower()
    meta["plane"] = off_plane.where(off_plane.notna(), meta["plane_dicom"])
    for col_cfg, dest, fallback in [("series_fat_sat_col", "fat_sat", meta["fat_sat_dicom"]),
                                    ("series_fluid_col", "fluid_sensitive", meta["fluid_sensitive"])]:
        c = sch.get(col_cfg)
        if c and c in meta.columns:
            off = meta[c].map({1: True, 0: False, "1": True, "0": False})
            meta[dest] = off.where(off.notna(), fallback.fillna(False).astype(bool)).astype(bool)

    # Concordancia entre la cabecera DICOM y la anotación oficial
    both = meta[off_plane.notna()]
    if len(both):
        acc_plane = (both["plane_dicom"] == both["plane"]).mean()
        acc_fs = (both["fat_sat_dicom"].fillna(False).astype(bool) == both["fat_sat"]).mean()
        log.info(f"Concordancia con la anotación oficial: plano {acc_plane:.1%} | supresión grasa {acc_fs:.1%}")
        conc = pd.DataFrame({"variable": ["plano", "supresion_grasa"], "concordancia": [acc_plane, acc_fs],
                             "n_series": [len(both), len(both)]})
        conc.to_csv(ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas") / "concordancia_dicom_vs_oficial.csv",
                    index=False)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger("s02")
    sch = cfg["schema"]
    images_root = Path(cfg["data"]["raw_dir"]) / sch["images_subdir"]
    ext = sch.get("dicom_ext", ".dcm").lower()
    workers = args.workers or cfg["preprocess"].get("num_workers", 8)

    # 1) Carpetas <images_root>/<study>/<series>
    tasks = []
    studies = sorted(p for p in images_root.iterdir() if p.is_dir())
    if args.limit:
        studies = studies[: args.limit]
    for st in studies:
        for se in sorted(p for p in st.iterdir() if p.is_dir()):
            tasks.append((st.name, str(se), ext))
    log.info(f"{len(studies):,} estudios y {len(tasks):,} series encontradas en {images_root}")

    # 2) Lectura de cabeceras en paralelo (un proceso por núcleo)
    rows = []
    with Timer("Lectura de cabeceras", log), ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="series"):
            rows.append(f.result())
    meta = pd.DataFrame(rows)

    # 3) Anotación oficial de series
    meta = merge_official_series(meta, cfg, log)

    out = ensure_dir(cfg["data"]["interim_dir"]) / "series_meta.parquet"
    meta.to_parquet(out, index=False)
    log.info(f"Guardado {out}")

    # 4) Resumen por plano y secuencia
    summ = (meta.groupby(["plane", "fat_sat", "fluid_sensitive", "weighting"], dropna=False)
            .agg(series=("series_id", "count"), estudios=("study_id", "nunique"), cortes_mediana=("n_slices", "median"))
            .reset_index().sort_values("series", ascending=False))
    tab = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas") / "series_resumen.csv"
    summ.to_csv(tab, index=False)
    log.info("\n" + summ.head(15).to_string(index=False))
    if "error" in meta.columns:
        log.info(f"Series con error de lectura: {meta['error'].notna().sum()}")


if __name__ == "__main__":
    main()
