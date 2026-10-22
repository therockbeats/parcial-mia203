"""
Lectura de las tablas de Kaggle según la sección `schema` del config.

Todas las demás partes del proyecto usan estas funciones, así que si Kaggle
cambia un nombre de columna solo hay que corregir el config, no el código.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def load_studies(cfg: dict) -> pd.DataFrame:
    """Tabla de estudios con la columna study_id normalizada a texto."""
    sch = cfg["schema"]
    df = _read_table(Path(cfg["data"]["raw_dir"]) / sch["studies_csv"])
    df = df.rename(columns={sch["study_col"]: "study_id"})
    df["study_id"] = df["study_id"].astype(str)
    return df


def load_gold(cfg: dict) -> pd.DataFrame:
    """
    Devuelve SOLO los estudios con etiqueta dorada (todas las columnas de etiqueta con valor),
    con columnas renombradas a los nombres internos (labels.names) y valores 0/1.
    """
    sch, names = cfg["schema"], cfg["labels"]["names"]
    df = load_studies(cfg)
    missing = [c for c in sch["label_cols"] if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas de etiqueta no encontradas en {sch['studies_csv']}: {missing}. "
                       "Corrige schema.label_cols en config.yaml según el inventario.")
    gold = df[["study_id"] + list(sch["label_cols"])].dropna(subset=sch["label_cols"], how="any").copy()
    gold.columns = ["study_id"] + list(names)
    thr = float(sch.get("gold_positive_threshold", 0.5))
    for n in names:
        gold[n] = (pd.to_numeric(gold[n], errors="coerce") >= thr).astype(int)
    return gold.reset_index(drop=True)


def load_reports(cfg: dict) -> pd.DataFrame:
    """Tabla study_id, report (texto libre del reporte radiológico)."""
    rep = cfg["schema"]["reports"]
    raw = Path(cfg["data"]["raw_dir"])
    if rep["source"] == "csv":
        df = _read_table(raw / rep["csv"])
        df = df.rename(columns={cfg["schema"]["study_col"]: "study_id", rep["col"]: "report"})
        df["study_id"] = df["study_id"].astype(str)
        df = df[["study_id", "report"]]
    elif rep["source"] == "folder":
        rows = []
        pattern = rep["glob"]
        folder = raw / Path(pattern).parent
        for f in sorted(folder.glob(Path(pattern).name.replace("{study_id}", "*"))):
            sid = f.stem
            rows.append({"study_id": sid, "report": f.read_text(encoding="utf-8", errors="ignore")})
        df = pd.DataFrame(rows)
    else:
        raise ValueError("schema.reports.source debe ser 'csv' o 'folder'")
    df["report"] = df["report"].fillna("").astype(str)
    return df[df["report"].str.strip().str.len() > 0].reset_index(drop=True)


def label_matrix(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    return df[names].to_numpy(dtype=np.float32)
