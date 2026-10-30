# Este módulo centraliza la lectura y preparación
# de los datos provenientes del conjunto RSNA,
# incluyendo estudios, etiquetas y reportes.



"""Lectura de las tablas del conjunto RSNA según la sección ``schema`` del config."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Lee automáticamente archivos CSV o Parquet
# según la extensión detectada en la ruta.

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)

# Construye la ruta del archivo que contiene
# la partición de entrenamiento, validación y prueba.
def split_path(cfg: dict) -> Path:
    """Ruta de la partición vigente: ``data.split_csv`` relativo a ``interim_dir``."""
    return Path(cfg["data"]["interim_dir"]) / cfg["data"].get("split_csv", "split.csv")

# Carga la tabla principal de estudios MRI y
# normaliza el identificador como texto.
def load_studies(cfg: dict) -> pd.DataFrame:
    """Tabla de estudios con el identificador normalizado como ``study_id`` (texto)."""
    sch = cfg["schema"]
    df = _read_table(Path(cfg["data"]["raw_dir"]) / sch["studies_csv"])
    
    # Estandariza el nombre de la columna de identificación.
    df = df.rename(columns={sch["study_col"]: "study_id"})
    df["study_id"] = df["study_id"].astype(str)
    return df

# Obtiene las etiquetas de referencia ("gold standard")
# utilizadas para entrenar y evaluar el modelo.
def load_gold(cfg: dict) -> pd.DataFrame:
    """
    Estudios con etiqueta dorada completa.

    Devuelve ``study_id`` y una columna binaria por hallazgo, con los nombres internos
    de ``labels.names``.
    """
    sch, names = cfg["schema"], cfg["labels"]["names"]
    df = load_studies(cfg)

    # Verifica que todas las columnas de etiquetas
    # existan en el archivo fuente.
    missing = [c for c in sch["label_cols"] if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas de etiqueta no encontradas en {sch['studies_csv']}: {missing}. "
                       "Revisar schema.label_cols en config.yaml.")
    # Conserva únicamente el identificador del estudio
    # y las columnas asociadas a los hallazgos.
    gold = df[["study_id"] + list(sch["label_cols"])].dropna(subset=sch["label_cols"], how="any").copy()
    gold.columns = ["study_id"] + list(names)
    thr = float(sch.get("gold_positive_threshold", 0.5))
    
    # Convierte las etiquetas a formato binario
    # usando el umbral definido en config.yaml.
    for n in names:
        gold[n] = (pd.to_numeric(gold[n], errors="coerce") >= thr).astype(int)
    return gold.reset_index(drop=True)

    # Carga los reportes radiológicos asociados
    # a cada estudio MRI.
def load_reports(cfg: dict) -> pd.DataFrame:
    """Tabla ``study_id``, ``report`` con los reportes no vacíos."""
    rep = cfg["schema"]["reports"]
    raw = Path(cfg["data"]["raw_dir"])

    # Caso 1: los reportes están almacenados
    # dentro de un archivo tabular.
    if rep["source"] == "csv":
        df = _read_table(raw / rep["csv"])
        df = df.rename(columns={cfg["schema"]["study_col"]: "study_id", rep["col"]: "report"})
        df["study_id"] = df["study_id"].astype(str)
        df = df[["study_id", "report"]]

    # Caso 2: los reportes están almacenados
    # como archivos de texto individuales.
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
    # Elimina reportes vacíos antes de continuar
    return df[df["report"].str.strip().str.len() > 0].reset_index(drop=True)

# Convierte las columnas de etiquetas a una matriz NumPy
# utilizada durante el entrenamiento del modelo.
def label_matrix(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Matriz (n_estudios, n_hallazgos) en float32."""
    return df[names].to_numpy(dtype=np.float32)
