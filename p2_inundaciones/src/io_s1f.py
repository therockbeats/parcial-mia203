"""Lectura de chips de Sen1Floods11, tabla de chips y normalización de Sentinel-1."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABEL_IGNORE = -1


def raw_root(cfg: dict) -> Path:
    """Raíz local de la versión descargada del bucket (``<raw_dir>/v1.1``)."""
    return Path(cfg["data"]["raw_dir"]) / cfg["dataset"]["version_prefix"]


def chip_id_from_name(name: str) -> str:
    """``Ghana_103272_S1Hand.tif`` -> ``Ghana_103272``."""
    stem = Path(name).stem
    return "_".join(stem.split("_")[:-1])


def event_from_chip(chip_id: str) -> str:
    """Evento (país o región) a partir del identificador del chip."""
    return chip_id.rsplit("_", 1)[0]


def read_split_csv(path: Path) -> list[str]:
    """Identificadores de chip de un CSV de partición (columnas: imagen S1, etiqueta; sin cabecera)."""
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(chip_id_from_name(line.split(",")[0]))
    return ids


def read_tif(path: Path) -> np.ndarray:
    """Lee todas las bandas de un GeoTIFF como arreglo (bandas, filas, columnas)."""
    import rasterio

    with rasterio.open(path) as r:
        return r.read()


def label_to_int8(arr: np.ndarray) -> np.ndarray:
    """Convierte una etiqueta a int8 con -1 (ignorar), 0 (no agua) y 1 (agua)."""
    a = np.asarray(arr).squeeze()
    out = np.full(a.shape, LABEL_IGNORE, dtype=np.int8)
    out[a == 0] = 0
    out[a == 1] = 1
    return out


def s1_to_unit(db: np.ndarray, cfg: dict) -> np.ndarray:
    """Recorta el S1 en dB a [clip_min, clip_max] y lo lleva a [0, 1]; los NaN pasan a 0."""
    pp = cfg["preprocess"]
    lo, hi = float(pp["clip_min"]), float(pp["clip_max"])
    x = np.nan_to_num(db.astype(np.float32), nan=lo)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)


def normalize_s1(db: np.ndarray, cfg: dict) -> np.ndarray:
    """S1 en dB (2, H, W) a entrada estandarizada del modelo."""
    pp = cfg["preprocess"]
    x = s1_to_unit(db, cfg)
    mean = np.asarray(pp["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(pp["std"], dtype=np.float32)[:, None, None]
    return (x - mean) / std


def normalize_s2(refl: np.ndarray, cfg: dict) -> np.ndarray:
    """Reflectancia S2 (bandas seleccionadas, entero x 10000) a [0, 1] aproximado."""
    return np.clip(refl.astype(np.float32) / float(cfg["preprocess"]["s2_scale"]), 0.0, 1.0)


def load_chips(cfg: dict) -> pd.DataFrame:
    """Tabla de chips generada por la etapa 3."""
    p = Path(cfg["data"]["interim_dir"]) / "chips.parquet"
    if not p.exists():
        raise FileNotFoundError("Falta chips.parquet: ejecutar s03_preparar_chips")
    return pd.read_parquet(p)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Umbral de Otsu sobre valores finitos."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(np.float64)
    w0 = np.cumsum(w)
    w1 = w0[-1] - w0
    m = np.cumsum(w * centers)
    mu0 = m / np.maximum(w0, 1e-12)
    mu1 = (m[-1] - m) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(between))])
