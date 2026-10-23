"""Capas auxiliares por chip: Sentinel-2, agua permanente, Lima (origen, cobertura, escena previa, relieve)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import label_path, s1_path, s2_path


def s1_db(cfg: dict, chip_id: str) -> np.ndarray:
    return np.load(s1_path(cfg, chip_id)).astype(np.float32)


def lima_aux(cfg: dict, chip_id: str, name: str):
    f = Path(cfg["data"]["processed_dir"]) / "lima_aux" / f"{chip_id}_{name}.npy"
    return np.load(f).astype(np.float32) if f.exists() else None


def s2_indices(cfg: dict, chip_id: str):
    """NDVI, MNDWI y composición RGB del chip dorado (None si no hay S2)."""
    f = s2_path(cfg, chip_id)
    if not f.exists():
        return None
    b = np.load(f).astype(np.float32) / float(cfg["preprocess"]["s2_scale"])
    blue, green, red, nir, swir1 = b[0], b[1], b[2], b[3], b[4]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    mndwi = (green - swir1) / (green + swir1 + eps)
    rgb = np.clip(np.stack([red, green, blue], axis=-1) * 3.5, 0, 1)
    return {"ndvi": ndvi, "mndwi": mndwi, "rgb": rgb}


def jrc(cfg: dict, chip_id: str):
    f = label_path(cfg, "JRCWaterHand", chip_id)
    return np.load(f) if f.exists() else None


def slope_deg(dem: np.ndarray, pixel_m: float = 10.0) -> np.ndarray:
    """Pendiente en grados a partir de un modelo de elevación con píxeles de ``pixel_m`` metros."""
    gy, gx = np.gradient(np.nan_to_num(dem, nan=np.nanmean(dem) if np.isfinite(dem).any() else 0.0), pixel_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def lulc_group(code: np.ndarray) -> np.ndarray:
    """Agrupa los códigos de cobertura (nomenclatura tipo Corine) en cinco clases."""
    g = np.full(code.shape, "sin dato", dtype=object)
    c = code.astype(int)
    g[(c >= 100) & (c < 200)] = "urbano"
    g[(c >= 200) & (c < 300)] = "agrícola"
    g[(c >= 300) & (c < 400)] = "vegetación natural"
    g[np.isin(c, [331, 332])] = "arena y roca desnuda"
    g[c >= 400] = "agua y humedales"
    return g
