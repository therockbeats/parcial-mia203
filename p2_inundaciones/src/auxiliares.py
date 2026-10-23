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


def _dem_gradients(dem: np.ndarray, pixel_m: float, pixel_x_m: float | None, smooth: int):
    """Derivadas del relieve hacia el este y hacia el norte (m/m).

    El SRTM de 30 m llega remuestreado por vecino más cercano a la grilla de 10 m, con escalones de
    3 x 3 píxeles; un filtro de media de ``smooth`` píxeles los suaviza antes de derivar.
    """
    from scipy.ndimage import uniform_filter

    z = np.nan_to_num(dem, nan=np.nanmean(dem) if np.isfinite(dem).any() else 0.0).astype(np.float64)
    if smooth > 1:
        z = uniform_filter(z, size=smooth, mode="nearest")
    d_row, d_col = np.gradient(z, pixel_m, pixel_x_m or pixel_m)
    return d_col, -d_row


def slope_deg(dem: np.ndarray, pixel_m: float = 10.0, pixel_x_m: float | None = None, smooth: int = 3) -> np.ndarray:
    """Pendiente en grados a partir de un modelo de elevación con píxeles de ``pixel_m`` metros."""
    dzde, dzdn = _dem_gradients(dem, pixel_m, pixel_x_m, smooth)
    return np.degrees(np.arctan(np.hypot(dzde, dzdn)))


def local_incidence_deg(dem: np.ndarray, incidence_deg: np.ndarray, sensor_azimuth_deg: float,
                        pixel_m: float = 10.0, pixel_x_m: float | None = None, smooth: int = 3) -> np.ndarray:
    """Ángulo de incidencia local: entre la normal del terreno y la dirección hacia el satélite.

    Menor que el ángulo nominal en laderas orientadas hacia el radar (escorzo) y mayor en las
    opuestas; a partir de 90 grados la ladera queda de espaldas al radar y no devuelve señal.
    """
    dzde, dzdn = _dem_gradients(dem, pixel_m, pixel_x_m, smooth)
    norm = np.sqrt(dzde ** 2 + dzdn ** 2 + 1.0)
    th = np.radians(np.nan_to_num(incidence_deg, nan=np.nanmean(incidence_deg)))
    az = np.radians(sensor_azimuth_deg)
    cos_loc = (-dzde * np.sin(th) * np.sin(az) - dzdn * np.sin(th) * np.cos(az) + np.cos(th)) / norm
    return np.degrees(np.arccos(np.clip(cos_loc, -1.0, 1.0)))


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
