"""Grilla de chips de la región de Lima y rasterización de capas vectoriales sobre ella."""
from __future__ import annotations

import math

import numpy as np


def lima_grid(cfg: dict) -> list[dict]:
    """
    Chips de ``chip_size`` píxeles que cubren ``lima.bbox`` en EPSG:4326.

    El tamaño de píxel es el de Sen1Floods11 (``pixel_deg``). Cada chip tiene identificador
    ``Lima_rXXcYY``, transformación afín y límites.
    """
    from rasterio.transform import from_origin

    lm = cfg["lima"]
    x0, y0, x1, y1 = lm["bbox"]
    px, n = float(lm["pixel_deg"]), int(lm["chip_size"])
    size = px * n
    ncol, nrow = math.ceil((x1 - x0) / size), math.ceil((y1 - y0) / size)
    chips = []
    for r in range(nrow):
        for c in range(ncol):
            left, top = x0 + c * size, y1 - r * size
            chips.append({"chip_id": f"Lima_r{r:02d}c{c:02d}", "row": r, "col": c,
                          "transform": from_origin(left, top, px, px),
                          "bounds": (left, top - size, left + size, top), "size": n})
    return chips


def clip_to_chip(geoms, chip: dict):
    """Recorta las geometrías al rectángulo del chip (con un margen), para rasterizar solo lo necesario."""
    import shapely

    left, bottom, right, top = chip["bounds"]
    pad = 2 * chip["transform"].a
    return shapely.clip_by_rect(np.asarray(list(geoms), dtype=object), left - pad, bottom - pad, right + pad, top + pad)


def rasterize_on_chip(geoms, chip: dict, value: int = 1, fill: int = 0, dtype="uint8") -> np.ndarray:
    """Rasteriza geometrías (EPSG:4326) sobre la grilla del chip; píxel marcado si su centro cae dentro."""
    from rasterio.features import rasterize

    shapes = [(g, value) for g in clip_to_chip(geoms, chip) if g is not None and not g.is_empty]
    n = chip["size"]
    if not shapes:
        return np.full((n, n), fill, dtype=dtype)
    return rasterize(shapes, out_shape=(n, n), transform=chip["transform"], fill=fill, dtype=dtype)


def rasterize_values(gdf, field: str, chip: dict, fill: int = 0) -> np.ndarray:
    """Rasteriza el valor entero de ``field`` de cada polígono sobre la grilla del chip."""
    from rasterio.features import rasterize

    n = chip["size"]
    geoms = clip_to_chip(gdf.geometry, chip) if len(gdf) else []
    shapes = [(g, int(v)) for g, v in zip(geoms, gdf[field]) if g is not None and not g.is_empty and v == v]
    if not shapes:
        return np.full((n, n), fill, dtype=np.int16)
    return rasterize(shapes, out_shape=(n, n), transform=chip["transform"], fill=fill, dtype="int16")


def reproject_to_chip(src_path, chip: dict, resampling: str = "bilinear", bands=None) -> np.ndarray:
    """Remuestrea un GeoTIFF a la grilla exacta del chip (EPSG:4326); fuera de cobertura queda NaN."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    n = chip["size"]
    with rasterio.open(src_path) as src:
        idx = bands or list(range(1, src.count + 1))
        out = np.full((len(idx), n, n), np.nan, dtype=np.float32)
        for k, b in enumerate(idx):
            reproject(source=rasterio.band(src, b), destination=out[k], src_transform=src.transform,
                      src_crs=src.crs, src_nodata=src.nodata, dst_transform=chip["transform"], dst_crs="EPSG:4326",
                      dst_nodata=np.nan, resampling=getattr(Resampling, resampling))
    return out
