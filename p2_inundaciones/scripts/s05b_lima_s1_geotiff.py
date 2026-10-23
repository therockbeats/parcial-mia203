"""
Etapa 5 (alternativa). Chips de Sentinel-1 de Lima a partir de GeoTIFF propios.

Para cuando la escena se procesa fuera de Earth Engine, por ejemplo con el grafo de SNAP
``p2_inundaciones/snap/s1_grd_db.xml`` (órbita, ruido térmico, calibración sigma0, corrección
del terreno a 10 m en WGS84 y dB). Cada GeoTIFF debe tener las bandas VV y VH en dB; se remuestrea
a la grilla de chips de Lima.

Salidas: las mismas que la etapa 5 (sin ángulo de incidencia ni elevación).

Uso:
    python -m p2_inundaciones.scripts.s05b_lima_s1_geotiff --post D:/MIA203/data/p2_lima/s1_snap/S1B_20170317_db.tif
        [--pre D:/MIA203/data/p2_lima/s1_snap/S1B_20170305_db.tif] [--bandas 2 1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p2_inundaciones.src.geo import lima_grid, reproject_to_chip


def band_indices(path: str, given) -> list[int]:
    """Índices (base 1) de VV y VH, a partir de ``given`` o de las descripciones de las bandas."""
    if given:
        return list(given)
    import rasterio

    with rasterio.open(path) as r:
        desc = [(d or "").upper() for d in r.descriptions]
    try:
        return [next(i + 1 for i, d in enumerate(desc) if "VV" in d), next(i + 1 for i, d in enumerate(desc) if "VH" in d)]
    except StopIteration:
        raise ValueError(f"No se identifican las bandas VV y VH en {path}; indicar --bandas") from None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--post", required=True)
    ap.add_argument("--pre", default=None)
    ap.add_argument("--bandas", type=int, nargs=2, default=None,
                    help="índices (base 1) de VV y VH; por defecto se deducen de las descripciones de banda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s05b")
    proc = Path(cfg["data"]["processed_dir"])
    ensure_dir(proc / "s1")
    aux = ensure_dir(proc / "lima_aux")
    chips = lima_grid(cfg)
    for key, path in (("post", args.post), ("pre", args.pre)):
        if not path:
            continue
        idx = band_indices(path, args.bandas)
        log.info(f"{key}: VV = banda {idx[0]}, VH = banda {idx[1]}")
        for chip in tqdm(chips, desc=key):
            arr = reproject_to_chip(path, chip, resampling="bilinear", bands=idx).astype(np.float16)
            if key == "post":
                np.save(proc / "s1" / f"{chip['chip_id']}.npy", arr)
            else:
                np.save(aux / f"{chip['chip_id']}_pre.npy", arr)
    log.info(f"{len(chips)} chips de Lima en {proc / 's1'}")


if __name__ == "__main__":
    main()
