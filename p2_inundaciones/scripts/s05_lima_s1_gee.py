"""
Etapa 5. Chips de Sentinel-1 de Lima (marzo de 2017) desde Google Earth Engine.

Usa la colección ``COPERNICUS/S1_GRD`` de Earth Engine, la misma fuente y el mismo
preprocesamiento de Sen1Floods11 (eliminación de ruido térmico, calibración radiométrica,
corrección del terreno con SRTM y conversión a dB). Para cada escena de ``lima.scenes``
(posterior y previa al evento) descarga, sobre la grilla de chips de 512 x 512 píxeles a 10 m en
EPSG:4326, las bandas VV, VH y el ángulo de incidencia; descarga además la elevación SRTM.

Salidas:
    <lima_dir>/s1_gee/<escena>/<chip>.tif, <lima_dir>/s1_gee/dem/<chip>.tif
    <processed_dir>/s1/<chip>.npy                 (escena posterior: VV, VH en dB)
    <processed_dir>/lima_aux/<chip>_pre.npy       (escena previa: VV, VH en dB)
    <processed_dir>/lima_aux/<chip>_angle.npy, <chip>_dem.npy
    <interim_dir>/lima_escenas.csv

Uso:
    python -m p2_inundaciones.scripts.s05_lima_s1_gee --listar
    python -m p2_inundaciones.scripts.s05_lima_s1_gee [--proyecto mi-proyecto-gee]
"""
from __future__ import annotations

import argparse
import datetime as dt
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p2_inundaciones.src.geo import lima_grid, reproject_to_chip

NODATA = -9999.0


def read_chip(path: Path, chip: dict, resampling: str) -> np.ndarray:
    """GeoTIFF descargado remuestreado a la grilla del chip, con los píxeles sin dato como NaN."""
    arr = reproject_to_chip(path, chip, resampling=resampling)
    arr[arr <= NODATA + 1] = np.nan
    return arr


def s1_collection(ee, cfg: dict, start: str, end: str):
    lm = cfg["lima"]
    aoi = ee.Geometry.Rectangle(lm["bbox"])
    return (ee.ImageCollection(lm["s1_collection"]).filterBounds(aoi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def describe(ee, col) -> pd.DataFrame:
    """Identificador, plataforma, órbita relativa, paso y fecha de cada imagen de la colección."""
    props = {"imagen": "system:index", "plataforma": "platform_number", "orbita_relativa": "relativeOrbitNumber_start",
             "paso": "orbitProperties_pass", "t": "system:time_start"}
    data = {k: col.aggregate_array(v).getInfo() for k, v in props.items()}
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["plataforma"] = "S1" + df["plataforma"].astype(str)
    df["fecha_utc"] = pd.to_datetime(df.pop("t"), unit="ms").dt.strftime("%Y-%m-%d %H:%M")
    return df.sort_values("fecha_utc").reset_index(drop=True)


def fetch(image, chip: dict, bands: list[str], dest: Path, retries: int = 3) -> None:
    """Descarga los ``bands`` de ``image`` sobre la grilla del chip como GeoTIFF."""
    import ee

    if dest.exists():
        return
    left, bottom, right, top = chip["bounds"]
    px = chip["transform"].a
    pad = 2 * px
    region = ee.Geometry.Rectangle([left - pad, bottom - pad, right + pad, top + pad], "EPSG:4326", False)
    url = image.select(bands).unmask(NODATA).getDownloadURL({"region": region, "crs": "EPSG:4326",
                                              "crs_transform": [px, 0, left, 0, -px, top], "format": "GEO_TIFF"})
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                data = r.read()
            dest.write_bytes(data)
            return
        except Exception:
            if k == retries - 1:
                raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--proyecto", default=None, help="proyecto de Google Cloud registrado en Earth Engine")
    ap.add_argument("--listar", action="store_true", help="solo listar las escenas de febrero a abril de 2017")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s05")
    lm = cfg["lima"]
    import ee

    ee.Initialize(project=args.proyecto or lm.get("gee_project") or None)
    interim = ensure_dir(cfg["data"]["interim_dir"])

    if args.listar:
        tab = describe(ee, s1_collection(ee, cfg, "2017-02-01", "2017-05-01"))
        tab.to_csv(interim / "lima_escenas_disponibles.csv", index=False)
        log.info("\n" + tab.to_string(index=False))
        return

    chips = lima_grid(cfg)
    lima_dir, proc = Path(cfg["data"]["lima_dir"]), Path(cfg["data"]["processed_dir"])
    ensure_dir(proc / "s1")
    aux = ensure_dir(proc / "lima_aux")
    used = []
    for key, date in lm["scenes"].items():
        d0 = dt.date.fromisoformat(date)
        col = (s1_collection(ee, cfg, str(d0), str(d0 + dt.timedelta(days=1)))
               .filter(ee.Filter.eq("relativeOrbitNumber_start", int(lm["relative_orbit"])))
               .filter(ee.Filter.eq("orbitProperties_pass", lm["orbit_pass"])))
        tab = describe(ee, col)
        if tab.empty:
            raise RuntimeError(f"No hay escena S1 para {date} en la órbita {lm['relative_orbit']}")
        tab["escena"] = key
        used.append(tab)
        log.info(f"Escena {key} ({date}): {', '.join(tab.imagen)}")
        image = col.mosaic()
        out = ensure_dir(lima_dir / "s1_gee" / key)
        for chip in tqdm(chips, desc=f"S1 {key}"):
            f = out / f"{chip['chip_id']}.tif"
            fetch(image, chip, ["VV", "VH", "angle"], f)
            arr = read_chip(f, chip, "nearest")
            if key == "post":
                np.save(proc / "s1" / f"{chip['chip_id']}.npy", arr[:2].astype(np.float16))
                np.save(aux / f"{chip['chip_id']}_angle.npy", arr[2].astype(np.float16))
            else:
                np.save(aux / f"{chip['chip_id']}_{key}.npy", arr[:2].astype(np.float16))

    dem = ee.Image("USGS/SRTMGL1_003")
    out = ensure_dir(lima_dir / "s1_gee" / "dem")
    for chip in tqdm(chips, desc="SRTM"):
        f = out / f"{chip['chip_id']}.tif"
        fetch(dem, chip, ["elevation"], f)
        np.save(aux / f"{chip['chip_id']}_dem.npy", read_chip(f, chip, "nearest")[0].astype(np.float32))
    pd.concat(used).to_csv(interim / "lima_escenas.csv", index=False)
    log.info(f"{len(chips)} chips de Lima en {proc / 's1'}")


if __name__ == "__main__":
    main()
