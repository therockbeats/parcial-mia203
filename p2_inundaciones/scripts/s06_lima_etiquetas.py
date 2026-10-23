"""
Etapa 6. Etiquetas de referencia de Lima a partir de la delineación Copernicus EMS EMSN038.

Rasteriza sobre la grilla de chips (EPSG:4326, 10 m) los polígonos de inundación de
``lima.water_layers``. La etiqueta vale 1 dentro de esos polígonos; -1 (ignorar) fuera del área
de interés de la activación y en el mar, lagunas y cauces no delineados (``lima.ignore_layers``),
porque la activación no los evalúa; y 0 en el resto. Se guardan además, por píxel, el origen del
polígono de inundación (cauce o área inundada) y la clase de cobertura del suelo, que se usan en el
análisis de generalización. Se conservan los chips con al menos ``min_valid_fraction`` de píxeles
válidos.

Salidas:
    <processed_dir>/labels/Lima/<chip>.npy
    <processed_dir>/lima_aux/<chip>_fuente.npy, <chip>_lulc.npy
    <interim_dir>/chips.parquet (filas del grupo "lima")
    p2_inundaciones/reports/tablas/lima_capas.csv, lima_etiquetas.csv
    p2_inundaciones/reports/figuras/lima_etiquetas.png

Uso:
    python -m p2_inundaciones.scripts.s06_lima_etiquetas [--listar-capas]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p2_inundaciones.src.geo import lima_grid, rasterize_on_chip, rasterize_values


def read_layer(path: str, layer: str):
    """Capa del geodatabase en EPSG:4326 y en dos dimensiones."""
    import geopandas as gpd
    import shapely

    g = gpd.read_file(path, layer=layer)
    g = g[g.geometry.notna()].copy()
    g["geometry"] = shapely.force_2d(g.geometry.values)
    return g.to_crs(4326)


def subset(gdf, chip: dict):
    """Geometrías que intersecan los límites del chip."""
    left, bottom, right, top = chip["bounds"]
    idx = list(gdf.sindex.query(__import__("shapely").box(left, bottom, right, top)))
    return gdf.iloc[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--listar-capas", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s06")
    lm = cfg["lima"]
    import pyogrio

    gdb = lm["gdb"]
    if not Path(gdb).exists():
        raise FileNotFoundError(f"No existe {gdb}: descomprimir EMSN038_Geodatabase.zip en {Path(gdb).parent}")
    layers = pd.DataFrame(pyogrio.list_layers(gdb), columns=["capa", "geometria"])
    tab_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")
    layers.to_csv(tab_dir / "lima_capas.csv", index=False)
    if args.listar_capas:
        log.info("\n" + layers.to_string(index=False))
        return

    import geopandas as gpd

    water = gpd.GeoDataFrame(pd.concat([read_layer(gdb, l) for l in lm["water_layers"]], ignore_index=True), crs=4326)
    field = lm.get("water_source_field")
    sources = sorted(water[field].dropna().astype(str).unique()) if field and field in water else []
    src_code = {s: i + 1 for i, s in enumerate(sources)}
    log.info(f"Polígonos de inundación: {len(water)} | orígenes: {src_code}")
    ignore = [read_layer(gdb, l) for l in lm.get("ignore_layers", [])]
    aoi = read_layer(gdb, lm["valid_layer"]) if lm.get("valid_layer") else None
    lulc = read_layer(gdb, lm["lulc_layer"]) if lm.get("lulc_layer") else None
    lulc_names = (lulc.groupby(lm["lulc_field"])["NAM"].first().to_dict()
                  if lulc is not None and "NAM" in lulc.columns else {})

    proc = Path(cfg["data"]["processed_dir"])
    lab_dir = ensure_dir(proc / "labels" / "Lima")
    aux = ensure_dir(proc / "lima_aux")
    rows = []
    for chip in lima_grid(cfg):
        cid = chip["chip_id"]
        s1f = proc / "s1" / f"{cid}.npy"
        if not s1f.exists():
            continue
        has = np.isfinite(np.load(s1f).astype(np.float32)).all(axis=0)
        wsub = subset(water, chip)
        lab = np.zeros((chip["size"],) * 2, dtype=np.int8)
        if aoi is not None:
            lab[rasterize_on_chip(subset(aoi, chip).geometry, chip) == 0] = -1
        for g in ignore:
            lab[rasterize_on_chip(subset(g, chip).geometry, chip) == 1] = -1
        fuente = np.zeros_like(lab)
        for s, code in src_code.items():
            m = rasterize_on_chip(wsub[wsub[field].astype(str) == s].geometry, chip) == 1
            fuente[m] = code
        wmask = rasterize_on_chip(wsub.geometry, chip) == 1
        lab[wmask] = 1
        valid = (lab >= 0) & has
        if valid.mean() < float(lm.get("min_valid_fraction", 0.5)):
            continue
        np.save(lab_dir / f"{cid}.npy", lab)
        np.save(aux / f"{cid}_fuente.npy", fuente)
        if lulc is not None:
            np.save(aux / f"{cid}_lulc.npy", rasterize_values(subset(lulc, chip), lm["lulc_field"], chip).astype(np.int16))
        rows.append({"chip_id": cid, "event": "Lima", "group": "lima", "split": "lima", "has_Lima": True,
                     "agua_Lima": float((lab[valid] == 1).mean()), "valido_Lima": float(valid.mean()),
                     "nodata_frac": float(1 - has.mean()), "row": chip["row"], "col": chip["col"],
                     **{f"pix_{s}": int(((fuente == c) & valid).sum()) for s, c in src_code.items()}})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Ningún chip de Lima con S1 y etiquetas válidas: ejecutar antes la etapa 5")
    interim = ensure_dir(cfg["data"]["interim_dir"])
    cp = interim / "chips.parquet"
    if cp.exists():
        prev = pd.read_parquet(cp)
        df = pd.concat([prev[prev.group != "lima"], df], ignore_index=True)
    df.to_parquet(cp, index=False)
    lima = df[df.group == "lima"]
    pd.DataFrame({"origen": list(src_code) + ["codigos_lulc"], "codigo": list(src_code.values()) + [str(lulc_names)]}).to_csv(
        tab_dir / "lima_codigos.csv", index=False)
    lima.drop(columns=["group", "split", "event", "has_Lima"]).to_csv(tab_dir / "lima_etiquetas.csv", index=False)
    log.info(f"Chips de Lima: {len(lima)} | con agua: {(lima.agua_Lima > 0).sum()} | "
             f"agua media: {100 * lima.agua_Lima.mean():.2f} %")

    nr, nc = int(lima.row.max()) + 1, int(lima.col.max()) + 1
    step = 4
    n = int(cfg["lima"]["chip_size"]) // step
    vv = np.full((nr * n, nc * n), np.nan, np.float32)
    lb = np.full((nr * n, nc * n), -1, np.int8)
    for r in lima.itertuples():
        i, j = int(r.row), int(r.col)
        s1 = np.load(proc / "s1" / f"{r.chip_id}.npy").astype(np.float32)[0, ::step, ::step]
        vv[i * n:(i + 1) * n, j * n:(j + 1) * n] = s1
        lb[i * n:(i + 1) * n, j * n:(j + 1) * n] = np.load(lab_dir / f"{r.chip_id}.npy")[::step, ::step]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(vv, cmap="gray", vmin=-25, vmax=0)
    ax[0].set_title("Sentinel-1 VV (dB), escena posterior")
    ax[1].imshow(vv, cmap="gray", vmin=-25, vmax=0)
    over = np.zeros(lb.shape + (4,))
    over[lb == 1] = [1, 0.1, 0.1, 0.8]
    over[lb == -1] = [0.2, 0.4, 1, 0.35]
    ax[1].imshow(over)
    ax[1].set_title("Inundación EMSN038 (rojo) y zonas ignoradas (azul)")
    for a in ax:
        a.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ensure_dir(Path(cfg["data"]["reports_dir"]) / "figuras") / "lima_etiquetas.png", dpi=150)
    log.info(f"Guardado {cp}")


if __name__ == "__main__":
    main()
