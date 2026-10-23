"""
Datos sintéticos con la estructura de Sen1Floods11 v1.1 y de la región de Lima.

Genera chips GeoTIFF de 512 x 512 píxeles en EPSG:4326 con los mismos tipos y codificaciones del
conjunto real, un GeoTIFF de Sentinel-1 que cubre la región de Lima y un GeoPackage con las capas
de la delineación EMSN038 que usa la etapa 6.

Uso:
    python tests/make_synthetic_p2.py --out <data_root>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PX = 8.9832e-05
N = 512


def blobs(rng, n=N, frac=0.25):
    from scipy import ndimage

    z = ndimage.gaussian_filter(rng.normal(size=(n, n)), 18)
    return z > np.quantile(z, 1 - frac)


def write(path: Path, arr: np.ndarray, dtype, x0: float, y0: float, nodata=None) -> None:
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    a = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=a.shape[2], height=a.shape[1], count=a.shape[0], dtype=dtype,
                       crs="EPSG:4326", transform=from_origin(x0, y0, PX, PX), nodata=nodata) as dst:
        dst.write(a.astype(dtype))


def s1_from_mask(rng, water):
    vv = np.where(water, -19.0, -8.0) + rng.normal(0, 2.0, water.shape)
    vh = np.where(water, -26.0, -15.0) + rng.normal(0, 2.0, water.shape)
    s1 = np.stack([vv, vh]).astype(np.float32)
    s1[:, :6, :] = np.nan
    return s1


def make_s1f(root: Path, rng) -> None:
    v = root / "v1.1"
    hand, weak = v / "data/flood_events/HandLabeled", v / "data/flood_events/WeaklyLabeled"
    splits = {"train": [], "valid": [], "test": [], "bolivia": []}
    k = 0
    for ev, counts in {"Ghana": (4, 2, 2), "India": (4, 2, 2), "Bolivia": (0, 0, 0)}.items():
        names = ["train"] * counts[0] + ["valid"] * counts[1] + ["test"] * counts[2]
        if ev == "Bolivia":
            names = ["bolivia"] * 3
        for sp in names:
            k += 1
            cid = f"{ev}_{1000 + k}"
            x0, y0 = -1.0 + k * 0.05, 9.0
            water = blobs(rng)
            write(hand / "S1Hand" / f"{cid}_S1Hand.tif", s1_from_mask(rng, water), "float32", x0, y0)
            lab = water.astype(np.int16)
            lab[:, :10] = -1
            write(hand / "LabelHand" / f"{cid}_LabelHand.tif", lab, "int16", x0, y0)
            otsu = (water ^ (rng.random(water.shape) < 0.05)).astype(np.int16)
            write(hand / "S1OtsuLabelHand" / f"{cid}_S1OtsuLabelHand.tif", otsu, "int16", x0, y0, nodata=-1)
            write(hand / "JRCWaterHand" / f"{cid}_JRCWaterHand.tif", (water & (rng.random(water.shape) < 0.3)).astype(np.uint8),
                  "uint8", x0, y0)
            s2 = np.full((13, N, N), 1500, np.int16)
            s2[7][water] = 300
            s2[11][water] = 200
            s2[1] = 900
            s2[10] = 10
            write(hand / "S2Hand" / f"{cid}_S2Hand.tif", s2, "int16", x0, y0, nodata=0)
            splits[sp].append(f"{cid}_S1Hand.tif,{cid}_LabelHand.tif")
    for i in range(12):
        ev = "Bolivia" if i < 2 else ("Ghana" if i % 2 else "India")
        cid = f"{ev}_{5000 + i}"
        x0, y0 = 5.0 + i * 0.05, 9.0
        water = blobs(rng)
        write(weak / "S1Weak" / f"{cid}_S1Weak.tif", s1_from_mask(rng, water), "float32", x0, y0)
        write(weak / "S1OtsuLabelWeak" / f"{cid}_S1OtsuLabelWeak.tif",
              (water ^ (rng.random(water.shape) < 0.08)).astype(np.int16), "int16", x0, y0, nodata=-1)
        s2l = (water ^ (rng.random(water.shape) < 0.1)).astype(np.int16)
        s2l[:20] = -1
        write(weak / "S2IndexLabelWeak" / f"{cid}_S2IndexLabelWeak.tif", s2l, "int16", x0, y0)
    sd = v / "splits/flood_handlabeled"
    sd.mkdir(parents=True, exist_ok=True)
    for sp, fname in {"train": "flood_train_data.csv", "valid": "flood_valid_data.csv", "test": "flood_test_data.csv",
                      "bolivia": "flood_bolivia_data.csv"}.items():
        (sd / fname).write_text("\n".join(splits[sp]) + "\n", encoding="utf-8")


def make_lima(root: Path, rng, bbox) -> None:
    import geopandas as gpd
    from shapely.geometry import Polygon, box

    x0, y0, x1, y1 = bbox
    step = 0.0005
    w, h = int((x1 - x0) / step) + 4, int((y1 - y0) / step) + 4
    from scipy import ndimage

    z = ndimage.gaussian_filter(rng.normal(size=(h, w)), 6)
    water = z > np.quantile(z, 0.95)
    vv = np.where(water, -18.0, -7.0) + rng.normal(0, 2, (h, w))
    vh = np.where(water, -25.0, -14.0) + rng.normal(0, 2, (h, w))
    import rasterio
    from rasterio.transform import from_origin

    out = root / "p2_lima" / "s1_snap"
    out.mkdir(parents=True, exist_ok=True)
    for name, shift in (("post", 0.0), ("pre", 1.5)):
        with rasterio.open(out / f"S1B_{name}_db.tif", "w", driver="GTiff", width=w, height=h, count=2, dtype="float32",
                           crs="EPSG:4326", transform=from_origin(x0 - 2 * step, y1 + 2 * step, step, step)) as dst:
            dst.write(np.stack([vv + shift, vh + shift]).astype(np.float32))
            dst.descriptions = ("Sigma0_VV_db", "Sigma0_VH_db")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    flood = gpd.GeoDataFrame({"source": ["river", "maximun height"]},
                             geometry=[box(cx - 0.05, cy - 0.01, cx + 0.05, cy + 0.01), box(cx - 0.02, cy + 0.03, cx + 0.03, cy + 0.06)],
                             crs=4326)
    aoi = gpd.GeoDataFrame({"Name": ["Map"]}, geometry=[box(x0 + 0.01, y0 + 0.01, x1 - 0.01, y1 - 0.01)], crs=4326)
    sea = gpd.GeoDataFrame({"NAM": ["Mar"]}, geometry=[box(x0 - 0.1, y0, x0 + 0.03, y1)], crs=4326)
    lake = gpd.GeoDataFrame({"NAM": ["Laguna"]}, geometry=[box(cx + 0.08, cy - 0.08, cx + 0.09, cy - 0.07)], crs=4326)
    river = gpd.GeoDataFrame({"NAM": ["Rio"]}, geometry=[Polygon([(cx - 0.12, cy - 0.003), (cx + 0.12, cy - 0.003),
                                                                   (cx + 0.12, cy + 0.003), (cx - 0.12, cy + 0.003)])], crs=4326)
    lulc = gpd.GeoDataFrame({"LULC": [111.0, 332.0, 211.0], "NAM": ["Continuous urban fabric", "Bare rocks", "Non-irrigated arable land"]},
                            geometry=[box(x0, y0, cx, y1), box(cx, y0, x1, cy), box(cx, cy, x1, y1)], crs=4326)
    gp = root / "p2_lima" / "ems" / "EMSN038_sintetico.gpkg"
    gp.parent.mkdir(parents=True, exist_ok=True)
    for layer, g in (("Potential_Maximum_flood", flood), ("EMSN038_AOI", aoi), ("CoastA", sea), ("LakeresA", lake),
                     ("WatrcrsA", river), ("LULC", lulc)):
        g.to_crs(32718).to_file(gp, layer=layer, driver="GPKG")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="data_root sintético")
    ap.add_argument("--bbox", type=float, nargs=4, default=[-77.16, -12.06, -76.84, -11.75])
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    root = Path(args.out)
    make_s1f(root / "p2_sen1floods11" / "raw", rng)
    make_lima(root, rng, args.bbox)
    print(f"Datos sintéticos de la Pregunta 2 en {root}")


if __name__ == "__main__":
    main()
