"""
Etapa 3. Tabla de chips y caché en .npy.

Convierte cada chip de Sen1Floods11 (dorados y débiles) a archivos .npy: Sentinel-1 en dB
(float16, NaN conservados), etiquetas en int8 (-1 ignorar, 0 no agua, 1 agua) y, para los chips
dorados, las bandas ópticas seleccionadas de Sentinel-2 junto con su fracción de nubes. Registra
por chip el evento, la partición, la fracción de agua y de píxeles sin dato.

Salidas:
    <processed_dir>/s1/<chip>.npy, <processed_dir>/s2/<chip>.npy, <processed_dir>/labels/<fuente>/<chip>.npy
    <interim_dir>/chips.parquet
    p2_inundaciones/reports/tablas/resumen_particiones.csv
    p2_inundaciones/reports/tablas/estadisticas_s1.csv

Uso:
    python -m p2_inundaciones.scripts.s03_preparar_chips [--workers 12]
    python -m p2_inundaciones.scripts.s03_preparar_chips --solo-estadisticas
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p2_inundaciones.src.io_s1f import (chip_id_from_name, event_from_chip, label_to_int8, raw_root, read_split_csv,
                                        read_tif, s1_to_unit)

HAND_LABELS = ["LabelHand", "S1OtsuLabelHand", "JRCWaterHand"]


def cloud_fraction(s2: np.ndarray, blue_thr: float, cirrus_thr: float) -> float:
    """Fracción de píxeles nubosos: azul (B2) brillante o cirros (B10) por encima del umbral."""
    valid = s2.sum(axis=0) > 0
    if not valid.any():
        return 1.0
    blue = s2[1].astype(np.float32) / 10000.0
    cirrus = s2[10].astype(np.float32) / 10000.0
    cloud = ((blue > blue_thr) | (cirrus > cirrus_thr)) & valid
    return float(cloud.sum() / valid.sum())


def train_stats(cfg: dict, chip_ids, proc: Path) -> pd.DataFrame:
    """Media y desviación estándar de VV y VH escalados a [0, 1] en los píxeles con dato del entrenamiento."""
    pp = cfg["preprocess"]
    vals = [[], []]
    for cid in chip_ids:
        db = np.load(proc / "s1" / f"{cid}.npy").astype(np.float32)
        ok = np.isfinite(db).all(axis=0)
        u = s1_to_unit(db, cfg)
        for b in range(2):
            vals[b].append(u[b][::2, ::2][ok[::2, ::2]])
    rows = []
    for b, name in enumerate(["VV", "VH"]):
        v = np.concatenate(vals[b]) if vals[b] else np.array([np.nan])
        lo, hi = float(pp["clip_min"]), float(pp["clip_max"])
        rows.append({"banda": name, "media_entrenamiento": float(np.mean(v)), "de_entrenamiento": float(np.std(v)),
                     "media_db": float(np.mean(v) * (hi - lo) + lo), "media_config": pp["mean"][b], "de_config": pp["std"][b]})
    return pd.DataFrame(rows)


def process_chip(task: dict) -> dict:
    """Convierte un chip a .npy y devuelve sus estadísticas."""
    import os

    proc = Path(task["processed"])
    cid = task["chip_id"]
    out = {"chip_id": cid}
    s1_out = proc / "s1" / f"{cid}.npy"
    if task["overwrite"] or not s1_out.exists():
        np.save(s1_out, read_tif(Path(task["s1"])).astype(np.float16))
    s1 = np.load(s1_out).astype(np.float32)
    has = np.isfinite(s1).all(axis=0)
    out["nodata_frac"] = float(1 - has.mean())
    for name, path in task["labels"].items():
        lab_out = proc / "labels" / name / f"{cid}.npy"
        if os.path.exists(path):
            if task["overwrite"] or not lab_out.exists():
                np.save(lab_out, label_to_int8(read_tif(Path(path))))
            lab = np.load(lab_out)
            out[f"has_{name}"] = True
            v = (lab >= 0) & has
            out[f"agua_{name}"] = float((lab[v] == 1).mean()) if v.any() else float("nan")
            out[f"valido_{name}"] = float(v.mean())
        else:
            out[f"has_{name}"] = False
    if task.get("s2"):
        s2_out = proc / "s2" / f"{cid}.npy"
        if os.path.exists(task["s2"]):
            s2 = read_tif(Path(task["s2"]))
            out["cloud_frac"] = cloud_fraction(s2, task["blue_thr"], task["cirrus_thr"])
            if task["overwrite"] or not s2_out.exists():
                np.save(s2_out, s2[task["s2_bands"]].astype(np.uint16))
            out["has_s2"] = True
        else:
            out["has_s2"] = False
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--solo-estadisticas", action="store_true", help="solo recalcular estadisticas_s1.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s03")
    if args.solo_estadisticas:
        df = pd.read_parquet(Path(cfg["data"]["interim_dir"]) / "chips.parquet")
        stats = train_stats(cfg, df.loc[(df.group == "hand") & (df.split == "train"), "chip_id"], Path(cfg["data"]["processed_dir"]))
        stats.to_csv(ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas") / "estadisticas_s1.csv", index=False)
        log.info("\n" + stats.round(4).to_string(index=False))
        return
    ds, pp = cfg["dataset"], cfg["preprocess"]
    root = raw_root(cfg)
    proc = Path(cfg["data"]["processed_dir"])
    for sub in ["s1", "s2"] + [f"labels/{n}" for n in HAND_LABELS + ds["weak_label_sources"]]:
        ensure_dir(proc / sub)
    interim = ensure_dir(cfg["data"]["interim_dir"])

    split_of = {}
    for k, v in ds["split_files"].items():
        f = root / ds["splits_dir"] / v
        if f.exists():
            for c in read_split_csv(f):
                split_of[c] = k
    hand, weak = root / ds["hand_dir"], root / ds["weak_dir"]
    common = {"processed": str(proc), "overwrite": args.overwrite, "s2_bands": list(pp["s2_bands"]),
              "blue_thr": float(cfg["fusion"]["blue_threshold"]), "cirrus_thr": float(cfg["fusion"]["cirrus_threshold"])}
    tasks, meta = [], {}
    for f in sorted((hand / "S1Hand").glob("*.tif")):
        cid = chip_id_from_name(f.name)
        tasks.append({**common, "chip_id": cid, "s1": str(f), "s2": str(hand / "S2Hand" / f"{cid}_S2Hand.tif"),
                      "labels": {n: str(hand / n / f"{cid}_{n}.tif") for n in HAND_LABELS}})
        meta[cid] = {"group": "hand", "split": split_of.get(cid, "sin_particion")}
    for f in sorted((weak / "S1Weak").glob("*.tif")):
        cid = chip_id_from_name(f.name)
        tasks.append({**common, "chip_id": cid, "s1": str(f), "s2": None,
                      "labels": {n: str(weak / n / f"{cid}_{n}.tif") for n in ds["weak_label_sources"]}})
        meta[cid] = {"group": "weak", "split": "weak"}
    log.info(f"Chips: {sum(m['group'] == 'hand' for m in meta.values())} dorados, "
             f"{sum(m['group'] == 'weak' for m in meta.values())} débiles")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers or int(pp.get("num_workers", 8))) as ex:
        futs = [ex.submit(process_chip, t) for t in tasks]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="chips"):
            rows.append(fu.result())
    df = pd.DataFrame(rows)
    df["event"] = df.chip_id.map(event_from_chip)
    df["group"] = df.chip_id.map(lambda c: meta[c]["group"])
    df["split"] = df.chip_id.map(lambda c: meta[c]["split"])
    for col in [c for c in df.columns if c.startswith("has_")]:
        df[col] = df[col].fillna(False).astype(bool)
    if "cloud_frac" not in df:
        df["cloud_frac"] = np.nan
    lima_old = interim / "chips.parquet"
    if lima_old.exists():
        prev = pd.read_parquet(lima_old)
        df = pd.concat([df, prev[prev.group == "lima"]], ignore_index=True)
    df.to_parquet(interim / "chips.parquet", index=False)

    hand_df = df[df.group == "hand"]
    summ = hand_df.groupby("split").agg(chips=("chip_id", "size"), agua_pct=("agua_LabelHand", "mean"),
                                        sin_dato_pct=("nodata_frac", "mean"), nubes_media=("cloud_frac", "mean"),
                                        libres_nubes=("cloud_frac", lambda s: int((s <= 0.1).sum())))
    weak_df = df[df.group == "weak"]
    summ.loc["weak"] = [len(weak_df), weak_df.get("agua_S1OtsuLabelWeak", pd.Series(dtype=float)).mean(),
                        weak_df.nodata_frac.mean(), np.nan, np.nan]
    summ[["agua_pct", "sin_dato_pct"]] *= 100
    tab = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")
    summ.round(3).to_csv(tab / "resumen_particiones.csv")

    stats = train_stats(cfg, hand_df.loc[hand_df.split == "train", "chip_id"], proc)
    stats.to_csv(tab / "estadisticas_s1.csv", index=False)
    log.info("\n" + summ.round(3).to_string())
    log.info("\n" + stats.round(4).to_string(index=False))
    log.info(f"Guardado {interim / 'chips.parquet'}")


if __name__ == "__main__":
    main()
