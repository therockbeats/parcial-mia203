"""Corridas por semilla de una configuración y ensamble de sus probabilidades."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import load_label, s1_path
from .engine import SPLIT_LABEL, per_chip_table
from .metrics import best_threshold


def run_name(name: str, seed: int, base_seed: int) -> str:
    return name if seed == base_seed else f"{name}_s{seed}"


def run_dirs(cfg: dict, name: str, seeds: list[int]) -> list[Path]:
    """Carpetas de las corridas evaluadas de ``name`` (una por semilla)."""
    base = int(cfg.get("seed", 42))
    runs = Path(cfg["data"]["runs_dir"])
    out = [runs / run_name(name, s, base) for s in seeds]
    return [d for d in out if (d / "summary.json").exists()]


def load_preds(run: Path, split: str):
    f = run / f"preds_{split}.npz"
    if not f.exists():
        return None, None
    z = np.load(f)
    return list(z["chip_ids"]), z["probs"].astype(np.float32) / 255.0


def labels_for(cfg: dict, chip_ids, split: str):
    """Etiquetas y máscaras válidas (etiqueta definida y dato SAR) de los chips."""
    labs, vals = [], []
    for cid in chip_ids:
        lab = load_label(cfg, SPLIT_LABEL[split], cid)
        has = np.isfinite(np.load(s1_path(cfg, cid)).astype(np.float32)).all(axis=0)
        labs.append(lab)
        vals.append((lab >= 0) & has)
    return np.stack(labs), np.stack(vals)


def ensemble_split(runs: list[Path], split: str):
    """Promedio de probabilidades de las corridas para una partición."""
    ids, acc, n = None, None, 0
    for r in runs:
        c, p = load_preds(r, split)
        if c is None:
            continue
        if ids is None:
            ids, acc = c, p.copy()
        else:
            if c != ids:
                raise ValueError(f"Chips distintos entre corridas en {split}")
            acc += p
        n += 1
    return (ids, acc / n) if n else (None, None)


def ensemble_tables(cfg: dict, runs: list[Path], splits: list[str]) -> tuple[float, dict]:
    """Umbral del ensamble (validación) y tablas por chip de cada partición."""
    ids, pv = ensemble_split(runs, "val")
    lv, vv = labels_for(cfg, ids, "val")
    thr = best_threshold(pv, lv, vv) if cfg["eval"].get("threshold", "val") == "val" else float(cfg["eval"]["threshold"])
    out = {}
    for split in splits:
        ids, p = ensemble_split(runs, split)
        if ids is None:
            continue
        lab, val = labels_for(cfg, ids, split)
        out[split] = per_chip_table(ids, p, lab, val, thr)
    return thr, out


def run_metrics(run: Path) -> dict:
    """IoU por partición de una corrida (desde metricas.csv) y su paso final."""
    m = pd.read_csv(run / "metricas.csv").set_index("particion")
    s = json.loads((run / "summary.json").read_text())
    out = {"corrida": run.name, "umbral": s["umbral"], "paso": s["paso"]}
    for split, r in m.iterrows():
        out[f"iou_{split}"] = r["iou"]
    return out
