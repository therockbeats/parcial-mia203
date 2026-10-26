"""Ubicación, configuración y selección de experimentos."""
from __future__ import annotations

import json
from pathlib import Path

from common.config import load_config

PATH_KEYS = ("raw_dir", "yolo_dir", "runs_dir", "reports_dir")


def exp_dir(cfg: dict, exp: str) -> Path:
    return Path(cfg["data"]["runs_dir"]) / exp


def load_exp_config(base_cfg: dict, exp: str, overrides: list[str] | None = None) -> dict:
    """Config efectiva guardada por la etapa 3, con las rutas de la máquina actual."""
    path = exp_dir(base_cfg, exp) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} no existe: entrenar primero el experimento {exp} (etapa 3)")
    cfg = load_config(path, overrides)
    cfg["data"] = {**cfg["data"], **{k: base_cfg["data"][k] for k in PATH_KEYS}}
    return cfg


def weights(cfg: dict, exp: str) -> Path:
    return exp_dir(cfg, exp) / "weights" / "best.pt"


def read_summary(cfg: dict, exp: str) -> dict | None:
    p = exp_dir(cfg, exp) / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def all_experiments(cfg: dict, extras: bool = True) -> list[dict]:
    """Experimentos exigidos y, si ``extras``, también los adicionales."""
    ab = cfg["ablation"]
    return list(ab["experiments"]) + (list(ab.get("extra_experiments") or []) if extras else [])


def best_experiment(cfg: dict) -> tuple[str | None, float]:
    """Mejor experimento evaluado según mAP@0.5:0.95 en la partición de selección (cis_val)."""
    key = f"map50_95_{cfg['eval']['select_split']}"
    best, best_v = None, -1.0
    for e in all_experiments(cfg):
        s = read_summary(cfg, e["name"])
        v = (s or {}).get(key, float("nan"))
        if v == v and v > best_v:
            best, best_v = e["name"], v
    return best, best_v
