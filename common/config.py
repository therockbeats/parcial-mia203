"""
Carga y composición de la configuración del proyecto.

Cada pregunta define sus parámetros en un ``config.yaml`` (rutas, esquema de datos,
hiperparámetros). Las rutas admiten los marcadores ``{data_root}``, ``{runs_root}`` y
``{repo_root}``; los dos primeros se resuelven con ``configs/paths.local.yaml``
(o con el archivo indicado en la variable de entorno ``MIA203_PATHS``).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    """Lee un archivo YAML y lo devuelve como diccionario."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_machine_paths() -> dict:
    """
    Rutas de datos y resultados de la máquina actual.

    Se lee el YAML indicado en ``MIA203_PATHS`` o, en su defecto,
    ``configs/paths.local.yaml``. Debe definir ``data_root`` y ``runs_root``.
    """
    env = os.environ.get("MIA203_PATHS")
    candidate = Path(env) if env else REPO_ROOT / "configs" / "paths.local.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"No se encontró {candidate}. "
            "Crear configs/paths.local.yaml a partir de configs/paths.example.yaml."
        )
    paths = load_yaml(candidate)
    for key in ("data_root", "runs_root"):
        if key not in paths:
            raise KeyError(f"Falta la clave '{key}' en {candidate}")
    return paths


def _fill_placeholders(obj: Any, values: dict) -> Any:
    """Sustituye recursivamente los marcadores ``{clave}`` en las cadenas del config."""
    if isinstance(obj, dict):
        return {k: _fill_placeholders(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill_placeholders(v, values) for v in obj]
    if isinstance(obj, str):
        for key, val in values.items():
            obj = obj.replace("{" + key + "}", str(val))
        return obj
    return obj


def _parse_value(text: str) -> Any:
    """Interpreta el valor de un override con la sintaxis de YAML (números, booleanos, listas)."""
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """
    Aplica overrides de la forma ``seccion.clave=valor``.

    Ejemplo: ``["train.loss=focal", "model.backbone=resnet18"]``.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override inválido '{item}' (formato esperado: seccion.clave=valor)")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _parse_value(raw)
    return cfg


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict:
    """Carga un config, aplica overrides y resuelve las rutas de la máquina."""
    cfg = load_yaml(config_path)
    cfg = apply_overrides(cfg, overrides)
    paths = load_machine_paths()
    cfg = _fill_placeholders(cfg, {**paths, "repo_root": REPO_ROOT.as_posix()})
    cfg["_machine_paths"] = paths
    cfg["_config_path"] = str(Path(config_path).resolve())
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Guarda la configuración efectiva de un experimento."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
