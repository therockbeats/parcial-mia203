"""
Carga de configuración (archivos YAML) para todo el proyecto.

Idea general
------------
Cada pregunta tiene su archivo ``config.yaml`` con TODOS los parámetros
(rutas, columnas de los CSV, hiperparámetros...). Así no hay "números mágicos"
escondidos dentro del código y cada experimento queda documentado.

Las rutas pueden usar marcadores ``{data_root}`` y ``{runs_root}``. Esos valores
se leen de ``configs/paths.local.yaml``, que es distinto en la Mac y en Windows.
De esta forma el MISMO código corre en ambas máquinas sin cambiar nada.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# Carpeta raíz del repositorio: este archivo está en <repo>/common/config.py,
# así que subimos dos niveles (parents[1]).
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    """Lee un archivo YAML y lo devuelve como diccionario de Python."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_machine_paths() -> dict:
    """
    Lee las rutas propias de esta máquina.

    Orden de búsqueda:
      1. Variable de entorno MIA203_PATHS (ruta a un YAML), si existe.
      2. configs/paths.local.yaml dentro del repositorio.
    """
    env = os.environ.get("MIA203_PATHS")
    candidate = Path(env) if env else REPO_ROOT / "configs" / "paths.local.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"No encuentro {candidate}.\n"
            "Copia configs/paths.example.yaml como configs/paths.local.yaml "
            "y ajusta data_root y runs_root para esta máquina."
        )
    paths = load_yaml(candidate)
    for key in ("data_root", "runs_root"):
        if key not in paths:
            raise KeyError(f"Falta la clave '{key}' en {candidate}")
    return paths


def _fill_placeholders(obj: Any, values: dict) -> Any:
    """Reemplaza {data_root} y {runs_root} en todos los textos del config (recursivo)."""
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
    """Convierte el texto de un override ('0.001', 'true', '[1,2]') al tipo correcto usando YAML."""
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """
    Aplica cambios del tipo  seccion.clave=valor  desde la línea de comandos.

    Ejemplo:  --set train.loss=focal --set model.backbone=resnet18
    Esto permite lanzar ablaciones sin duplicar archivos de configuración.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override inválido '{item}'. Usa seccion.clave=valor")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _parse_value(raw)
    return cfg


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict:
    """Carga el config de una pregunta, aplica overrides y rellena las rutas de la máquina."""
    cfg = load_yaml(config_path)
    cfg = apply_overrides(cfg, overrides)
    paths = load_machine_paths()
    cfg = _fill_placeholders(cfg, {**paths, "repo_root": REPO_ROOT.as_posix()})
    cfg["_machine_paths"] = paths
    cfg["_config_path"] = str(Path(config_path).resolve())
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Guarda el config efectivo de un experimento (para reproducibilidad)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
