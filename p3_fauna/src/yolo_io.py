"""Formato YOLO: rutas de etiquetas, lectura de cajas, archivos de datos y dispositivo de Ultralytics."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import yaml


def label_path(image_path: str | Path) -> Path:
    """Misma regla que Ultralytics: el último directorio 'images' se reemplaza por 'labels'."""
    parts = list(Path(image_path).parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(image_path: str | Path) -> np.ndarray:
    """Filas (clase, xc, yc, w, h) normalizadas; matriz vacía si la imagen es de fondo."""
    p = label_path(image_path)
    if not p.exists():
        return np.zeros((0, 5))
    return np.loadtxt(p, ndmin=2)


def labels_xyxy(image_path: str | Path, width: int, height: int) -> np.ndarray:
    """Filas (clase, x1, y1, x2, y2) en píxeles."""
    g = read_labels(image_path)
    xc, yc, w, h = g[:, 1] * width, g[:, 2] * height, g[:, 3] * width, g[:, 4] * height
    return np.stack([g[:, 0], xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1)


def link_file(src: Path, dst: Path, mode: str = "hardlink") -> None:
    """Enlaza la imagen original sin duplicar espacio; si el sistema no lo permite, la copia."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    order = {"hardlink": ["hardlink", "symlink", "copy"], "symlink": ["symlink", "hardlink", "copy"]}.get(mode, ["copy"])
    for m in order:
        try:
            if m == "hardlink":
                os.link(src, dst)
            elif m == "symlink":
                os.symlink(Path(src).resolve(), dst)
            else:
                shutil.copy2(src, dst)
            return
        except OSError:
            continue


def write_data_yaml(path: Path, root: Path, train: str | Path, val: str | Path, names: list[str] | dict) -> Path:
    names = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
    data = {"path": Path(root).as_posix(), "train": Path(train).as_posix(), "val": Path(val).as_posix(),
            "names": names}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def class_names_from_yolo(yolo_dir: str | Path) -> list[str]:
    names = yaml.safe_load((Path(yolo_dir) / "names.yaml").read_text(encoding="utf-8"))
    return [names[i] for i in sorted(names)]


def ultralytics_device(prefer: str | int = "auto"):
    """Traduce 'auto' al formato de Ultralytics: 0 (CUDA), 'mps' o 'cpu'."""
    if str(prefer) != "auto":
        return prefer
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
