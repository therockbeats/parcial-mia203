"""Conjunto de chips para entrenamiento y evaluación a partir de la caché en .npy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .io_s1f import normalize_s1, normalize_s2


def s1_path(cfg: dict, chip_id: str) -> Path:
    return Path(cfg["data"]["processed_dir"]) / "s1" / f"{chip_id}.npy"


def s2_path(cfg: dict, chip_id: str) -> Path:
    return Path(cfg["data"]["processed_dir"]) / "s2" / f"{chip_id}.npy"


def label_path(cfg: dict, source: str, chip_id: str) -> Path:
    return Path(cfg["data"]["processed_dir"]) / "labels" / source / f"{chip_id}.npy"


def load_input(cfg: dict, chip_id: str, use_s2: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Entrada normalizada (C, H, W) y máscara de píxeles con dato SAR (H, W)."""
    db = np.load(s1_path(cfg, chip_id)).astype(np.float32)
    has_data = np.isfinite(db).all(axis=0)
    x = normalize_s1(db, cfg)
    if use_s2:
        x = np.concatenate([x, normalize_s2(np.load(s2_path(cfg, chip_id)), cfg)], axis=0)
    return x.astype(np.float32), has_data


def load_label(cfg: dict, source: str, chip_id: str) -> np.ndarray:
    return np.load(label_path(cfg, source, chip_id)).astype(np.int8)


class ChipDataset(Dataset):
    """
    Pares (entrada, objetivo, máscara) a partir de ``items`` = [(chip_id, fuente_de_etiqueta)].

    En entrenamiento se toma un recorte aleatorio de ``crop`` píxeles y se aplican volteos y
    rotaciones de 90 grados; en evaluación se devuelve el chip completo. La máscara vale 1 en los
    píxeles con etiqueta válida y dato SAR.
    """

    def __init__(self, cfg: dict, items: list[tuple[str, str]], train: bool, crop: int | None = None,
                 use_s2: bool = False):
        self.cfg, self.items, self.train = cfg, items, train
        self.crop = crop
        self.use_s2 = use_s2

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        chip_id, source = self.items[i]
        x, has_data = load_input(self.cfg, chip_id, self.use_s2)
        lab = load_label(self.cfg, source, chip_id)
        valid = (lab >= 0) & has_data
        y = (lab == 1).astype(np.float32)
        w = valid.astype(np.float32)
        if self.train:
            h, wd = y.shape
            c = min(self.crop or h, h, wd)
            r0 = np.random.randint(0, h - c + 1)
            c0 = np.random.randint(0, wd - c + 1)
            x, y, w = x[:, r0:r0 + c, c0:c0 + c], y[r0:r0 + c, c0:c0 + c], w[r0:r0 + c, c0:c0 + c]
            if np.random.rand() < 0.5:
                x, y, w = x[:, :, ::-1], y[:, ::-1], w[:, ::-1]
            if np.random.rand() < 0.5:
                x, y, w = x[:, ::-1, :], y[::-1, :], w[::-1, :]
            k = np.random.randint(4)
            x, y, w = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k), np.rot90(w, k)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(w)))
