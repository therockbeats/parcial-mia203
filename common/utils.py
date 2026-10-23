"""Semillas, selección de dispositivo, registro y utilidades de archivos."""
from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fija las semillas de Python, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str = "auto"):
    """Devuelve el dispositivo de cómputo; con ``prefer="auto"`` prioriza CUDA, luego MPS y CPU."""
    import torch

    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Logger con salida a consola y, opcionalmente, a archivo."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class Timer:
    """Context manager que registra el tiempo transcurrido de un bloque."""

    def __init__(self, label: str, logger: logging.Logger | None = None):
        self.label, self.logger = label, logger

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        msg = f"{self.label}: {dt/60:.1f} min" if dt > 120 else f"{self.label}: {dt:.1f} s"
        (self.logger.info if self.logger else print)(msg)


def ensure_dir(path: str | Path) -> Path:
    """Crea el directorio si no existe y lo devuelve como ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
