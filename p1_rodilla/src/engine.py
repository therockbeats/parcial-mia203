"""Construcción de datasets y loaders, precisión mixta e inferencia."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data_io import split_path
from .dataset import KneeVolumeDataset
from .labels_common import to_targets
from .labels_rules import POS


def load_split(cfg: dict) -> pd.DataFrame:
    """Partición vigente (``data.split_csv``) restringida a los estudios con volumen preprocesado."""
    df = pd.read_csv(split_path(cfg), dtype={"study_id": str, "series_id": str})
    proc = Path(cfg["data"]["processed_dir"])
    ok = [(proc / f"{r.study_id}__{r.series_id}.npy").exists() for r in df.itertuples()]
    return df[ok].reset_index(drop=True)


def make_dataset(cfg: dict, df: pd.DataFrame, train: bool) -> KneeVolumeDataset:
    """Dataset con objetivos y pesos; en el conjunto dorado todas las etiquetas pesan 1."""
    names = cfg["labels"]["names"]
    if (df["split"] == "gold").all():
        y = (df[names].to_numpy() == POS).astype(np.float32)
        w = np.ones_like(y)
    else:
        y, w = to_targets(df, names, cfg["labels"])
    return KneeVolumeDataset(df, cfg["data"]["processed_dir"], y, w, cfg["model"]["slices_in"], train=train)


def make_loader(cfg: dict, ds: KneeVolumeDataset, train: bool) -> DataLoader:
    """DataLoader; con ``sampler: oversample_rare`` muestrea según la rareza de los positivos."""
    t = cfg["train"]
    sampler = None
    if train and t.get("sampler") == "oversample_rare":
        # Peso de muestreo: 1 + máximo de (positivo / prevalencia) entre los hallazgos del estudio
        y = ds.y.numpy()
        freq = y.mean(0) + 1e-3
        weights = 1.0 + (y / freq).max(1)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(ds), replacement=True)
    nw = int(t.get("num_workers", 4))
    return DataLoader(ds, batch_size=int(t["batch_size"]), shuffle=train and sampler is None, sampler=sampler,
                      num_workers=nw, pin_memory=torch.cuda.is_available(), drop_last=train,
                      persistent_workers=nw > 0)


def autocast_ctx(device: torch.device, enabled: bool):
    """Contexto de precisión mixta bf16 en CUDA; float32 en MPS y CPU."""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict(model, loader, device, amp: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probabilidades, objetivos y pesos de todos los lotes del loader."""
    model.eval()
    ps, ys, ws = [], [], []
    for x, y, w in loader:
        x = x.to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            logits = model(x)
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(y.numpy())
        ws.append(w.numpy())
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(ws)
