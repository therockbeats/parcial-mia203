"""Partición de chips, cargadores de datos, inferencia y evaluación por chip."""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import ChipDataset, load_input, load_label
from .io_s1f import load_chips
from .metrics import confusion

SPLIT_LABEL = {"train": "LabelHand", "val": "LabelHand", "test": "LabelHand", "bolivia": "LabelHand", "lima": "Lima"}


def filtered_chips(cfg: dict) -> pd.DataFrame:
    """Tabla de chips con el filtro de nubosidad de ``data_filter`` aplicado a los chips dorados."""
    df = load_chips(cfg)
    mc = (cfg.get("data_filter") or {}).get("max_cloud_fraction")
    if mc is not None:
        hand = df.group == "hand"
        df = df[~hand | (df.cloud_frac <= float(mc))]
    return df.reset_index(drop=True)


def split_items(cfg: dict, split: str) -> list[tuple[str, str]]:
    """Pares (chip, fuente de etiqueta) de una partición de evaluación."""
    df = filtered_chips(cfg)
    ids = df.loc[df.split == split, "chip_id"].tolist()
    return [(c, SPLIT_LABEL[split]) for c in ids]


def train_items(cfg: dict) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Pares de entrenamiento según ``train.supervision`` y pesos de muestreo por chip."""
    t, ds = cfg["train"], cfg["dataset"]
    df = filtered_chips(cfg)
    gold = [(c, "LabelHand") for c in df.loc[df.split == "train", "chip_id"]]
    weak_df = df[(df.group == "weak") & (df[f"has_{t['weak_source']}"])]
    weak_df = weak_df[~weak_df.event.isin(ds.get("weak_exclude_events", []))]
    weak = [(c, t["weak_source"]) for c in weak_df.chip_id]
    sup = t["supervision"]
    if sup == "gold":
        return gold, np.ones(len(gold))
    if sup == "weak":
        return weak, np.ones(len(weak))
    if sup == "gold_weak":
        f = float(t.get("gold_fraction", 0.5))
        w = np.concatenate([np.full(len(gold), f / max(len(gold), 1)), np.full(len(weak), (1 - f) / max(len(weak), 1))])
        return gold + weak, w
    raise ValueError(f"supervisión desconocida: {sup}")


def make_train_loader(cfg: dict, items, weights) -> DataLoader:
    t = cfg["train"]
    ds = ChipDataset(cfg, items, train=True, crop=int(t["crop"]), use_s2=bool(t.get("use_s2", False)))
    n = int(t["steps"]) * int(t["batch_size"])
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=n, replacement=True)
    nw = int(t.get("num_workers", 4))
    return DataLoader(ds, batch_size=int(t["batch_size"]), sampler=sampler, num_workers=nw, drop_last=True,
                      pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def predict_items(model, cfg: dict, items, device, amp: bool = True, batch_size: int = 4):
    """Probabilidades (N, H, W), etiquetas (N, H, W) y máscaras válidas para chips completos."""
    model.eval()
    use_s2 = bool(cfg["train"].get("use_s2", False))
    probs, labels, valids = [], [], []
    for k in range(0, len(items), batch_size):
        batch = items[k:k + batch_size]
        xs, labs, vals = [], [], []
        for chip_id, source in batch:
            x, has_data = load_input(cfg, chip_id, use_s2)
            lab = load_label(cfg, source, chip_id)
            xs.append(x)
            labs.append(lab)
            vals.append((lab >= 0) & has_data)
        xb = torch.from_numpy(np.stack(xs)).to(device)
        with autocast_ctx(device, amp):
            logits = model(xb)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(np.stack(labs))
        valids.append(np.stack(vals))
    if not probs:
        return np.zeros((0, 1, 1), np.float32), np.zeros((0, 1, 1), np.int8), np.zeros((0, 1, 1), bool)
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(valids)


def per_chip_table(chip_ids, probs, labels, valids, thr: float) -> pd.DataFrame:
    """Conteos de la matriz de confusión por chip con el umbral ``thr``."""
    rows = []
    for cid, p, t, v in zip(chip_ids, probs, labels, valids):
        c = confusion(p >= thr, t, v)
        rows.append({"chip_id": cid, **c})
    return pd.DataFrame(rows)
