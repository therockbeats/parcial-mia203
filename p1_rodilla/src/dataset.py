"""
Dataset de volúmenes con representación 2.5D.

Cada corte seleccionado se representa con sus vecinos ``(i-1, i, i+1)`` como tres
canales, lo que aporta contexto en profundidad y es compatible con codificadores
preentrenados de tres canales.

Cada elemento es ``(x, y, w)``:
    x : (S, 3, H, W)  cortes en representación 2.5D
    y : (K,)          objetivo binario por hallazgo
    w : (K,)          peso de cada hallazgo en la pérdida (0 = excluido)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class KneeVolumeDataset(Dataset):
    """Volúmenes preprocesados ``<study_id>__<series_id>.npy`` con objetivos y pesos."""

    def __init__(self, df, processed_dir: str | Path, y: np.ndarray, w: np.ndarray,
                 slices_in: int = 16, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.w = torch.as_tensor(w, dtype=torch.float32)
        self.slices_in = slices_in
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, i: int) -> np.ndarray:
        r = self.df.iloc[i]
        return np.load(self.processed_dir / f"{r.study_id}__{r.series_id}.npy")

    def _choose_slices(self, n_total: int) -> np.ndarray:
        """Ventana de ``slices_in`` cortes: centrada en evaluación, aleatoria en entrenamiento."""
        k = min(self.slices_in, n_total)
        start = (n_total - k) // 2
        if self.train and n_total > k:
            start = np.random.randint(0, n_total - k + 1)
        return np.arange(start, start + k)

    @staticmethod
    def _augment(vol: torch.Tensor) -> torch.Tensor:
        """
        Transformación afín y de intensidad común a todos los cortes del volumen.

        Rotación de ±10°, escala ±10 %, traslación ±5 %, contraste ×[0.85, 1.15] y brillo
        ±0.08. Se excluye el volteo anteroposterior por alterar la anatomía.
        """
        import torchvision.transforms.v2.functional as TF

        angle = float(np.random.uniform(-10, 10))
        scale = float(np.random.uniform(0.9, 1.1))
        h = vol.shape[-1]
        tx, ty = [int(np.random.uniform(-0.05, 0.05) * h) for _ in range(2)]
        vol = TF.affine(vol, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0])
        contrast = float(np.random.uniform(0.85, 1.15))
        brightness = float(np.random.uniform(-0.08, 0.08))
        return (vol * contrast + brightness).clamp(0, 1)

    def __getitem__(self, i: int):
        vol = self._load(i)
        idx = self._choose_slices(vol.shape[0])
        v = torch.from_numpy(vol).float() / 255.0
        if self.train:
            v = self._augment(v)
        # Vecinos (i-1, i, i+1) con replicación en los bordes
        prev_idx = np.clip(idx - 1, 0, vol.shape[0] - 1)
        next_idx = np.clip(idx + 1, 0, vol.shape[0] - 1)
        x = torch.stack([v[prev_idx], v[idx], v[next_idx]], dim=1)   # (S, 3, H, W)
        # Normalización con media y desviación aproximadas de ImageNet en escala de grises
        x = (x - 0.45) / 0.225
        return x, self.y[i], self.w[i]
