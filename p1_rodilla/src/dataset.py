"""
Dataset de PyTorch para la Pregunta 1 (enfoque 2.5D).

Qué es 2.5D
-----------
Un volumen de MRI tiene S cortes 2D. Una CNN 3D completa es cara y hay pocos datos.
El enfoque 2.5D pasa CADA corte por una CNN 2D, pero en vez de darle un solo corte
le damos 3 cortes vecinos (i-1, i, i+1) como si fueran los canales R, G, B de una foto.
Así la red ve algo de contexto en profundidad y además puede reutilizar pesos
preentrenados en ImageNet (que esperan exactamente 3 canales).

Cada elemento que devuelve el Dataset:
    x : tensor (S, 3, H, W)  -> S "imágenes" de 3 canales
    y : tensor (12,)         -> objetivo 0/1 por hallazgo
    w : tensor (12,)         -> peso de cada hallazgo en la pérdida (0 = ignorar)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class KneeVolumeDataset(Dataset):
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
        return np.load(self.processed_dir / f"{r.study_id}__{r.series_id}.npy")  # (S_total, H, W) uint8

    def _choose_slices(self, n_total: int) -> np.ndarray:
        """Elige 'slices_in' cortes centrales; en entrenamiento desplaza la ventana al azar (aumento de datos)."""
        k = min(self.slices_in, n_total)
        start = (n_total - k) // 2
        if self.train and n_total > k:
            start = np.random.randint(0, n_total - k + 1)
        return np.arange(start, start + k)

    @staticmethod
    def _augment(vol: torch.Tensor) -> torch.Tensor:
        """
        Aumento de datos aplicado IGUAL a todos los cortes del volumen:
        pequeña rotación/escala/traslación + cambios de brillo y contraste.
        No usamos volteo anteroposterior: cambiaría la anatomía de la rodilla.
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
        v = torch.from_numpy(vol).float() / 255.0          # (S_total, H, W) en [0, 1]
        if self.train:
            v = self._augment(v)
        # Triplete (i-1, i, i+1) para cada corte elegido, con bordes repetidos
        prev_idx = np.clip(idx - 1, 0, vol.shape[0] - 1)
        next_idx = np.clip(idx + 1, 0, vol.shape[0] - 1)
        x = torch.stack([v[prev_idx], v[idx], v[next_idx]], dim=1)   # (S, 3, H, W)
        # Normalización tipo ImageNet (media/desv. aproximadas para escala de grises)
        x = (x - 0.45) / 0.225
        return x, self.y[i], self.w[i]
