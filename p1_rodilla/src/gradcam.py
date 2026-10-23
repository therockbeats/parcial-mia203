"""
Grad-CAM (Selvaraju et al., 2017) sobre la salida de ``encoder.forward_features``.

Para el hallazgo k: alpha_c = media espacial de d logit_k / d A_c y
CAM = ReLU(sum_c alpha_c A_c), reescalado al tamaño de la imagen y normalizado a [0, 1].
Se obtiene un mapa por corte.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Calcula Grad-CAM interceptando ``forward_features`` del codificador."""

    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        enc = model.encoder
        self._orig = enc.forward_features

        def hooked(x):
            a = self._orig(x)
            a.retain_grad()
            self.acts = a
            return a

        enc.forward_features = hooked

    def remove(self):
        """Restaura el método original del codificador."""
        self.model.encoder.forward_features = self._orig

    def __call__(self, x: torch.Tensor, k: int):
        """Para ``x`` (1, S, 3, H, W) devuelve mapas (S, H, W), pesos de atención (S,) y probabilidad."""
        self.model.zero_grad(set_to_none=True)
        logits, att = self.model(x, return_attention=True)
        logits[0, k].backward()
        A, G = self.acts, self.acts.grad                 # (S, C, h, w)
        alpha = G.mean(dim=(2, 3), keepdim=True)         # (S, C, 1, 1)
        cam = F.relu((alpha * A).sum(dim=1, keepdim=True))   # (S, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam.detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        prob = torch.sigmoid(logits[0, k]).item()
        return cam, (att[0].detach().cpu().numpy() if att is not None else None), prob


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superposición del mapa (colormap jet) sobre la imagen en gris, en RGB [0, 1]."""
    import matplotlib.cm as cm

    g = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * np.stack([g] * 3, -1) + alpha * heat, 0, 1)
