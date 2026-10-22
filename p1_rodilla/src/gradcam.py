"""
Grad-CAM implementado desde cero (Selvaraju et al., 2017).

Idea
----
Para el hallazgo k, queremos saber QUÉ PÍXELES empujaron su logit hacia arriba.
1. Guardamos el mapa de características A de la última capa convolucional
   (forma C x h x w) y el gradiente dLogit_k/dA (misma forma).
2. Peso de cada canal c: alpha_c = promedio espacial del gradiente del canal c.
3. Mapa: CAM = ReLU( sum_c alpha_c * A_c )   (solo lo que AUMENTA el logit).
4. Se reescala al tamaño de la imagen y se superpone en color.

En nuestro modelo 2.5D hay un mapa por corte; mostramos el corte con mayor
peso de atención (el que el modelo "miró" más) o el de mayor energía del CAM.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        enc = model.encoder
        self._orig = enc.forward_features

        def hooked(x):
            a = self._orig(x)
            a.retain_grad()          # pedimos a PyTorch que guarde el gradiente de este tensor
            self.acts = a
            return a

        enc.forward_features = hooked

    def remove(self):
        self.model.encoder.forward_features = self._orig

    def __call__(self, x: torch.Tensor, k: int):
        """x: (1, S, 3, H, W). Devuelve (cams (S,H,W) en [0,1], atención (S,) o None, prob_k)."""
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
    """Superpone el mapa (colormap 'jet') sobre la imagen en gris. Devuelve RGB float [0,1]."""
    import matplotlib.cm as cm

    g = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * np.stack([g] * 3, -1) + alpha * heat, 0, 1)
