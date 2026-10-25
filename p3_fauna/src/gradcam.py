"""
Grad-CAM para YOLOv8 y medidas de dependencia del fondo.

Grad-CAM (Selvaraju et al., 2017): el puntaje es la probabilidad máxima de la clase objetivo
sobre todas las posiciones de la cabeza; el mapa es ReLU(sum_k mean(dS/dA_k) * A_k) en la capa
elegida. Concentración en la caja: (energía del mapa dentro de las cajas / energía total) /
(área de las cajas / área total); un valor > 1 indica que el mapa se concentra en el animal.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


def letterbox(img_bgr: np.ndarray, size: int = 640):
    """Redimensiona conservando la proporción y rellena con gris (como Ultralytics)."""
    h, w = img_bgr.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0), (top, left, nh, nw)


def unletterbox(cam: np.ndarray, meta, orig_hw) -> np.ndarray:
    top, left, nh, nw = meta
    return cv2.resize(cam[top:top + nh, left:left + nw], (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)


class YoloGradCAM:
    """Grad-CAM sobre una capa de ``DetectionModel`` (``YOLO(pesos).model``)."""

    def __init__(self, detection_model, layer_idx: int, device):
        self.m = detection_model.to(device).float().eval()
        for p in self.m.parameters():
            p.requires_grad_(True)
        self.device = device
        self.act = self.grad = None
        self.m.model[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.act = out
        out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x: torch.Tensor, target_cls: int | None = None):
        x = x.to(self.device).requires_grad_(True)
        self.m.zero_grad()
        out = self.m(x)
        y = out[0] if isinstance(out, (list, tuple)) else out    # (1, 4 + nc, posiciones)
        cls_scores = y[0, 4:, :]
        if target_cls is None:
            target_cls = int(cls_scores.max(1).values.argmax())
        score = cls_scores[target_cls].max()
        score.backward()
        weights = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.act).sum(1))[0].detach()
        cam = torch.nn.functional.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear")[0, 0].cpu().numpy()
        rng = cam.max() - cam.min()
        cam = (cam - cam.min()) / rng if rng > 0 else np.zeros_like(cam)
        return cam, target_cls, float(score.detach())


def energy_in_boxes(cam: np.ndarray, boxes_xyxy: np.ndarray) -> float:
    """Concentración del mapa en las cajas reales, normalizada por el área que ocupan."""
    if len(boxes_xyxy) == 0 or cam.sum() == 0:
        return float("nan")
    mask = np.zeros_like(cam, dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxy.astype(int):
        mask[max(y1, 0):y2, max(x1, 0):x2] = True
    return float((cam[mask].sum() / cam.sum()) / (mask.mean() + 1e-9))


def mask_animals(img_bgr: np.ndarray, boxes_xyxy: np.ndarray, sigma: float = 25) -> np.ndarray:
    """Reemplaza cada caja por una versión fuertemente desenfocada de sí misma (borra al animal)."""
    out = img_bgr.copy()
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma)
    for x1, y1, x2, y2 in boxes_xyxy.astype(int):
        out[max(y1, 0):y2, max(x1, 0):x2] = blur[max(y1, 0):y2, max(x1, 0):x2]
    return out


def overlay(img_bgr: np.ndarray, cam: np.ndarray, gt: np.ndarray, title: str) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img_bgr, 0.55, heat, 0.45, 0)
    for _, x1, y1, x2, y2 in gt:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
    cv2.putText(out, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out
