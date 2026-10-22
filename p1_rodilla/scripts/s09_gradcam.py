"""
PASO 9. Interpretabilidad con Grad-CAM (entregable 5).

Para cada hallazgo de gradcam.findings (al menos 3), toma n_cases estudios DORADOS positivos
con mayor probabilidad predicha y muestra:
    corte con más atención | mapa Grad-CAM superpuesto
Criterio de análisis: ¿el mapa resalta la región anatómica
correcta? (menisco -> cuernos meniscales; LCA -> escotadura intercondílea; derrame ->
receso suprapatelar; contusión -> hueso subcondral).

Salidas: <runs_dir>/<exp>/gradcam/<hallazgo>.png
Uso:  python -m p1_rodilla.scripts.s09_gradcam --exp B_resnet50_bce
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_device, get_logger
from p1_rodilla.src.engine import load_split, make_dataset
from p1_rodilla.src.gradcam import GradCAM, overlay
from p1_rodilla.src.labels_rules import POS
from p1_rodilla.src.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    args = ap.parse_args()
    base = load_config(args.config)
    exp_dir = Path(base["data"]["runs_dir"]) / args.exp
    cfg = load_config(exp_dir / "config.yaml")
    log = get_logger("s09")
    names = cfg["labels"]["names"]
    device = get_device(cfg["train"].get("device", "auto"))
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(exp_dir / "best.pt", map_location=device)["model"])
    model.eval()
    cam = GradCAM(model)

    split = load_split(cfg)
    gold = split[split.split == "gold"].reset_index(drop=True)
    ds = make_dataset(cfg, gold, train=False)
    preds = pd.read_csv(exp_dir / "preds_gold.csv", dtype={"study_id": str}) if (exp_dir / "preds_gold.csv").exists() else None
    out_dir = ensure_dir(exp_dir / "gradcam")
    n_cases = int(cfg["gradcam"].get("n_cases", 3))

    for finding in cfg["gradcam"]["findings"]:
        k = names.index(finding)
        pos = gold.index[gold[finding] == POS].tolist()
        if not pos:
            log.warning(f"{finding}: no hay estudios dorados positivos")
            continue
        if preds is not None:
            pos = sorted(pos, key=lambda i: -float(preds.loc[preds.study_id == gold.study_id[i], finding].iloc[0]))
        pos = pos[:n_cases]
        fig, axes = plt.subplots(len(pos), 2, figsize=(7, 3.4 * len(pos)), squeeze=False)
        for row, i in enumerate(pos):
            x, _, _ = ds[i]
            x = x.unsqueeze(0).to(device)
            cams, att, prob = cam(x, k)
            s = int(np.argmax(att)) if att is not None else int(np.argmax(cams.reshape(len(cams), -1).sum(1)))
            img = x[0, s, 1].detach().cpu().numpy()          # canal central del triplete
            axes[row, 0].imshow(img, cmap="gray")
            axes[row, 0].set_title(f"…{gold.study_id[i][-8:]} | corte {s}", fontsize=9)
            axes[row, 1].imshow(overlay(img, cams[s]))
            axes[row, 1].set_title(f"Grad-CAM {finding} | p={prob:.2f}", fontsize=9)
            for a in axes[row]:
                a.axis("off")
        fig.tight_layout()
        p = out_dir / f"{finding}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        log.info(f"Guardado {p}")
    cam.remove()


if __name__ == "__main__":
    main()
