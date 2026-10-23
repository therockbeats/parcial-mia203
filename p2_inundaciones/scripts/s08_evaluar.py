"""
Etapa 8. Evaluación de un experimento.

El umbral de decisión se elige en la validación dorada (máximo IoU agregado) y se aplica sin
cambios a la prueba en distribución (chips dorados de los eventos de entrenamiento), a Bolivia
(evento no visto) y a Lima. Para la clase agua se reportan IoU, F1, precisión, sensibilidad,
omisión y comisión sobre todos los píxeles válidos, el IoU medio por chip, el IC 95 % del IoU por
bootstrap de chips, la matriz de confusión y el desglose por evento.

Salidas en <runs_dir>/<exp>/:
    metricas.csv, por_chip_<particion>.csv, por_evento_test.csv, preds_<particion>.npz,
    confusion_test.png, summary.json

Uso:
    python -m p2_inundaciones.scripts.s08_evaluar --exp GW_bcedice
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common.config import REPO_ROOT, load_config
from common.utils import get_device, get_logger
from p2_inundaciones.src.engine import per_chip_table, predict_items, split_items
from p2_inundaciones.src.metrics import best_threshold, bootstrap_iou, chip_table_scores
from p2_inundaciones.src.unet import build_model


def plot_confusion(c: dict, title: str, path: Path) -> None:
    m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
    norm = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{norm[i, j]:.3f}\n({int(m[i, j]):,} px)", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks([0, 1], ["no agua", "agua"])
    ax.set_yticks([0, 1], ["no agua", "agua"])
    ax.set_xlabel("predicción")
    ax.set_ylabel("referencia")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--exp", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    base = load_config(args.config)
    exp_dir = Path(base["data"]["runs_dir"]) / args.exp
    cfg = load_config(exp_dir / "config.yaml", args.set)
    cfg["data"] = {**cfg["data"], **{k: base["data"][k] for k in ("raw_dir", "interim_dir", "processed_dir", "lima_dir", "runs_dir")}}
    log = get_logger("s08", exp_dir / "eval.log")
    device = get_device(cfg["train"].get("device", "auto"))
    amp = bool(cfg["train"].get("amp", True))
    model = build_model(cfg).to(device)
    ck = torch.load(exp_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    log.info(f"{args.exp}: pesos del paso {ck['step']} (IoU val {ck['val_iou']:.4f})")

    val = split_items(cfg, "val")
    pv, lv, vv = predict_items(model, cfg, val, device, amp)
    np.savez_compressed(exp_dir / "preds_val.npz", probs=(pv * 255).round().astype(np.uint8),
                        chip_ids=np.array([c for c, _ in val]))
    thr_cfg = cfg["eval"].get("threshold", "val")
    thr = best_threshold(pv, lv, vv) if thr_cfg == "val" else float(thr_cfg)
    log.info(f"Umbral: {thr:.2f}")

    rows, summary = [], {"exp": args.exp, "umbral": thr, "paso": int(ck["step"]), "iou_val_05": float(ck["val_iou"])}
    splits = ["val"] + [s for s in cfg["eval"]["splits"] if not (s == "lima" and cfg["train"].get("use_s2"))]
    for split in splits:
        items = val if split == "val" else split_items(cfg, split)
        if not items:
            log.info(f"{split}: sin chips; se omite")
            continue
        if split == "val":
            probs, labels, valids = pv, lv, vv
        else:
            probs, labels, valids = predict_items(model, cfg, items, device, amp)
            np.savez_compressed(exp_dir / f"preds_{split}.npz", probs=(probs * 255).round().astype(np.uint8),
                                chip_ids=np.array([c for c, _ in items]))
        ids = [c for c, _ in items]
        pc = per_chip_table(ids, probs, labels, valids, thr)
        pc.to_csv(exp_dir / f"por_chip_{split}.csv", index=False)
        s = chip_table_scores(pc)
        lo, hi = bootstrap_iou(pc.tp.values, pc.fp.values, pc.fn.values, int(cfg["eval"]["bootstrap"]), int(cfg.get("seed", 0)))
        s05 = chip_table_scores(per_chip_table(ids, probs, labels, valids, 0.5))
        rows.append({"particion": split, "chips": len(ids), **s, "iou_ic95_inf": lo, "iou_ic95_sup": hi,
                     "iou_umbral_05": s05["iou"], "tp": int(pc.tp.sum()), "fp": int(pc.fp.sum()),
                     "fn": int(pc.fn.sum()), "tn": int(pc.tn.sum())})
        summary[f"iou_{split}"] = s["iou"]
        summary[f"iou_{split}_ic95"] = [lo, hi]
        if split == "test":
            plot_confusion({k: pc[k].sum() for k in ("tp", "fp", "fn", "tn")},
                           f"Prueba en distribución: {args.exp}", exp_dir / "confusion_test.png")
            pc["event"] = pc.chip_id.str.rsplit("_", n=1).str[0]
            ev = pd.DataFrame([{"evento": e, "chips": len(g), **chip_table_scores(g)} for e, g in pc.groupby("event")])
            ev.to_csv(exp_dir / "por_evento_test.csv", index=False)
    met = pd.DataFrame(rows)
    met.to_csv(exp_dir / "metricas.csv", index=False)
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cols = ["particion", "chips", "iou", "iou_ic95_inf", "iou_ic95_sup", "f1", "precision", "recall", "iou_medio_chip", "iou_umbral_05"]
    log.info("\n" + met[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
