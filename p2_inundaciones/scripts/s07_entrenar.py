"""
Etapa 7. Entrenamiento de un experimento.

Entrena la U-Net con un presupuesto fijo de pasos (``train.steps``), de modo que las
configuraciones con distinta cantidad de datos reciben el mismo número de actualizaciones. Usa
AdamW con calentamiento lineal y decaimiento coseno, precisión mixta bf16 en CUDA y recorte de
gradiente. Cada ``eval_every`` pasos evalúa el IoU agregado en la validación dorada (umbral 0.5)
y conserva el mejor modelo; se detiene tras ``patience`` evaluaciones sin mejora.

Con ``train.init_from`` el modelo parte de los pesos de otro experimento (preentrenamiento con
etiquetas débiles seguido de ajuste fino con etiquetas doradas).

Salidas en <runs_dir>/<exp>/:
    best.pt, history.csv, config.yaml, train.log

Uso:
    python -m p2_inundaciones.scripts.s07_entrenar --exp GW_bcedice
    python -m p2_inundaciones.scripts.s07_entrenar --exp prueba --set train.steps=50 --set train.eval_every=25
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch

from common.config import REPO_ROOT, load_config, save_config
from common.utils import ensure_dir, get_device, get_logger, set_seed
from p2_inundaciones.src.engine import autocast_ctx, make_train_loader, predict_items, split_items, train_items
from p2_inundaciones.src.losses import SegLoss
from p2_inundaciones.src.metrics import confusion, scores
from p2_inundaciones.src.unet import build_model, count_parameters


def lr_lambda(warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def val_iou(model, cfg, items, device, amp) -> float:
    probs, labels, valids = predict_items(model, cfg, items, device, amp)
    c = confusion(probs >= 0.5, labels, valids)
    return scores(c["tp"], c["fp"], c["fn"], c["tn"])["iou"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--exp", required=True)
    ap.add_argument("--set", action="append", default=[], help="override seccion.clave=valor (repetible)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(int(cfg.get("seed", 42)))
    out = ensure_dir(Path(cfg["data"]["runs_dir"]) / args.exp)
    log = get_logger("s07", out / "train.log")
    save_config(cfg, out / "config.yaml")
    t = cfg["train"]
    device = get_device(t.get("device", "auto"))
    amp = bool(t.get("amp", True))

    items, weights = train_items(cfg)
    val = split_items(cfg, "val")
    if not items or not val:
        raise RuntimeError("Sin chips de entrenamiento o validación: revisar la etapa 3 y el config")
    n_gold = sum(1 for _, s in items if s == "LabelHand")
    log.info(f"Experimento {args.exp} | {device} | entrenamiento: {n_gold} dorados + {len(items) - n_gold} débiles "
             f"({t['supervision']}) | validación: {len(val)}")

    model = build_model(cfg).to(device)
    if t.get("init_from"):
        ck = Path(cfg["data"]["runs_dir"]) / str(t["init_from"]) / "best.pt"
        model.load_state_dict(torch.load(ck, map_location=device)["model"])
        log.info(f"Pesos iniciales: {ck}")
    log.info(f"U-Net: profundidad {cfg['model']['depth']}, salto {cfg['model']['skip']}, "
             f"{count_parameters(model):,} parámetros")
    criterion = SegLoss(t["loss"], float(t.get("pos_weight", 1.0)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    steps = int(t["steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(int(t.get("warmup_steps", 0)), steps))
    loader = make_train_loader(cfg, items, weights)

    best, wait, hist, run_loss, t0 = -1.0, 0, [], 0.0, time.time()
    step = 0
    for x, y, w in loader:
        model.train()
        x, y, w = x.to(device, non_blocking=True), y.to(device), w.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp):
            logits = model(x)
        loss = criterion(logits, y, w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(t.get("grad_clip", 1.0)))
        opt.step()
        sched.step()
        run_loss += loss.item()
        step += 1
        if step % int(t["eval_every"]) == 0 or step == steps:
            iou = val_iou(model, cfg, val, device, amp)
            row = {"paso": step, "perdida": run_loss / int(t["eval_every"]), "iou_val": iou,
                   "lr": sched.get_last_lr()[0], "seg": round(time.time() - t0, 1)}
            hist.append(row)
            pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
            log.info(f"paso {step:5d} | pérdida {row['perdida']:.4f} | IoU val {iou:.4f} | {row['seg']} s")
            run_loss = 0.0
            if iou == iou and iou > best:
                best, wait = iou, 0
                torch.save({"model": model.state_dict(), "step": step, "val_iou": iou}, out / "best.pt")
            else:
                wait += 1
                if wait >= int(t.get("patience", 10)):
                    log.info(f"Detención temprana tras {wait} evaluaciones sin mejora")
                    break
        if step >= steps:
            break
    if best < 0:
        torch.save({"model": model.state_dict(), "step": step, "val_iou": float("nan")}, out / "best.pt")
    log.info(f"Mejor IoU de validación: {best:.4f}. Pesos en {out / 'best.pt'}")


if __name__ == "__main__":
    main()
