"""
PASO 7. Entrenamiento de un modelo (un experimento).

Ejemplos (desde la raíz del repositorio):
    # Modelo B: ResNet-50 preentrenada (valores por defecto del config)
    python -m p1_rodilla.scripts.s07_entrenar --exp B_resnet50_bce

    # Modelo A: CNN desde cero
    python -m p1_rodilla.scripts.s07_entrenar --exp A_scratch_bce --set model.backbone=scratch --set model.pretrained=false --set train.lr=1e-3

    # Prueba rápida en la Mac (2 épocas, pocos estudios)
    python -m p1_rodilla.scripts.s07_entrenar --exp prueba_mac --set train.epochs=2 --smoke

Qué hace en cada época:
    1. Recorre los lotes de entrenamiento: forward -> pérdida -> backward -> paso del optimizador.
    2. Evalúa en validación: AUC-ROC macro (métrica oficial) usando solo etiquetas con peso > 0.
    3. Si el AUC mejora, guarda best.pt. Si no mejora en 'early_stopping_patience' épocas, se detiene.

Salidas en <runs_dir>/<exp>/:
    best.pt  (pesos), history.csv (pérdida y AUC por época), config.yaml (config efectivo), train.log
"""
from __future__ import annotations

import argparse
import math
import time

import pandas as pd
import torch

from common.config import REPO_ROOT, load_config, save_config
from common.utils import ensure_dir, get_device, get_logger, set_seed
from p1_rodilla.src.engine import autocast_ctx, load_split, make_dataset, make_loader, predict
from p1_rodilla.src.losses import build_loss
from p1_rodilla.src.metrics import macro_auc
from p1_rodilla.src.models import build_model, count_parameters


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Learning rate: sube linealmente durante el warm-up y luego baja como un coseno hasta ~0."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--exp", required=True, help="nombre del experimento (carpeta de salida)")
    ap.add_argument("--set", action="append", default=[], help="override seccion.clave=valor (repetible)")
    ap.add_argument("--smoke", action="store_true", help="prueba rápida: 24 estudios de train y 12 de val")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg.get("seed", 42))
    out = ensure_dir(f"{cfg['data']['runs_dir']}/{args.exp}")
    log = get_logger("s07", out / "train.log")
    save_config(cfg, out / "config.yaml")
    t = cfg["train"]
    device = get_device(t.get("device", "auto"))
    log.info(f"Experimento {args.exp} | dispositivo {device}")

    # ---------- Datos ----------
    split = load_split(cfg)
    tr_df, va_df = split[split.split == "train"], split[split.split == "val"]
    if args.smoke:
        tr_df, va_df = tr_df.head(24), va_df.head(12)
    ds_tr, ds_va = make_dataset(cfg, tr_df, train=True), make_dataset(cfg, va_df, train=False)
    dl_tr, dl_va = make_loader(cfg, ds_tr, train=True), make_loader(cfg, ds_va, train=False)
    log.info(f"Train: {len(ds_tr)} estudios | Val: {len(ds_va)} estudios")

    # ---------- Modelo, pérdida, optimizador ----------
    model = build_model(cfg).to(device)
    log.info(f"Modelo: {cfg['model']['backbone']} (preentrenado={cfg['model'].get('pretrained')}) | "
             f"parámetros entrenables: {count_parameters(model):,}")
    criterion = build_loss(cfg, ds_tr.y, ds_tr.w).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    steps_per_epoch = max(1, len(dl_tr))
    scheduler = cosine_with_warmup(optimizer, int(t.get("warmup_epochs", 1)) * steps_per_epoch,
                                   int(t["epochs"]) * steps_per_epoch)
    amp = bool(t.get("amp", True))

    # ---------- Bucle de entrenamiento ----------
    best, wait, history = -1.0, 0, []
    for epoch in range(1, int(t["epochs"]) + 1):
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        for x, y, w in dl_tr:
            x, y, w = x.to(device, non_blocking=True), y.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)          # 1) limpiar gradientes anteriores
            with autocast_ctx(device, amp):
                logits = model(x)                           # 2) forward
            loss = criterion(logits.float(), y, w)          # 3) pérdida (en float32 por estabilidad)
            loss.backward()                                 # 4) backward: gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(t.get("grad_clip", 1.0)))
            optimizer.step()                                # 5) actualizar pesos
            scheduler.step()                                # 6) actualizar learning rate
            running += loss.item()
            nb += 1
        p_va, y_va, w_va = predict(model, dl_va, device, amp)
        auc = macro_auc(y_va, p_va, mask=w_va)
        row = {"epoch": epoch, "train_loss": running / max(nb, 1), "val_macro_auc": auc,
               "lr": scheduler.get_last_lr()[0], "seg": round(time.time() - t0, 1)}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        log.info(f"época {epoch:02d} | pérdida {row['train_loss']:.4f} | AUC val {auc:.4f} | {row['seg']} s")
        if not math.isnan(auc) and auc > best:
            best, wait = auc, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_auc": auc}, out / "best.pt")
        else:
            wait += 1
            if wait >= int(t.get("early_stopping_patience", 6)):
                log.info(f"Early stopping: {wait} épocas sin mejorar")
                break
    if best < 0:  # validación sin positivos suficientes (p. ej. en --smoke): guardamos el último
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_auc": float("nan")}, out / "best.pt")
    log.info(f"Mejor AUC macro en validación: {best:.4f}. Pesos en {out / 'best.pt'}")


if __name__ == "__main__":
    main()
