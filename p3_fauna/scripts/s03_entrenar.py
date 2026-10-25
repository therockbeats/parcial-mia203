"""
Etapa 3. Entrenamiento de un experimento (ajuste fino de YOLOv8 con Ultralytics).

Arma la lista de entrenamiento del experimento a partir de ``sampling`` (proporción de imágenes
vacías como fondo y Repeat Factor Sampling para especies poco frecuentes), escribe su data.yaml y
entrena con los argumentos de ``train``. La validación durante el entrenamiento (detención
temprana y mejor modelo) usa ``eval.select_split`` (cis_val): las cámaras trans quedan reservadas
para medir la generalización.

Salidas en <runs_dir>/<exp>/:
    weights/best.pt, history.csv, config.yaml, train.txt, data.yaml, muestreo.json, train.log
    (más los archivos propios de Ultralytics: args.yaml, results.csv, results.png, ...)

Uso:
    python -m p3_fauna.scripts.s03_entrenar --exp B_yolov8s
    python -m p3_fauna.scripts.s03_entrenar --exp prueba --set train.epochs=1 --set train.imgsz=320
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from common.config import REPO_ROOT, load_config, save_config
from common.utils import ensure_dir, get_logger, set_seed
from p3_fauna.src.sampling import build_train_list
from p3_fauna.src.yolo_io import class_names_from_yolo, ultralytics_device, write_data_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    ap.add_argument("--exp", required=True)
    ap.add_argument("--set", action="append", default=[], help="override seccion.clave=valor (repetible)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    yolo = Path(cfg["data"]["yolo_dir"])
    out = ensure_dir(Path(cfg["data"]["runs_dir"]) / args.exp)
    log = get_logger("s03", out / "train.log")
    save_config(cfg, out / "config.yaml")

    meta = pd.read_csv(yolo / "images_meta.csv")
    entries, summ = build_train_list(meta[meta.particion == "train"], cfg["sampling"], seed)
    (out / "train.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    (out / "muestreo.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    names = class_names_from_yolo(yolo)
    val_list = yolo / "lists" / f"{cfg['eval']['select_split']}.txt"
    data_yaml = write_data_yaml(out / "data.yaml", yolo, out / "train.txt", val_list, names)
    log.info(f"Experimento {args.exp} | entrenamiento: {summ['imagenes_con_animal']:,} con animal + "
             f"{summ['imagenes_vacias']:,} vacías -> {summ['entradas_lista']:,} entradas por época "
             f"(RFS {'sí' if summ['factores_rfs'] else 'no'}) | validación: {val_list.name}")
    if summ["factores_rfs"]:
        top = sorted(summ["factores_rfs"].items(), key=lambda kv: -kv[1])[:5]
        log.info("Factores RFS más altos: " + ", ".join(f"{k} {v:.2f}" for k, v in top))

    from ultralytics import YOLO

    t = dict(cfg["train"])
    model_name, device = t.pop("model"), ultralytics_device(t.pop("device", "auto"))
    log.info(f"Modelo {model_name} | dispositivo {device} | imgsz {t['imgsz']} | épocas {t['epochs']} | lote {t['batch']}")
    t0 = time.time()
    YOLO(model_name).train(data=str(data_yaml), project=str(out.parent), name=args.exp, exist_ok=True,
                           device=device, seed=seed, deterministic=True, verbose=False, **t)
    if (out / "results.csv").exists():
        hist = pd.read_csv(out / "results.csv")
        hist.columns = [c.strip() for c in hist.columns]
        hist.to_csv(out / "history.csv", index=False)
        col = "metrics/mAP50-95(B)"
        if col in hist:
            i = int(hist[col].idxmax())
            log.info(f"Mejor época {int(hist.loc[i, 'epoch'])}: mAP@0.5:0.95 {cfg['eval']['select_split']} {hist.loc[i, col]:.4f}")
    if not (out / "weights" / "best.pt").exists() and (out / "weights" / "last.pt").exists():
        shutil.copy2(out / "weights" / "last.pt", out / "weights" / "best.pt")
    log.info(f"Entrenamiento: {(time.time() - t0) / 60:.1f} min. Pesos en {out / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
