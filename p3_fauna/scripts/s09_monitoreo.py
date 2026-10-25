"""
Etapa 9. Módulo de monitoreo ecológico en una cámara no vista.

Procesa todos los fotogramas (incluidos los vacíos) de una cámara de ``monitor.split``, aplica las
reglas de ``p3_fauna/src/monitoreo.py`` (presencia por fotograma, MaxN por secuencia, eventos
independientes, índice de abundancia relativa, actividad por periodo y franja horaria) y genera
el mismo reporte con las etiquetas de referencia para comparar.

Salidas en <runs_dir>/<exp>/monitoreo/:
    reporte_camara_<id>.md, por_especie.csv, por_periodo.csv, por_franja.csv, eventos.csv,
    detecciones.csv, validacion.csv

Uso:
    python -m p3_fauna.scripts.s09_monitoreo --exp B_yolov8s [--camara 38]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p3_fauna.src.experimentos import best_experiment, exp_dir, load_exp_config, weights
from p3_fauna.src.monitoreo import build_report, report_markdown
from p3_fauna.src.yolo_io import class_names_from_yolo, read_labels, ultralytics_device


def predict_frames(model, frames: list[str], cfg: dict, device) -> pd.DataFrame:
    rows = []
    res = model.predict(frames, imgsz=cfg["train"]["imgsz"], conf=cfg["eval"]["deploy_conf"], device=device,
                        stream=True, verbose=False)
    for path, r in zip(frames, res):
        cls, conf = r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()
        if len(cls) == 0:
            rows.append({"image": path, "species": None, "conf": np.nan})
        rows += [{"image": path, "species": r.names[c], "conf": float(p)} for c, p in zip(cls, conf)]
    return pd.DataFrame(rows)


def reference_frames(frames: list[str], names: list[str]) -> pd.DataFrame:
    rows = []
    for path in frames:
        lab = read_labels(path)
        if len(lab) == 0:
            rows.append({"image": path, "species": None})
        rows += [{"image": path, "species": names[int(c)]} for c in lab[:, 0]]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    ap.add_argument("--exp", default=None)
    ap.add_argument("--camara", default=None)
    args = ap.parse_args()
    base = load_config(args.config)
    exp = args.exp or best_experiment(base)[0] or base["ablation"]["reference"]
    cfg = load_exp_config(base, exp)
    cfg["eval"], mc = base["eval"], base["monitor"]
    out = ensure_dir(exp_dir(cfg, exp) / "monitoreo")
    log = get_logger("s09", out / "monitoreo.log")
    yolo = Path(cfg["data"]["yolo_dir"])

    meta = pd.read_csv(yolo / "images_meta.csv", dtype={"camara": str})
    meta = meta[meta.particion == mc["split"]]
    cam = str(args.camara or mc.get("camera") or meta.camara.value_counts().index[0])
    frames = meta[meta.camara == cam].sort_values("date_captured")
    if frames.empty:
        raise ValueError(f"La cámara {cam} no tiene fotogramas en {mc['split']}")
    log.info(f"Cámara {cam} ({mc['split']}): {len(frames):,} fotogramas | modelo {exp}")

    from ultralytics import YOLO

    model = YOLO(str(weights(cfg, exp)))
    det = predict_frames(model, frames.image.tolist(), cfg, ultralytics_device(cfg["train"].get("device", "auto")))
    det.to_csv(out / "detecciones.csv", index=False)
    pred = build_report(frames, det, int(mc["independence_minutes"]), mc["period"])
    ref = build_report(frames, reference_frames(frames.image.tolist(), class_names_from_yolo(yolo)),
                       int(mc["independence_minutes"]), mc["period"])
    md, cmp = report_markdown(cam, pred, ref, cfg["eval"]["deploy_conf"])
    (out / f"reporte_camara_{cam}.md").write_text(md, encoding="utf-8")
    pred["by_species"].to_csv(out / "por_especie.csv", index=False)
    pred["by_period"].to_csv(out / "por_periodo.csv", index=False)
    pred["by_hour"].to_csv(out / "por_franja.csv")
    pred["events"].to_csv(out / "eventos.csv", index=False)
    if cmp is not None:
        cmp.to_csv(out / "validacion.csv", index=False)
    log.info("\n" + md)


if __name__ == "__main__":
    main()
