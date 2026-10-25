"""
Etapa 8. ¿El modelo aprende el animal o el fondo de la cámara?

Compara cámaras vistas (cis_test) y no vistas (trans_test) con tres pruebas:

A. Grad-CAM de la especie real en la capa 9 (fin del backbone) y 15 (P3 del neck).
B. Concentración del mapa en las cajas reales (> 1: se concentra en el animal).
C. Enmascarado: se borra cada animal con desenfoque fuerte y se vuelve a predecir; si el modelo
   sigue diciendo la misma especie, usa el contexto. Se reporta la tasa por partición.

Salidas en <runs_dir>/<exp>/fondo/:
    detalle.csv, resumen.csv, concentracion.png, L<capa>_<particion>_<k>.jpg

Uso:
    python -m p3_fauna.scripts.s08_fondo_gradcam --exp B_yolov8s
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_device, get_logger
from p3_fauna.src.experimentos import best_experiment, exp_dir, load_exp_config, weights
from p3_fauna.src.gradcam import YoloGradCAM, energy_in_boxes, letterbox, mask_animals, overlay, unletterbox
from p3_fauna.src.yolo_io import labels_xyxy, ultralytics_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    ap.add_argument("--exp", default=None)
    args = ap.parse_args()
    base = load_config(args.config)
    exp = args.exp or best_experiment(base)[0] or base["ablation"]["reference"]
    cfg = load_exp_config(base, exp)
    cfg["eval"], cfg["background"] = base["eval"], base["background"]
    out = ensure_dir(exp_dir(cfg, exp) / "fondo")
    log = get_logger("s08", out / "fondo.log")
    bg, imgsz = cfg["background"], int(cfg["train"]["imgsz"])
    dev_yolo = ultralytics_device(cfg["train"].get("device", "auto"))
    dev_torch = get_device("auto")
    if dev_torch.type == "mps":   # los hooks de gradiente son más estables en CPU que en MPS
        dev_torch = get_device("cpu")

    from ultralytics import YOLO

    w = weights(cfg, exp)
    predictor = YOLO(str(w))
    meta = pd.read_csv(Path(cfg["data"]["yolo_dir"]) / "images_meta.csv")
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    samples = {}
    for split in ("cis_test", "trans_test"):
        pool = meta[(meta.particion == split) & (meta.n_boxes > 0)].image.values
        if len(pool):
            samples[split] = rng.choice(pool, min(int(bg["n_images"]), len(pool)), replace=False)

    rows = []
    for li, layer in enumerate(bg["layers"]):
        cam_model = YoloGradCAM(YOLO(str(w)).model, int(layer), dev_torch)
        for split, paths in samples.items():
            for k, path in enumerate(paths):
                img = cv2.imread(path)
                H, W = img.shape[:2]
                gt = labels_xyxy(path, W, H)
                x, lb = letterbox(img, imgsz)
                cam, c, score = cam_model(x, int(gt[0, 0]))
                cam = unletterbox(cam, lb, (H, W))
                row = {"capa": layer, "particion": split, "image": path, "especie": predictor.names[c],
                       "puntaje": score, "concentracion_caja": energy_in_boxes(cam, gt[:, 1:])}
                if li == 0:
                    r = predictor.predict(mask_animals(img, gt[:, 1:], bg["mask_blur_sigma"]), imgsz=imgsz,
                                          conf=cfg["eval"]["deploy_conf"], device=dev_yolo, verbose=False)[0]
                    row["especie_sin_animal"] = bool(set(gt[:, 0].astype(int)) & set(r.boxes.cls.cpu().numpy().astype(int)))
                rows.append(row)
                if k < int(bg["n_figures"]):
                    cv2.imwrite(str(out / f"L{layer}_{split}_{k:02d}.jpg"),
                                overlay(img, cam, gt, f"{split} capa {layer} {predictor.names[c]} conc={row['concentracion_caja']:.2f}"))
        log.info(f"Capa {layer}: {sum(len(p) for p in samples.values())} imágenes")

    det = pd.DataFrame(rows)
    det.to_csv(out / "detalle.csv", index=False)
    res = det.groupby(["capa", "particion"]).agg(
        imagenes=("image", "size"), concentracion_mediana=("concentracion_caja", "median"),
        frac_concentracion_mayor_1=("concentracion_caja", lambda s: (s > 1).mean()),
        tasa_especie_sin_animal=("especie_sin_animal", "mean")).reset_index()
    res.to_csv(out / "resumen.csv", index=False)
    log.info("\n" + res.round(3).to_string(index=False))
    log.info("Lectura: menor concentración en trans y una tasa alta de 'especie sin animal' en cis indican que el "
             "modelo se apoya en el fondo de las cámaras conocidas.")

    fig, axes = plt.subplots(1, len(bg["layers"]), figsize=(4.5 * len(bg["layers"]), 3.5), squeeze=False)
    for ax, layer in zip(axes[0], bg["layers"]):
        d = det[det.capa == layer]
        data = [np.log10(d[d.particion == s].concentracion_caja.dropna().clip(lower=1e-3)) for s in samples]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(samples) + 1), list(samples))
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_title(f"Capa {layer}")
        ax.set_ylabel("log10 concentración en la caja")
    fig.tight_layout()
    fig.savefig(out / "concentracion.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
