"""
Etapa 7. Robustez ante el tamaño del animal y la oclusión.

Grupos por caja: tamaño relativo (área de la caja / área de la imagen: pequeño < 1 % <= mediano
< 10 % <= grande) y oclusión (proxy de la etapa 2 o, si existe, la anotación manual de
``robustness.manual_occlusion_csv``, cuyo acuerdo con el proxy se reporta). Dos medidas:

- recall por caja: la caja se detecta con la especie correcta (IoU >= 0.5, conf >= deploy_conf);
  vale aunque una imagen tenga cajas de grupos distintos.
- mAP del subconjunto: imágenes cuyas cajas pertenecen todas al grupo, más todas las vacías
  (para que cuenten los falsos positivos).

Salidas en <runs_dir>/<exp>/robustez/:
    robustez.csv, recall_especie_tamano_<particion>.csv, cajas_<particion>.csv, robustez.png

Uso:
    python -m p3_fauna.scripts.s07_robustez --exp B_yolov8s
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p3_fauna.src.experimentos import best_experiment, exp_dir, load_exp_config, weights
from p3_fauna.src.metrics import box_hits
from p3_fauna.src.yolo_io import class_names_from_yolo, labels_xyxy, ultralytics_device, write_data_yaml


def per_box_recall(model, boxes: pd.DataFrame, cfg: dict, device) -> pd.DataFrame:
    boxes = boxes.copy()
    boxes["detectada"] = False
    for img, grp in boxes.groupby("image"):
        r = model.predict(img, imgsz=cfg["train"]["imgsz"], conf=cfg["eval"]["deploy_conf"], device=device,
                          verbose=False)[0]
        H, W = r.orig_shape
        gt = labels_xyxy(img, W, H)
        hits = box_hits(gt, r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))
        boxes.loc[grp.index, "detectada"] = hits[grp.ann_idx.values]
    return boxes


def subset_map(model, cfg, imgs, boxes, split, col, out, device) -> pd.DataFrame:
    yolo = Path(cfg["data"]["yolo_dir"])
    names = class_names_from_yolo(yolo)
    empties = imgs[(imgs.particion == split) & (imgs.n_boxes == 0)].image.tolist()
    pure = boxes.groupby("image")[col].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None).dropna()
    rows = []
    for level, sel in pure.groupby(pure):
        lst = out / "listas" / f"{split}_{col}_{level}.txt"
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_text("\n".join(list(sel.index) + empties) + "\n", encoding="utf-8")
        data = write_data_yaml(lst.with_suffix(".yaml"), yolo, lst, lst, names)
        m = model.val(data=str(data), split="val", imgsz=cfg["train"]["imgsz"], batch=cfg["eval"]["batch"],
                      conf=cfg["eval"]["conf"], iou=cfg["eval"]["iou"], device=device, plots=False, verbose=False,
                      project=str(out), name=f"val_{split}_{col}_{level}", exist_ok=True)
        rows.append({"grupo": level, "imagenes_puras": len(sel), "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    ap.add_argument("--exp", default=None)
    args = ap.parse_args()
    base = load_config(args.config)
    exp = args.exp or best_experiment(base)[0] or base["ablation"]["reference"]
    cfg = load_exp_config(base, exp)
    cfg["eval"], cfg["robustness"] = base["eval"], base["robustness"]
    out = ensure_dir(exp_dir(cfg, exp) / "robustez")
    log = get_logger("s07", out / "robustez.log")
    device = ultralytics_device(cfg["train"].get("device", "auto"))
    rb = cfg["robustness"]
    yolo = Path(cfg["data"]["yolo_dir"])

    from ultralytics import YOLO

    model = YOLO(str(weights(cfg, exp)))
    all_boxes = pd.read_csv(yolo / "boxes_meta.csv")
    imgs = pd.read_csv(yolo / "images_meta.csv")
    all_boxes["tamaño"] = pd.cut(all_boxes.area_rel, bins=rb["size_bins"], labels=rb["size_names"]).astype(str)
    man = Path(rb.get("manual_occlusion_csv") or "")
    if man.is_file():
        m = pd.read_csv(man).rename(columns={"occlusion": "oclusion_manual"})
        all_boxes = all_boxes.merge(m[["image_id", "ann_idx", "oclusion_manual"]], on=["image_id", "ann_idx"], how="left")
        has = all_boxes.oclusion_manual.notna()
        log.info(f"Oclusión manual: {has.sum()} cajas; acuerdo con el proxy "
                 f"{(all_boxes.loc[has, 'oclusion_manual'] == all_boxes.loc[has, 'oclusion']).mean():.1%}")
        all_boxes["oclusion"] = all_boxes.oclusion_manual.fillna(all_boxes.oclusion)

    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    tabs = []
    for split in rb["splits"]:
        boxes = all_boxes[all_boxes.particion == split]
        if boxes.empty:
            continue
        sel = boxes.image.unique()
        if len(sel) > int(rb["max_images"]):
            sel = rng.choice(sel, int(rb["max_images"]), replace=False)
        rec = per_box_recall(model, boxes[boxes.image.isin(sel)], cfg, device)
        rec.to_csv(out / f"cajas_{split}.csv", index=False)
        rec.pivot_table(index="especie", columns="tamaño", values="detectada", aggfunc="mean") \
           .to_csv(out / f"recall_especie_tamano_{split}.csv")
        for col in ("tamaño", "oclusion"):
            t = (rec.groupby(col).detectada.agg(cajas="size", recall="mean")
                 .join(subset_map(model, cfg, imgs, boxes, split, col, out, device).set_index("grupo")))
            t.index.name = "grupo"
            tabs.append(t.reset_index().assign(factor=col, particion=split))
    if not tabs:
        log.info("Sin cajas para evaluar")
        return
    tab = pd.concat(tabs)[["particion", "factor", "grupo", "cajas", "recall", "imagenes_puras", "mAP50", "mAP50_95"]]
    tab.to_csv(out / "robustez.csv", index=False)
    log.info("\n" + tab.round(3).to_string(index=False))

    order = {"tamaño": rb["size_names"], "oclusion": ["bajo", "medio", "alto"]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, col in zip(axes, ("tamaño", "oclusion")):
        d = tab[tab.factor == col].pivot(index="grupo", columns="particion", values="recall")
        d.reindex([g for g in order[col] if g in d.index]).plot.bar(ax=ax, rot=0)
        ax.set_title(f"Recall por caja según {col}")
        ax.set_xlabel("")
    axes[0].set_ylabel(f"recall (conf >= {cfg['eval']['deploy_conf']}, IoU >= 0.5)")
    fig.tight_layout()
    fig.savefig(out / "robustez.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
