"""
Etapa 4. Evaluación de un experimento.

Para cada partición de ``eval.splits`` (todas las imágenes vacías incluidas) calcula con
Ultralytics mAP@0.5, mAP@0.5:0.95, precisión y recall globales y por especie, y la matriz de
confusión. En las particiones de prueba evalúa además el uso real con ``deploy_conf``: presencia
de fauna por imagen (precisión, recall, falsos positivos en imágenes vacías) y especie correcta
(la detección más confiable pertenece a una especie presente). La brecha de generalización es la
diferencia cis - trans en cada métrica.

Salidas en <runs_dir>/<exp>/:
    metricas.csv, por_especie.csv, por_imagen_<particion>.csv, summary.json, eval.log,
    confusion_<particion>.png, val_<particion>/ (curvas de Ultralytics)

Uso:
    python -m p3_fauna.scripts.s04_evaluar --exp B_yolov8s
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import get_logger
from p3_fauna.src.experimentos import exp_dir, load_exp_config, weights
from p3_fauna.src.metrics import presence_scores
from p3_fauna.src.yolo_io import predict_paths, read_labels, ultralytics_device


def presence_table(model, images: list[str], cfg: dict, device) -> pd.DataFrame:
    rows = []
    res = predict_paths(model, images, int(cfg["eval"]["batch"]), imgsz=cfg["train"]["imgsz"],
                        conf=cfg["eval"]["deploy_conf"], device=device)
    for path, r in res:
        gt = set(read_labels(path)[:, 0].astype(int).tolist())
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        top = int(cls[conf.argmax()]) if len(cls) else -1
        rows.append({"image": path, "hay_animal": bool(gt), "detecta_animal": bool(len(cls)),
                     "n_detecciones": int(len(cls)), "especie_top": r.names.get(top, "") if top >= 0 else "",
                     "conf_top": float(conf.max()) if len(conf) else np.nan,
                     "especie_correcta": top in gt})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    ap.add_argument("--exp", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    base = load_config(args.config)
    cfg = load_exp_config(base, args.exp, args.set)
    cfg["eval"] = base["eval"]
    out = exp_dir(cfg, args.exp)
    log = get_logger("s04", out / "eval.log")
    yolo = Path(cfg["data"]["yolo_dir"])
    device = ultralytics_device(cfg["train"].get("device", "auto"))

    from ultralytics import YOLO

    model = YOLO(str(weights(cfg, args.exp)))
    boxes = pd.read_csv(yolo / "boxes_meta.csv")
    n_train = boxes[boxes.particion == "train"].groupby("especie").size()
    e = cfg["eval"]
    rows, per_cls, summary = [], [], {"exp": args.exp, "deploy_conf": e["deploy_conf"]}
    for split in e["splits"]:
        data = yolo / f"eval_{split}.yaml"
        if not data.exists():
            log.info(f"{split}: sin {data.name}; se omite")
            continue
        m = model.val(data=str(data), split="val", imgsz=cfg["train"]["imgsz"], batch=e["batch"], conf=e["conf"],
                      iou=e["iou"], device=device, plots=bool(e.get("plots", True)), project=str(out),
                      name=f"val_{split}", exist_ok=True, verbose=False)
        row = {"particion": split, "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
               "precision": float(m.box.mp), "recall": float(m.box.mr)}
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            name = m.names[int(c)]
            per_cls.append({"particion": split, "especie": name, "P": p, "R": r, "AP50": ap50, "AP50_95": ap,
                            "cajas_prueba": int(((boxes.particion == split) & (boxes.especie == name)).sum()),
                            "cajas_train": int(n_train.get(name, 0))})
        cm = out / f"val_{split}" / "confusion_matrix_normalized.png"
        if cm.exists():
            shutil.copy2(cm, out / f"confusion_{split}.png")
        if "test" in split:
            imgs = [x for x in (yolo / "lists" / f"{split}.txt").read_text(encoding="utf-8").splitlines() if x]
            pt = presence_table(model, imgs, cfg, device)
            pt.to_csv(out / f"por_imagen_{split}.csv", index=False)
            row.update(presence_scores(pt))
        rows.append(row)
        summary.update({f"map50_{split}": row["mAP50"], f"map50_95_{split}": row["mAP50_95"]})
        if "fp_en_vacias" in row:
            summary.update({f"fp_vacias_{split}": row["fp_en_vacias"], f"recall_presencia_{split}": row["recall_presencia"]})

    met = pd.DataFrame(rows)
    idx = met.set_index("particion")
    for a, b in [("cis_val", "trans_val"), ("cis_test", "trans_test")]:
        if a in idx.index and b in idx.index:
            num = idx.select_dtypes("number")
            gap = (num.loc[a] - num.loc[b]).to_dict()
            met = pd.concat([met, pd.DataFrame([{"particion": f"brecha {a} - {b}", **gap}])], ignore_index=True)
            summary[f"brecha_map50_{a.split('_')[1]}"] = gap["mAP50"]
            summary[f"brecha_map50_95_{a.split('_')[1]}"] = gap["mAP50_95"]
    met.to_csv(out / "metricas.csv", index=False)
    pc = pd.DataFrame(per_cls)
    if len(pc):
        wide = pc.pivot_table(index="especie", columns="particion", values=["AP50", "AP50_95"])
        wide.columns = [f"{m}_{s}" for m, s in wide.columns]
        wide["cajas_train"] = n_train.reindex(wide.index).fillna(0).astype(int)
        if {"AP50_cis_test", "AP50_trans_test"} <= set(wide.columns):
            wide["brecha_AP50_test"] = wide["AP50_cis_test"] - wide["AP50_trans_test"]
        wide.sort_values("cajas_train", ascending=False).to_csv(out / "por_especie.csv")
        pc.to_csv(out / "por_especie_largo.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cols = [c for c in ["particion", "mAP50", "mAP50_95", "precision", "recall", "recall_presencia", "fp_en_vacias",
                        "especie_correcta"] if c in met]
    log.info("\n" + met[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
