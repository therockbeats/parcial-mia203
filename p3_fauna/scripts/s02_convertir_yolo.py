"""
Etapa 2. Conversión de COCO Camera Traps a formato YOLO.

Cada imagen se enlaza (sin duplicar espacio) en <yolo_dir>/images/<particion>/ y sus cajas se
escriben en <yolo_dir>/labels/<particion>/ como ``clase xc yc w h`` normalizados. Las imágenes
vacías no tienen archivo de etiquetas: Ultralytics las trata como fondo. Las imágenes con animal
sin caja se excluyen. Las cajas se reescalan si la resolución del archivo difiere de la del JSON.

Se conservan todas las vacías; la proporción usada en entrenamiento se decide por experimento
en la etapa 3. También se calculan, por caja, el área relativa y el proxy de oclusión.

Salidas en <yolo_dir>/:
    images/, labels/, lists/<particion>.txt, eval_<particion>.yaml, names.yaml,
    images_meta.csv (una fila por imagen), boxes_meta.csv (una fila por caja)

Uso:
    python -m p3_fauna.scripts.s02_convertir_yolo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p3_fauna.src.cct import CCTSplit, class_names, occlusion_level, occlusion_proxy
from p3_fauna.src.yolo_io import label_path, link_file, write_data_yaml


def convert_split(name: str, split: CCTSplit, cls_index: dict, cfg: dict, log) -> tuple[list, list]:
    img_dir = Path(cfg["data"]["raw_dir"]) / cfg["data"]["images_subdir"]
    out = Path(cfg["data"]["yolo_dir"])
    occ_bins = cfg["robustness"]["occ_bins"]
    with_box, empty, ambiguous = split.classify_images()
    img_rows, box_rows, entries, missing = [], [], [], 0
    for img_id in with_box + empty:
        im = split.images[img_id]
        src = img_dir / im["file_name"]
        if not src.exists():
            missing += 1
            continue
        dst = (out / "images" / name / Path(im["file_name"]).name).absolute()
        link_file(src, dst, cfg["dataset"]["link_mode"])
        entries.append(str(dst))
        with Image.open(src) as pil:
            W, H = pil.size
        sx, sy = W / (im.get("width") or W), H / (im.get("height") or H)
        anns = [a for a in split.anns[img_id] if "bbox" in a and split.categories[a["category_id"]] in cls_index]
        boxes = [[a["bbox"][0] * sx, a["bbox"][1] * sy, a["bbox"][2] * sx, a["bbox"][3] * sy] for a in anns]
        lines, species = [], set()
        for k, (a, b) in enumerate(zip(anns, boxes)):
            cname = split.categories[a["category_id"]]
            x, y, w, h = b
            lines.append(f"{cls_index[cname]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            species.add(cname)
            score, trunc, ov = occlusion_proxy(b, boxes[:k] + boxes[k + 1:], W, H)
            box_rows.append({"particion": name, "image_id": img_id, "image": str(dst), "ann_idx": k, "especie": cname,
                             "area_rel": (w * h) / (W * H), "truncada": trunc, "solape": ov,
                             "oclusion_score": score, "oclusion": occlusion_level(score, occ_bins)})
        lbl = label_path(dst)
        if lines:
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif lbl.exists():
            lbl.unlink()
        img_rows.append({"particion": name, "image_id": img_id, "image": str(dst), "camara": str(im.get("location")),
                         "date_captured": im.get("date_captured"), "seq_id": im.get("seq_id"),
                         "frame_num": im.get("frame_num"), "ancho": W, "alto": H, "n_boxes": len(lines),
                         "species": ";".join(sorted(species)) or "empty"})
    lists = ensure_dir(out / "lists")
    (lists / f"{name}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    log.info(f"[{name}] con caja {len(with_box):,} | vacías {len(empty):,} | ambiguas excluidas {len(ambiguous):,} "
             f"| sin archivo {missing:,} | en la lista {len(entries):,}")
    return img_rows, box_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    cfg = load_config(ap.parse_args().config)
    log = get_logger("s02")
    ann_dir = Path(cfg["data"]["raw_dir"]) / cfg["data"]["annotations_subdir"]
    out = ensure_dir(cfg["data"]["yolo_dir"])
    splits = {n: CCTSplit(ann_dir / f, cfg["dataset"]["empty_category"]) for n, f in cfg["dataset"]["split_files"].items()}
    names = class_names(list(splits.values()))
    cls_index = {n: i for i, n in enumerate(names)}
    log.info(f"{len(names)} especies: {', '.join(names)}")
    (out / "names.yaml").write_text(yaml.safe_dump({i: n for i, n in enumerate(names)}, allow_unicode=True),
                                    encoding="utf-8")

    imgs, boxes = [], []
    for name, split in splits.items():
        a, b = convert_split(name, split, cls_index, cfg, log)
        imgs += a
        boxes += b
        write_data_yaml(out / f"eval_{name}.yaml", out, f"lists/{name}.txt", f"lists/{name}.txt", names)
    pd.DataFrame(imgs).to_csv(out / "images_meta.csv", index=False)
    pd.DataFrame(boxes).to_csv(out / "boxes_meta.csv", index=False)
    log.info(f"Conjunto YOLO en {out}")


if __name__ == "__main__":
    main()
