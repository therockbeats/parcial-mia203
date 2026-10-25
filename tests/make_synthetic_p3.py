"""
Datos sintéticos con la estructura de CCT-20 (LILA BC).

Genera imágenes JPEG con un fondo fijo por cámara y animales dibujados como figuras de color por
especie, y los cinco JSON en formato COCO Camera Traps (train, cis_val, trans_val, cis_test,
trans_test) con imágenes vacías, imágenes ambiguas (animal sin caja), secuencias y fechas. La
resolución declarada en el JSON es el doble de la real, como prueba del reescalado de cajas.

Uso:
    python tests/make_synthetic_p3.py --out <data_root>
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 320, 240
SPECIES = {1: ("opossum", (200, 60, 60)), 6: ("bobcat", (60, 160, 60)), 9: ("coyote", (60, 60, 200))}
CATEGORIES = [{"id": 30, "name": "empty"}] + [{"id": k, "name": v[0]} for k, v in SPECIES.items()]
SPLITS = {"train": (["10", "20", "30"], 90), "cis_val": (["10", "20"], 24), "trans_val": (["70"], 24),
          "cis_test": (["10", "30"], 30), "trans_test": (["80", "90"], 60)}


def background(cam: str) -> np.ndarray:
    rng = np.random.default_rng(int(cam))
    base = rng.integers(60, 180, 3)
    img = np.clip(base + rng.normal(0, 18, (H, W, 3)), 0, 255)
    img[int(H * 0.6):] *= 0.7    # "suelo"
    return img.astype(np.uint8)


def make_split(name: str, cams: list[str], n: int, img_dir: Path, rng) -> dict:
    images, anns = [], []
    t0 = datetime(2013, 4, 1, 6, 0)
    for i in range(n):
        cam = cams[i % len(cams)]
        iid = f"{name}_{i:04d}"
        seq = f"seq_{name}_{i // 3}"
        ts = t0 + timedelta(minutes=17 * (i // 3) + i % 3)
        img = Image.fromarray(background(cam))
        draw = ImageDraw.Draw(img)
        r = rng.random()
        if r < 0.45:
            anns.append({"id": f"a{iid}", "image_id": iid, "category_id": 30})
        else:
            for _ in range(1 if rng.random() < 0.8 else 2):
                cid = int(rng.choice([1, 1, 1, 6, 9]))
                w, h = rng.integers(25, 120), rng.integers(20, 90)
                x, y = rng.integers(0, W - w), rng.integers(0, H - h)
                draw.ellipse([x, y, x + w, y + h], fill=SPECIES[cid][1])
                a = {"id": f"a{iid}_{len(anns)}", "image_id": iid, "category_id": cid}
                if r < 0.95:   # el 5 % restante: animal sin caja (ambiguo)
                    a["bbox"] = [float(2 * x), float(2 * y), float(2 * w), float(2 * h)]
                anns.append(a)
        img.save(img_dir / f"{iid}.jpg", quality=90)
        images.append({"id": iid, "file_name": f"{iid}.jpg", "width": 2 * W, "height": 2 * H, "location": int(cam),
                       "date_captured": ts.strftime("%Y-%m-%d %H:%M:%S"), "seq_id": seq, "frame_num": i % 3,
                       "seq_num_frames": 3})
    return {"info": {"description": "CCT-20 sintético"}, "images": images, "annotations": anns, "categories": CATEGORIES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="data_root")
    args = ap.parse_args()
    raw = Path(args.out) / "p3_cct20" / "raw"
    img_dir, ann_dir = raw / "eccv_18_all_images_sm", raw / "eccv_18_annotation_files"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, (cams, n) in SPLITS.items():
        d = make_split(name, cams, n, img_dir, rng)
        (ann_dir / f"{name}_annotations.json").write_text(json.dumps(d), encoding="utf-8")
        print(f"{name}: {n} imágenes, cámaras {cams}")


if __name__ == "__main__":
    main()
