"""
Lectura de anotaciones COCO Camera Traps (CCT-20) y descriptores por caja.

Estructura del JSON:
    images:      [{id, file_name, width, height, location, date_captured, seq_id, frame_num, ...}]
    annotations: [{image_id, category_id, bbox: [x, y, w, h] en píxeles (opcional)}]
    categories:  [{id, name}]
Las imágenes vacías tienen una anotación de la categoría "empty" sin caja.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class CCTSplit:
    """Una partición de CCT-20 cargada desde uno o varios JSON (los adicionales se suman)."""

    def __init__(self, json_paths: str | Path | list, empty_name: str = "empty"):
        paths = json_paths if isinstance(json_paths, (list, tuple)) else [json_paths]
        self.images, self.categories, self.anns = {}, {}, defaultdict(list)
        for p in paths:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            self.images.update({im["id"]: im for im in d["images"]})
            for c in d["categories"]:
                if self.categories.get(c["id"], c["name"]) != c["name"]:
                    raise ValueError(f"{p}: la categoría {c['id']} no coincide con la de CCT-20")
                self.categories[c["id"]] = c["name"]
            for a in d["annotations"]:
                self.anns[a["image_id"]].append(a)
        self.empty_ids = {cid for cid, n in self.categories.items() if n.lower() == empty_name}

    def species_names(self) -> set[str]:
        return {n for cid, n in self.categories.items() if cid not in self.empty_ids}

    def classify_images(self) -> tuple[list, list, list]:
        """
        Devuelve (con_caja, vacias, ambiguas).

        Ambiguas: etiqueta de animal sin caja; no indican dónde está el animal y se excluyen.
        """
        with_box, empty, ambiguous = [], [], []
        for img_id in self.images:
            animal = [a for a in self.anns.get(img_id, []) if a["category_id"] not in self.empty_ids]
            if not animal:
                empty.append(img_id)
            elif all("bbox" in a for a in animal):
                with_box.append(img_id)
            else:
                ambiguous.append(img_id)
        return with_box, empty, ambiguous


def split_paths(cfg: dict, name: str, existing_only: bool = True) -> list[Path]:
    """JSON de una partición: el oficial de CCT-20 más los adicionales de ``dataset.extra_files``."""
    ann_dir = Path(cfg["data"]["raw_dir"]) / cfg["data"]["annotations_subdir"]
    extra = [ann_dir / f for f in (cfg["dataset"].get("extra_files") or {}).get(name, [])]
    return [ann_dir / cfg["dataset"]["split_files"][name]] + [p for p in extra if p.exists() or not existing_only]


def load_splits(cfg: dict) -> dict[str, CCTSplit]:
    return {n: CCTSplit(split_paths(cfg, n), cfg["dataset"]["empty_category"]) for n in cfg["dataset"]["split_files"]}


def class_names(splits: list[CCTSplit]) -> list[str]:
    """Lista ordenada y estable de especies (sin la categoría vacía)."""
    names: set[str] = set()
    for s in splits:
        names |= s.species_names()
    return sorted(names)


def iou_xywh(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def occlusion_proxy(box, others, width: int, height: int, margin: int = 2) -> tuple[float, bool, float]:
    """
    Proxy de oclusión: caja truncada por el borde de la imagen y solape con otras cajas.

    score = 0.5 * truncada + IoU máximo con otra caja. No captura la oclusión por vegetación,
    que debe estimarse con una muestra anotada a mano.
    """
    x, y, w, h = box
    truncated = x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin
    overlap = max((iou_xywh(box, o) for o in others), default=0.0)
    return 0.5 * float(truncated) + overlap, bool(truncated), float(overlap)


def occlusion_level(score: float, bins=(0.15, 0.5)) -> str:
    return "bajo" if score < bins[0] else ("medio" if score < bins[1] else "alto")
