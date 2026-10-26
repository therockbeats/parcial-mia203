"""
Etapa 1b. Imágenes vacías para el entrenamiento.

La partición train de CCT-20 no contiene imágenes vacías (solo imágenes con animal), pero el
enunciado pide manejar explícitamente las vacías como clase de fondo. Esta etapa las toma del
Caltech Camera Traps completo (mismo proyecto; ~70 % de sus imágenes son vacías):

1. Lee las anotaciones por imagen del dataset completo ("Image-level annotations" de LILA BC).
2. Descarta toda imagen o secuencia que ya esté en alguna partición de CCT-20 y toda cámara trans
   (sin fuga hacia validación o prueba). Las 10 cámaras de entrenamiento solo tienen 2,130 vacías
   en el dataset completo y todas están en cis_val / cis_test, de modo que las vacías provienen de
   cámaras que no forman parte de CCT-20 (si hubiera vacías libres de las cámaras de entrenamiento,
   se usarían primero).
3. Toma ``empties.n_images`` imágenes repartidas en ronda entre cámaras (muchos fondos distintos),
   las descarga, las reduce a ``empties.max_side`` px (como CCT-20) y las guarda junto a las
   imágenes de CCT-20.
4. Escribe ``train_vacias_annotations.json`` (formato COCO Camera Traps), que las etapas 1 y 2
   suman a la partición train.

Salidas:
    <raw_dir>/<annotations_subdir>/train_vacias_annotations.json
    <raw_dir>/<images_subdir>/<id>.jpg (imágenes vacías)
    p3_fauna/reports/tablas/vacias_entrenamiento.csv

Uso:
    python -m p3_fauna.scripts.s01b_vacias_entrenamiento
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p3_fauna.src.cct import CCTSplit

PATTERNS = ["caltech_images*.json", "caltech_images*.zip", "caltech_camera_traps*.json", "caltech_camera_traps*.zip"]


def find_metadata(cfg: dict) -> Path:
    p = cfg["empties"].get("full_metadata")
    if p:
        return Path(p)
    raw = Path(cfg["data"]["raw_dir"])
    for pat in PATTERNS + ["*.json", "*.zip"]:   # al final, cualquier JSON o ZIP suelto en raw_dir
        hits = sorted(raw.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No se encontró en {raw} el archivo 'Image-level annotations' del Caltech Camera Traps "
                            f"completo ({', '.join(PATTERNS)}). Descargarlo de LILA BC y dejarlo en esa carpeta.")


def read_json_any(path: Path) -> dict:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".json"))
            return json.loads(z.read(name).decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def fetch(base: str, file_name: str, retries: int = 3) -> bytes:
    rel = file_name.replace("\\", "/").lstrip("/")
    if rel.startswith("cct_images/"):
        rel = rel[len("cct_images/"):]
    if not base.startswith("http"):
        return (Path(base) / rel).read_bytes()
    url = f"{base.rstrip('/')}/{rel}"
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError(url)


def save_resized(data: bytes, dst: Path, max_side: int) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        s = max_side / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        im.save(tmp, "JPEG", quality=90)
        shutil.move(tmp, dst)
        return im.size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    cfg = load_config(ap.parse_args().config)
    log = get_logger("s01b")
    e = cfg["empties"]
    raw = Path(cfg["data"]["raw_dir"])
    img_dir = raw / cfg["data"]["images_subdir"]
    ann_dir = raw / cfg["data"]["annotations_subdir"]
    seed = int(cfg.get("seed", 42))

    # CCT-20: cámaras e imágenes / secuencias ya usadas
    cct = {n: CCTSplit(ann_dir / f, cfg["dataset"]["empty_category"]) for n, f in cfg["dataset"]["split_files"].items()}
    used_ids = {i for s in cct.values() for i in s.images}
    used_seq = {im.get("seq_id") for s in cct.values() for im in s.images.values()} - {None}
    train = cct["train"]
    train_cams = {str(im.get("location")) for im in train.images.values()}
    trans_cams = {str(im.get("location")) for n in cct if n.startswith("trans") for im in cct[n].images.values()}
    cct_cams = {str(im.get("location")) for s in cct.values() for im in s.images.values()}
    log.info(f"CCT-20: {len(train_cams)} cámaras de entrenamiento, {len(trans_cams)} trans | "
             f"imágenes a excluir: {len(used_ids):,}")

    # Dataset completo: vacías fuera de CCT-20 y nunca de cámaras trans
    meta_path = find_metadata(cfg)
    log.info(f"Anotaciones del dataset completo: {meta_path}")
    full = read_json_any(meta_path)
    empty_full = {c["id"] for c in full["categories"] if c["name"].lower() == cfg["dataset"]["empty_category"]}
    cats_by_img: dict = {}
    for a in full["annotations"]:
        cats_by_img.setdefault(a["image_id"], set()).add(a["category_id"])
    rows = []
    for im in full["images"]:
        cats = cats_by_img.get(im["id"], set())
        loc = str(im.get("location"))
        if (cats and cats <= empty_full and loc not in trans_cams
                and im["id"] not in used_ids and im.get("seq_id") not in used_seq):
            rows.append({"id": im["id"], "file_name": im["file_name"], "location": loc,
                         "date_captured": im.get("date_captured"), "seq_id": im.get("seq_id"),
                         "frame_num": im.get("frame_num"), "orig": im,
                         "origen": "camara_train" if loc in train_cams else
                                   ("camara_cct20" if loc in cct_cams else "camara_fuera_de_cct20")})
    cand = pd.DataFrame(rows)
    if cand.empty:
        raise RuntimeError("No hay imágenes vacías candidatas en el dataset completo")
    log.info("Vacías candidatas por origen:\n" + cand.origen.value_counts().to_string())
    if e.get("solo_camaras_train") and not (cand.origen == "camara_train").any():
        raise RuntimeError("Las cámaras de entrenamiento no tienen vacías fuera de CCT-20 (todas están en cis_val / "
                           "cis_test); poner empties.solo_camaras_train: false")
    if e.get("solo_camaras_train"):
        cand = cand[cand.origen == "camara_train"]
    elif (cand.origen == "camara_train").any():
        log.info("Hay vacías de las cámaras de entrenamiento: se usan primero")

    # Muestra repartida entre cámaras (ronda por cámara) para cubrir muchos fondos distintos;
    # primero las de cámaras de entrenamiento si las hubiera
    n = min(int(e["n_images"]), len(cand))
    cand = cand.sample(frac=1.0, random_state=seed)
    cand["rango"] = cand.groupby("location").cumcount()
    cand["prioridad"] = (cand.origen != "camara_train").astype(int)
    sel = cand.sort_values(["prioridad", "rango"], kind="stable").head(n)
    log.info(f"Seleccionadas {len(sel):,} vacías de {sel.location.nunique()} cámaras "
             f"(máx. {sel.location.value_counts().max()} por cámara); descargando desde {e['image_base']}")

    # Descarga y reducción
    done, failed = {}, []

    def work(r):
        dst = img_dir / Path(r.file_name).name
        if dst.exists():
            with Image.open(dst) as im:
                return r.id, im.size
        return r.id, save_resized(fetch(e["image_base"], r.file_name), dst, int(e["max_side"]))

    with ThreadPoolExecutor(int(e.get("workers", 8))) as ex:
        futs = {ex.submit(work, r): r.id for r in sel.itertuples()}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                iid, size = f.result()
                done[iid] = size
            except Exception as err:
                failed.append((futs[f], str(err)[:120]))
            if k % 250 == 0 or k == len(futs):
                log.info(f"  {k:,}/{len(futs):,} procesadas ({len(failed)} con error)")
    if failed:
        log.info(f"{len(failed)} descargas fallidas (ejemplo: {failed[0]}); se omiten. Volver a ejecutar para reintentar.")
    if len(done) < 0.9 * len(sel):
        raise RuntimeError(f"Solo se obtuvieron {len(done)} de {len(sel)} imágenes; no se escribe el JSON. Revisar la "
                           f"conexión o empties.image_base y volver a ejecutar (las ya descargadas se reutilizan).")

    # JSON adicional con las categorías de CCT-20
    empty_id = next(iter(train.empty_ids))
    ok = sel[sel.id.isin(done)]
    images, anns = [], []
    for r in ok.itertuples():
        w, h = done[r.id]
        images.append({"id": r.id, "file_name": Path(r.file_name).name, "width": w, "height": h,
                       "location": r.orig.get("location"), "date_captured": r.date_captured, "seq_id": r.seq_id,
                       "frame_num": r.frame_num, "fuente": "caltech_camera_traps_completo", "origen": r.origen})
        anns.append({"id": f"vacia_{r.id}", "image_id": r.id, "category_id": empty_id})
    out = {"info": {"description": "Imágenes vacías de las cámaras de entrenamiento de CCT-20, tomadas del Caltech "
                                   "Camera Traps completo (etapa 1b)", "metadata": meta_path.name},
           "images": images, "annotations": anns,
           "categories": [{"id": k, "name": v} for k, v in train.categories.items()]}
    (ann_dir / "train_vacias_annotations.json").write_text(json.dumps(out), encoding="utf-8")
    tab = ok.groupby(["origen", "location"]).size().rename("vacias").reset_index()
    tables = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")
    tab.to_csv(tables / "vacias_entrenamiento.csv", index=False)
    log.info("Vacías por origen:\n" + tab.groupby("origen").vacias.agg(["sum", "count"])
             .rename(columns={"sum": "imagenes", "count": "camaras"}).to_string())
    log.info(f"{len(images):,} vacías listas en {ann_dir / 'train_vacias_annotations.json'}")


if __name__ == "__main__":
    main()
