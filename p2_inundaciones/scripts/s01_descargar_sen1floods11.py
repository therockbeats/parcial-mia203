"""
Etapa 1. Descarga de Sen1Floods11 v1.1 desde el bucket público de Google Cloud Storage.

Lista los objetos de cada prefijo de ``dataset.download_prefixes`` con la API JSON pública
(sin credenciales ni gsutil) y descarga en paralelo los que falten o difieran en tamaño,
conservando la estructura del bucket bajo ``<raw_dir>``.

Salida:
    <raw_dir>/v1.1/...

Uso:
    python -m p2_inundaciones.scripts.s01_descargar_sen1floods11 [--hilos 16] [--sin-s2]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from common.config import REPO_ROOT, load_config
from common.utils import get_logger

API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
MEDIA = "https://storage.googleapis.com/{bucket}/{name}"


def list_objects(bucket: str, prefix: str) -> list[dict]:
    """Objetos (nombre y tamaño) bajo un prefijo, recorriendo todas las páginas."""
    items, token = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": 1000}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(API.format(bucket=bucket) + "?" + urllib.parse.urlencode(q), timeout=60) as r:
            d = json.load(r)
        items += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            return items


def download(bucket: str, name: str, size: int, dest: Path, retries: int = 4) -> str:
    """Descarga un objeto si falta o su tamaño no coincide."""
    if dest.exists() and dest.stat().st_size == size:
        return "existente"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = MEDIA.format(bucket=bucket, name=urllib.parse.quote(name))
    for k in range(retries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return "descargado"
        except Exception as e:
            if k == retries - 1:
                return f"error: {e}"
            time.sleep(2 ** k)
    return "error"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--hilos", type=int, default=16)
    ap.add_argument("--sin-s2", action="store_true", help="omitir S2Hand (solo necesario para la fusión óptica)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s01")
    ds = cfg["dataset"]
    raw = Path(cfg["data"]["raw_dir"])
    prefixes = [p for p in ds["download_prefixes"] if not (args.sin_s2 and "S2Hand" in p)]

    objs = []
    for p in prefixes:
        full = f"{ds['version_prefix']}/{p}"
        found = [o for o in list_objects(ds["bucket"], full) if o["name"].endswith((".tif", ".csv"))]
        log.info(f"{full}: {len(found)} archivos, {sum(int(o['size']) for o in found)/1e9:.2f} GB")
        objs += found
    log.info(f"Total: {len(objs)} archivos, {sum(int(o['size']) for o in objs)/1e9:.2f} GB")

    status = {}
    with ThreadPoolExecutor(max_workers=args.hilos) as ex:
        futs = {ex.submit(download, ds["bucket"], o["name"], int(o["size"]), raw / o["name"]): o["name"] for o in objs}
        for f in tqdm(as_completed(futs), total=len(futs), desc="descarga"):
            s = f.result()
            status[futs[f]] = s
    errors = {k: v for k, v in status.items() if v.startswith("error")}
    log.info(f"Descargados: {sum(v == 'descargado' for v in status.values())} | "
             f"existentes: {sum(v == 'existente' for v in status.values())} | errores: {len(errors)}")
    for k, v in list(errors.items())[:20]:
        log.warning(f"{k}: {v}")
    if errors:
        log.warning("Volver a ejecutar la etapa para reintentar los archivos con error.")


if __name__ == "__main__":
    main()
