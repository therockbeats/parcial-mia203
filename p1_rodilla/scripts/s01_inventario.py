"""
PASO 1. Inventario de la descarga de Kaggle (RSNA Knee Abnormality Detection).

Objetivo: saber EXACTAMENTE qué trae la descarga antes de escribir nada más:
nombres de archivos, columnas de cada CSV, cómo están organizadas las carpetas
de DICOM y qué metadatos traen las imágenes.

Genera dos archivos DENTRO del repositorio (son pequeños, se suben a GitHub):
    p1_rodilla/reports/inventario_datos.md    <- legible para humanos
    p1_rodilla/reports/inventario_datos.json  <- el mismo contenido para programas
Con ese reporte se ajusta la sección "schema" de p1_rodilla/config.yaml.

Uso (desde la raíz del repositorio):
    python -m p1_rodilla.scripts.s01_inventario
Opcional:  --max-dicom 40   (cuántas series muestrear para leer cabeceras)

Nota de privacidad: el reporte recorta los textos a 160 caracteres. Mantén el
repositorio de GitHub en modo PRIVADO hasta la entrega (reglas de Kaggle).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import Timer, get_logger

TABULAR = {".csv", ".parquet", ".tsv", ".json", ".jsonl", ".txt"}
DICOM_TAGS = [
    "Modality", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "SeriesDescription", "ProtocolName", "SequenceName", "ScanningSequence",
    "ScanOptions", "RepetitionTime", "EchoTime", "InversionTime", "SliceThickness",
    "SpacingBetweenSlices", "PixelSpacing", "Rows", "Columns",
    "ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber",
    "PhotometricInterpretation", "BitsStored", "InstitutionName",
]


def short(value, n: int = 160) -> str:
    """Convierte cualquier valor a texto de máximo n caracteres (para el reporte)."""
    s = str(value).replace("\n", " ").replace("|", "/")
    return s if len(s) <= n else s[:n] + "…"


def walk_tree(root: Path, log) -> dict:
    """
    Recorre TODA la carpeta una sola vez y cuenta:
      - archivos por extensión,
      - cuántas carpetas hay en cada nivel de profundidad,
      - ejemplos de rutas de archivos por nivel (para deducir el patrón de carpetas).
    os.scandir es mucho más rápido que glob para ~800 mil archivos.
    """
    ext_count, ext_bytes = Counter(), Counter()
    dirs_per_depth = Counter()
    files_per_depth = Counter()
    examples = defaultdict(list)
    top_level = []
    stack = [(root, 0)]
    n_files = 0
    while stack:
        current, depth = stack.pop()
        try:
            entries = list(os.scandir(current))
        except PermissionError:
            continue
        for e in entries:
            if depth == 0:
                top_level.append((e.name, "carpeta" if e.is_dir() else "archivo"))
            if e.is_dir(follow_symlinks=False):
                dirs_per_depth[depth + 1] += 1
                stack.append((Path(e.path), depth + 1))
            else:
                n_files += 1
                ext = Path(e.name).suffix.lower() or "(sin extensión)"
                ext_count[ext] += 1
                try:
                    ext_bytes[ext] += e.stat().st_size
                except OSError:
                    pass
                files_per_depth[depth + 1] += 1
                if len(examples[depth + 1]) < 5:
                    examples[depth + 1].append(str(Path(e.path).relative_to(root)))
                if n_files % 100_000 == 0:
                    log.info(f"   ... {n_files:,} archivos recorridos")
    return {
        "n_files": n_files,
        "top_level": sorted(top_level),
        "ext_count": dict(ext_count.most_common()),
        "ext_gb": {k: round(v / 1024**3, 2) for k, v in ext_bytes.most_common()},
        "dirs_per_depth": dict(sorted(dirs_per_depth.items())),
        "files_per_depth": dict(sorted(files_per_depth.items())),
        "examples_per_depth": dict(sorted(examples.items())),
    }


def describe_table(path: Path) -> dict:
    """Lee un CSV/Parquet y resume columnas, tipos, nulos y ejemplos."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t", low_memory=False)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        df = pd.read_csv(path, low_memory=False)
    cols = []
    for c in df.columns:
        s = df[c]
        info = {
            "columna": str(c),
            "tipo": str(s.dtype),
            "no_nulos": int(s.notna().sum()),
            "unicos": int(s.nunique(dropna=True)),
            "ejemplos": [short(v, 60) for v in s.dropna().unique()[:4]],
        }
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            info["min"], info["max"] = float(s.min()), float(s.max())
        cols.append(info)
    return {"archivo": str(path), "filas": int(len(df)), "columnas": cols,
            "primeras_filas": [{k: short(v) for k, v in r.items()} for r in df.head(3).to_dict("records")]}


def find_series_dirs(images_root: Path, ext: str, limit: int, seed: int = 0) -> list[Path]:
    """Busca carpetas que contienen archivos DICOM (hasta 'limit', elegidas al azar)."""
    candidates = []
    for dirpath, _, filenames in os.walk(images_root):
        if any(f.lower().endswith(ext) for f in filenames):
            candidates.append(Path(dirpath))
        if len(candidates) >= limit * 20:
            break
    random.Random(seed).shuffle(candidates)
    return candidates[:limit]


def read_dicom_sample(series_dirs: list[Path], ext: str, images_root: Path) -> list[dict]:
    """Lee la cabecera (sin píxeles) del corte central de cada serie y prueba decodificar uno."""
    import pydicom

    rows = []
    for sd in series_dirs:
        files = sorted(f for f in sd.iterdir() if f.suffix.lower() == ext)
        if not files:
            continue
        f = files[len(files) // 2]
        row = {"carpeta": str(sd.relative_to(images_root)), "n_archivos": len(files)}
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            for t in DICOM_TAGS:
                row[t] = short(getattr(ds, t, ""), 80)
            row["TransferSyntax"] = str(ds.file_meta.TransferSyntaxUID.name) if hasattr(ds, "file_meta") else ""
            row["StudyInstanceUID"] = short(getattr(ds, "StudyInstanceUID", ""), 70)
            row["SeriesInstanceUID"] = short(getattr(ds, "SeriesInstanceUID", ""), 70)
        except Exception as e:
            row["error_cabecera"] = short(e)
        rows.append(row)
    # Prueba de decodificación de píxeles en las dos primeras series
    for row, sd in zip(rows[:2], series_dirs[:2]):
        try:
            files = sorted(f for f in sd.iterdir() if f.suffix.lower() == ext)
            arr = pydicom.dcmread(files[len(files) // 2]).pixel_array
            row["prueba_pixeles"] = f"OK shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()}"
        except Exception as e:
            row["prueba_pixeles"] = f"ERROR: {short(e)}"
    return rows


def to_markdown(inv: dict) -> str:
    """Arma el reporte legible en Markdown."""
    L = ["# Inventario de datos: RSNA Knee Abnormality Detection", ""]
    L.append(f"Carpeta analizada: `{inv['raw_dir']}`  ")
    t = inv["arbol"]
    L.append(f"Archivos totales: **{t['n_files']:,}**")
    L += ["", "## Contenido del primer nivel", ""]
    L += [f"- `{n}` ({k})" for n, k in t["top_level"]]
    L += ["", "## Archivos por extensión", "", "| Extensión | Cantidad | GB |", "|---|---:|---:|"]
    L += [f"| {e} | {c:,} | {t['ext_gb'].get(e, 0)} |" for e, c in t["ext_count"].items()]
    L += ["", "## Estructura de carpetas (por profundidad)", "", "| Nivel | Carpetas | Archivos | Ejemplos de archivos |", "|---:|---:|---:|---|"]
    depths = sorted(set(t["dirs_per_depth"]) | set(t["files_per_depth"]), key=int)
    for d in depths:
        ex = "<br>".join(f"`{x}`" for x in t["examples_per_depth"].get(d, [])[:3])
        L.append(f"| {d} | {t['dirs_per_depth'].get(d, 0):,} | {t['files_per_depth'].get(d, 0):,} | {ex} |")
    L += ["", "## Tablas (CSV / Parquet / JSON)", ""]
    for tab in inv["tablas"]:
        if "error" in tab:
            L += [f"### `{tab['archivo']}`", f"Error al leer: {tab['error']}", ""]
            continue
        L += [f"### `{Path(tab['archivo']).name}`  ({tab['filas']:,} filas)", "",
              "| Columna | Tipo | No nulos | Únicos | Rango | Ejemplos |", "|---|---|---:|---:|---|---|"]
        for c in tab["columnas"]:
            rng = f"{c['min']:.6g} a {c['max']:.6g}" if "min" in c else ""
            L.append(f"| `{c['columna']}` | {c['tipo']} | {c['no_nulos']:,} | {c['unicos']:,} | {rng} | {' · '.join(c['ejemplos'])} |")
        L.append("")
    L += ["## Muestra de cabeceras DICOM", ""]
    if inv["dicom"]:
        keys = ["carpeta", "n_archivos", "SeriesDescription", "ImageOrientationPatient", "ScanOptions",
                "RepetitionTime", "EchoTime", "SliceThickness", "PixelSpacing", "Rows", "Columns",
                "Manufacturer", "MagneticFieldStrength", "TransferSyntax", "prueba_pixeles", "error_cabecera"]
        L += ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        for r in inv["dicom"]:
            L.append("| " + " | ".join(short(r.get(k, ""), 60) for k in keys) + " |")
    else:
        L.append("No se encontraron DICOM en la carpeta de imágenes indicada en el config.")
    L += ["", "## Siguiente paso", "",
          "La sección `schema:` de `p1_rodilla/config.yaml` debe coincidir con los nombres de "
          "archivos y columnas descritos en este inventario."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--max-dicom", type=int, default=40, help="series a muestrear")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger("s01")
    raw = Path(cfg["data"]["raw_dir"])
    if not raw.exists():
        raise SystemExit(f"No existe {raw}. Revisa data_root en configs/paths.local.yaml")

    inv = {"raw_dir": str(raw)}
    with Timer("Recorrido de carpetas", log):
        inv["arbol"] = walk_tree(raw, log)

    # Tablas: todo CSV/Parquet/JSON en los dos primeros niveles
    inv["tablas"] = []
    # Solo nivel 1 y 2 (las tablas de Kaggle están arriba; no recorremos los 800 mil DICOM otra vez)
    level1 = [p for p in raw.iterdir()]
    level2 = [q for p in level1 if p.is_dir() and p.name != cfg["schema"].get("images_subdir")
              for q in p.iterdir()]
    for p in sorted(level1 + level2):
        if p.is_file() and p.suffix.lower() in TABULAR - {".txt"}:
            log.info(f"Leyendo tabla {p.name}")
            try:
                inv["tablas"].append(describe_table(p))
            except Exception as e:
                inv["tablas"].append({"archivo": str(p), "error": short(e)})

    # DICOM: carpeta de imágenes indicada en el config (o toda la descarga si no existe)
    images_root = raw / cfg["schema"].get("images_subdir", "")
    if not images_root.exists():
        log.warning(f"No existe {images_root}; busco DICOM en toda la descarga")
        images_root = raw
    ext = cfg["schema"].get("dicom_ext", ".dcm").lower()
    with Timer("Muestreo de cabeceras DICOM", log):
        dirs = find_series_dirs(images_root, ext, args.max_dicom, cfg.get("seed", 0))
        inv["dicom"] = read_dicom_sample(dirs, ext, images_root)

    out_dir = Path(cfg["data"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inventario_datos.json").write_text(json.dumps(inv, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "inventario_datos.md").write_text(to_markdown(inv), encoding="utf-8")
    log.info(f"Listo. Revisa {out_dir / 'inventario_datos.md'}")


if __name__ == "__main__":
    main()
