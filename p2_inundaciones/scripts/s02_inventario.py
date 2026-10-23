"""
Etapa 2. Inventario de la descarga de Sen1Floods11.

Cuenta los archivos de cada carpeta, verifica que cada chip de las particiones oficiales tenga
imagen S1 y etiqueta, resume los chips por evento y partición, y revisa la cabecera de una
muestra de archivos (sistema de referencia, tamaño, tipo y resolución).

Salidas:
    p2_inundaciones/reports/inventario_datos.md
    p2_inundaciones/reports/tablas/chips_por_evento.csv

Uso:
    python -m p2_inundaciones.scripts.s02_inventario
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p2_inundaciones.src.io_s1f import chip_id_from_name, event_from_chip, raw_root, read_split_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s02")
    ds = cfg["dataset"]
    root = raw_root(cfg)
    if not root.exists():
        raise FileNotFoundError(f"No existe {root}: ejecutar antes s01_descargar_sen1floods11")

    lines = ["# Inventario de datos: Sen1Floods11 v1.1", "", f"Carpeta analizada: `{root}`", "",
             "## Archivos por carpeta", "", "| Carpeta | Archivos .tif |", "|---|---:|"]
    folders = {}
    for group in (ds["hand_dir"], ds["weak_dir"]):
        for d in sorted((root / group).glob("*")):
            if d.is_dir():
                n = len(list(d.glob("*.tif")))
                folders[d.name] = n
                lines.append(f"| `{group}/{d.name}` | {n:,} |")

    splits = {k: read_split_csv(root / ds["splits_dir"] / v) for k, v in ds["split_files"].items()
              if (root / ds["splits_dir"] / v).exists()}
    lines += ["", "## Particiones oficiales (chips dorados)", "", "| Partición | Chips | Sin S1Hand | Sin LabelHand |",
              "|---|---:|---:|---:|"]
    for k, ids in splits.items():
        miss_s1 = sum(not (root / ds["hand_dir"] / "S1Hand" / f"{c}_S1Hand.tif").exists() for c in ids)
        miss_lb = sum(not (root / ds["hand_dir"] / "LabelHand" / f"{c}_LabelHand.tif").exists() for c in ids)
        lines.append(f"| {k} | {len(ids)} | {miss_s1} | {miss_lb} |")

    rows = []
    for k, ids in splits.items():
        for c in ids:
            rows.append({"evento": event_from_chip(c), "particion": k})
    weak_ids = [chip_id_from_name(p.name) for p in (root / ds["weak_dir"] / "S1Weak").glob("*.tif")]
    rows += [{"evento": event_from_chip(c), "particion": "debil"} for c in weak_ids]
    tab = pd.DataFrame(rows).pivot_table(index="evento", columns="particion", aggfunc="size", fill_value=0)
    tab["total"] = tab.sum(axis=1)
    out = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas") / "chips_por_evento.csv"
    tab.to_csv(out)
    cols = list(tab.columns)
    lines += ["", "## Chips por evento", "", "| Evento | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    lines += [f"| {ev} | " + " | ".join(str(int(v)) for v in r) + " |" for ev, r in tab.iterrows()]

    import rasterio

    lines += ["", "## Muestra de cabeceras", "", "| Archivo | Bandas | Tipo | Tamaño | CRS | Píxel (grados) | Nodata |",
              "|---|---:|---|---|---|---|---|"]
    for name in ("S1Hand", "LabelHand", "S1OtsuLabelHand", "JRCWaterHand", "S2Hand"):
        f = next((root / ds["hand_dir"] / name).glob("*.tif"), None)
        if f is not None:
            with rasterio.open(f) as r:
                lines.append(f"| `{f.name}` | {r.count} | {r.dtypes[0]} | {r.width}x{r.height} | {r.crs} | "
                             f"{r.transform.a:.4e} | {r.nodata} |")
    for name in ("S1Weak", "S1OtsuLabelWeak", "S2IndexLabelWeak"):
        f = next((root / ds["weak_dir"] / name).glob("*.tif"), None)
        if f is not None:
            with rasterio.open(f) as r:
                lines.append(f"| `{f.name}` | {r.count} | {r.dtypes[0]} | {r.width}x{r.height} | {r.crs} | "
                             f"{r.transform.a:.4e} | {r.nodata} |")

    md = Path(cfg["data"]["reports_dir"]) / "inventario_datos.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n" + tab.to_string())
    log.info(f"Carpetas: {folders}")
    log.info(f"Guardado {md}")


if __name__ == "__main__":
    main()
