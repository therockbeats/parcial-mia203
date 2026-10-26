"""
Etapa 1. Inventario de CCT-20.

Para cada partición cuenta imágenes con caja, vacías y ambiguas (animal sin caja), cámaras,
cajas por especie y archivos de imagen presentes en disco; verifica que las cámaras trans no
aparezcan en entrenamiento.

Salidas en p3_fauna/reports/:
    inventario_datos.md, tablas/inventario_particiones.csv, tablas/cajas_por_especie.csv,
    tablas/camaras_por_particion.csv

Uso:
    python -m p3_fauna.scripts.s01_inventario
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p3_fauna.src.cct import class_names, load_splits, split_paths
from p3_fauna.src.monitoreo import df_to_md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    cfg = load_config(ap.parse_args().config)
    log = get_logger("s01")
    raw = Path(cfg["data"]["raw_dir"])
    img_dir = raw / cfg["data"]["images_subdir"]
    ann_dir = raw / cfg["data"]["annotations_subdir"]
    reports = Path(cfg["data"]["reports_dir"])
    tables = ensure_dir(reports / "tablas")

    splits = load_splits(cfg)
    names = class_names(list(splits.values()))
    rows, boxes, cams = [], [], []
    for name, s in splits.items():
        with_box, empty, amb = s.classify_images()
        locs = {str(im.get("location")) for im in s.images.values()}
        on_disk = sum((img_dir / im["file_name"]).exists() for im in s.images.values())
        n_box = sum(len([a for a in s.anns[i] if "bbox" in a]) for i in with_box)
        rows.append({"particion": name, "imagenes": len(s.images), "con_caja": len(with_box), "vacias": len(empty),
                     "ambiguas": len(amb), "pct_vacias": round(100 * len(empty) / max(len(s.images), 1), 1),
                     "cajas": n_box, "camaras": len(locs), "en_disco": on_disk})
        for i in with_box:
            for a in s.anns[i]:
                if "bbox" in a:
                    boxes.append({"particion": name, "especie": s.categories[a["category_id"]]})
        cams += [{"particion": name, "camara": c} for c in sorted(locs)]

    inv = pd.DataFrame(rows)
    inv.to_csv(tables / "inventario_particiones.csv", index=False)
    per_sp = (pd.DataFrame(boxes).groupby(["especie", "particion"]).size().unstack(fill_value=0)
              .reindex(index=names, columns=list(splits), fill_value=0))
    per_sp["total"] = per_sp.sum(axis=1)
    per_sp = per_sp.sort_values("total", ascending=False)
    per_sp.to_csv(tables / "cajas_por_especie.csv")
    cam = pd.DataFrame(cams)
    cam.to_csv(tables / "camaras_por_particion.csv", index=False)

    train_cams = set(cam[cam.particion == "train"].camara)
    checks = []
    for name in splits:
        c = set(cam[cam.particion == name].camara)
        checks.append(f"- `{name}`: {len(c)} cámaras, {len(c & train_cams)} compartidas con entrenamiento")

    md = ["# Inventario de datos: Caltech Camera Traps (CCT-20)", "",
          f"Imágenes: `{img_dir}`", f"Anotaciones: `{ann_dir}`", "",
          "Archivos por partición: " + "; ".join(f"`{n}`: " + ", ".join(x.name for x in split_paths(cfg, n))
                                                 for n in splits), "",
          "## Imágenes por partición", "", df_to_md(inv), "",
          "Ambiguas: imágenes con etiqueta de animal sin caja; se excluyen de la detección. La partición "
          "train de CCT-20 no trae vacías; las de `train_vacias_annotations.json` provienen del Caltech "
          "Camera Traps completo, de cámaras ajenas a CCT-20 y nunca de cámaras trans (etapa 1b).", "",
          "## Cámaras (cis = vistas en entrenamiento, trans = no vistas)", "", *checks, "",
          f"## Cajas por especie ({len(names)} especies)", "", df_to_md(per_sp)]
    (reports / "inventario_datos.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("\n" + inv.to_string(index=False))
    log.info("\n" + "\n".join(checks))
    log.info(f"Inventario en {reports / 'inventario_datos.md'}")


if __name__ == "__main__":
    main()
