"""
Prueba de extremo a extremo de la Pregunta 3 con datos sintéticos.

    python tests/smoke_test_p3.py

Ejecuta el pipeline completo con una configuración reducida (YOLOv8n sin pesos preentrenados,
1 época, 160 px, CPU) en un directorio temporal, sin acceder a los datos reales ni a la red.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def run(args: list[str], env: dict) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=REPO, env=env)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mia203_p3_smoke_"))
    paths = tmp / "paths.yaml"
    paths.write_text(yaml.safe_dump({"data_root": str(tmp / "data"), "runs_root": str(tmp / "runs")}))
    env = dict(os.environ, MIA203_PATHS=str(paths), YOLO_OFFLINE="1")
    run([str(REPO / "tests" / "make_synthetic_p3.py"), "--out", str(tmp / "data")], env)

    cfg = yaml.safe_load((REPO / "p3_fauna" / "config.yaml").read_text(encoding="utf-8"))
    cfg["data"]["reports_dir"] = str(tmp / "reports")
    cfg["train"].update({"model": "yolov8n.yaml", "pretrained": False, "epochs": 1, "imgsz": 160, "batch": 4,
                         "workers": 0, "device": "cpu", "plots": False, "amp": False, "close_mosaic": 0})
    cfg["eval"].update({"batch": 4, "plots": False, "deploy_conf": 0.001})
    cfg["robustness"]["max_images"] = 20
    raw = tmp / "data" / "p3_cct20" / "raw"
    cfg["empties"].update({"full_metadata": str(raw / "caltech_images_sintetico.json"),
                           "image_base": str(raw / "cct_images_full"), "n_images": 20, "max_side": 320,
                           "workers": 4})
    cfg["background"].update({"n_images": 6, "n_figures": 2})
    cfg["ablation"]["extra_experiments"] = [{"name": "D_con_rfs", "set": ["sampling.repeat_factor_sampling=true"]}]
    cfg_path = tmp / "config_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    run(["-m", "p3_fauna.run_all", "--config", str(cfg_path), "--extras"], env)
    vac = json.loads((raw / "eccv_18_annotation_files" / "train_vacias_annotations.json").read_text(encoding="utf-8"))
    ids = {im["id"] for im in vac["images"]}
    assert len(ids) == 20 and not any(int(i.split("_")[-1]) < 5 for i in ids), "exclusión de secuencias de CCT-20"
    assert all(im["location"] in (10, 20, 30) for im in vac["images"]), "vacías fuera de las cámaras de train"
    assert all(max(im["width"], im["height"]) <= 320 for im in vac["images"]), "vacías sin reducir"
    final = tmp / "reports" / "resultados"
    assert any(final.glob("*/monitoreo/reporte_camara_*.md")), "falta el reporte de monitoreo"
    assert any(final.glob("*/robustez/robustez.csv")), "falta la tabla de robustez"
    assert any(final.glob("*/fondo/resumen.csv")), "falta el resumen de dependencia del fondo"
    print(f"\nPRUEBA DE HUMO SUPERADA. Archivos temporales en {tmp}.")


if __name__ == "__main__":
    main()
