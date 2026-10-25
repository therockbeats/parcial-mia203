"""
Prueba de extremo a extremo de la Pregunta 3 con datos sintéticos.

    python tests/smoke_test_p3.py

Ejecuta el pipeline completo con una configuración reducida (YOLOv8n sin pesos preentrenados,
1 época, 160 px, CPU) en un directorio temporal, sin acceder a los datos reales ni a la red.
"""
from __future__ import annotations

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
    cfg["background"].update({"n_images": 6, "n_figures": 2})
    cfg["ablation"]["experiments"] = [
        {"name": "B_yolov8s", "set": []},
        {"name": "C_sin_vacias", "set": ["sampling.empty_ratio_train=0.0"]},
        {"name": "D_sin_rfs", "set": ["sampling.repeat_factor_sampling=false"]},
    ]
    cfg_path = tmp / "config_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    run(["-m", "p3_fauna.run_all", "--config", str(cfg_path)], env)
    final = tmp / "reports" / "resultados"
    assert any(final.glob("*/monitoreo/reporte_camara_*.md")), "falta el reporte de monitoreo"
    assert any(final.glob("*/robustez/robustez.csv")), "falta la tabla de robustez"
    assert any(final.glob("*/fondo/resumen.csv")), "falta el resumen de dependencia del fondo"
    print(f"\nPRUEBA DE HUMO SUPERADA. Archivos temporales en {tmp}.")


if __name__ == "__main__":
    main()
