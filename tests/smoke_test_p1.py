"""
Prueba de humo de la Pregunta 1: corre TODO el pipeline con datos falsos y pequeños.

    python tests/smoke_test_p1.py

Sirve para comprobar, en la Mac o en Windows, que el entorno y el código funcionan
ANTES de gastar horas con los datos reales. Tarda ~3-5 minutos en CPU.
No accede a los datos reales ni a las rutas locales: todo se genera en un directorio temporal.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mia203_smoke_"))
    paths = tmp / "paths.yaml"
    paths.write_text(yaml.safe_dump({"data_root": str(tmp / "data"), "runs_root": str(tmp / "runs")}))
    env = dict(os.environ, MIA203_PATHS=str(paths))

    # 1) Datos sintéticos con la estructura supuesta de Kaggle
    raw = tmp / "data" / "p1_rsna_knee" / "raw"
    subprocess.run([sys.executable, str(REPO / "tests" / "make_synthetic_p1.py"), "--out", str(raw), "--studies", "60"],
                   check=True, cwd=REPO, env=env)

    # 2) Config reducido: imágenes de 64 px, 2 épocas, solo 2 experimentos
    cfg = yaml.safe_load((REPO / "p1_rodilla" / "config.yaml").read_text(encoding="utf-8"))
    cfg["data"]["reports_dir"] = str(tmp / "reports")   # no ensuciar p1_rodilla/reports del repo
    cfg["selection"].update({"n_studies": 40, "min_pos_per_label": 3, "min_slices": 10})
    cfg["preprocess"].update({"n_slices": 12, "size": 64, "num_workers": 2})
    cfg["model"].update({"backbone": "scratch", "pretrained": False, "slices_in": 8})
    cfg["train"].update({"epochs": 2, "batch_size": 4, "num_workers": 0})
    cfg["eval"]["bootstrap"] = 50
    cfg["ablation"]["experiments"] = [
        {"name": "A_scratch_bce", "set": ["model.backbone=scratch"]},
        {"name": "C_scratch_focal", "set": ["model.backbone=scratch", "train.loss=focal"]},
    ]
    cfg_path = tmp / "config_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    # 3) Pipeline completo
    subprocess.run([sys.executable, "-m", "p1_rodilla.run_all", "--config", str(cfg_path), "--smoke"],
                   check=True, cwd=REPO, env=env)
    print(f"\nPRUEBA DE HUMO SUPERADA. Archivos temporales en {tmp}.")


if __name__ == "__main__":
    main()
