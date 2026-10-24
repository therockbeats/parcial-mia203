"""
Prueba de extremo a extremo de la Pregunta 2 con datos sintéticos.

    python tests/smoke_test_p2.py

Ejecuta el inventario, los chips de Lima a partir de GeoTIFF y el pipeline completo con una
configuración reducida en un directorio temporal, sin acceder a los datos reales ni a la red.
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
    tmp = Path(tempfile.mkdtemp(prefix="mia203_p2_smoke_"))
    paths = tmp / "paths.yaml"
    paths.write_text(yaml.safe_dump({"data_root": str(tmp / "data"), "runs_root": str(tmp / "runs")}))
    env = dict(os.environ, MIA203_PATHS=str(paths))
    run([str(REPO / "tests" / "make_synthetic_p2.py"), "--out", str(tmp / "data")], env)

    cfg = yaml.safe_load((REPO / "p2_inundaciones" / "config.yaml").read_text(encoding="utf-8"))
    cfg["data"]["reports_dir"] = str(tmp / "reports")
    cfg["preprocess"]["num_workers"] = 2
    cfg["lima"]["gdb"] = str(tmp / "data" / "p2_lima" / "ems" / "EMSN038_sintetico.gpkg")
    cfg["lima"]["min_valid_fraction"] = 0.2
    cfg["model"].update({"base_channels": 8, "depth": 3})
    cfg["train"].update({"steps": 24, "eval_every": 12, "batch_size": 4, "crop": 128, "num_workers": 0,
                         "warmup_steps": 4, "device": "cpu"})
    cfg["eval"]["bootstrap"] = 50
    cfg["ablation"]["seeds"] = [42, 1]
    cfg["ablation"]["final"] = "GW_bcedice"
    cfg["ablation"]["experiments"] = [e for e in cfg["ablation"]["experiments"]
                                      if e["name"] in ("G_bcedice", "W_otsu", "GW_bcedice", "WG_ajuste", "GW_atencion",
                                                       "G_atencion")]
    for e in cfg["ablation"]["experiments"]:
        if e["name"] == "WG_ajuste":
            e["set"] = [s for s in e["set"] if not s.startswith("train.steps")] + ["train.steps=12"]
    cfg_path = tmp / "config_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    c = ["--config", str(cfg_path)]

    run(["-m", "p2_inundaciones.scripts.s02_inventario", *c], env)
    run(["-m", "p2_inundaciones.scripts.s03_preparar_chips", *c], env)
    run(["-m", "p2_inundaciones.scripts.s05b_lima_s1_geotiff", *c,
         "--post", str(tmp / "data/p2_lima/s1_snap/S1B_post_db.tif"),
         "--pre", str(tmp / "data/p2_lima/s1_snap/S1B_pre_db.tif")], env)
    run(["-m", "p2_inundaciones.run_all", *c], env)
    print(f"\nPRUEBA DE HUMO SUPERADA. Archivos temporales en {tmp}.")


if __name__ == "__main__":
    main()
