"""
Pipeline completo de la Pregunta 3.

    python -m p3_fauna.run_all [--desde N] [--extras]

Etapas: vacías de entrenamiento desde el dataset completo (1b, si aún no existen), inventario (1),
conversión a YOLO (2), entrenamiento y evaluación de los experimentos que pide el enunciado (5, que
llama a 3 y 4; con --extras también los adicionales) y, sobre el mejor según mAP@0.5:0.95 en cis_val:
desbalance (6), robustez (7), dependencia del fondo con Grad-CAM (8), monitoreo ecológico (9) y
exportación (10). La verificación del entorno (0) se ejecuta por separado.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common.config import REPO_ROOT, load_config
from p3_fauna.src.experimentos import best_experiment


def sh(module: str, *extra: str) -> None:
    """Ejecuta una etapa como módulo de Python."""
    cmd = [sys.executable, "-m", module, *extra]
    print("\n" + "=" * 80 + "\n>> " + " ".join(cmd) + "\n" + "=" * 80, flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=1)
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    ap.add_argument("--extras", action="store_true", help="entrenar también los experimentos adicionales")
    args = ap.parse_args()
    c = ["--config", args.config]
    cfg = load_config(args.config)
    ann_dir = Path(cfg["data"]["raw_dir"]) / cfg["data"]["annotations_subdir"]
    extra = [ann_dir / f for f in (cfg["dataset"].get("extra_files") or {}).get("train", [])]
    if args.desde <= 1 and any(not p.exists() for p in extra):
        sh("p3_fauna.scripts.s01b_vacias_entrenamiento", *c)
    for k, module, more in [(1, "s01_inventario", []), (2, "s02_convertir_yolo", []),
                            (5, "s05_ablacion", ["--extras"] if args.extras else [])]:
        if k >= args.desde:
            sh(f"p3_fauna.scripts.{module}", *c, *more)
    best, best_map = best_experiment(cfg)
    if best is None:
        raise RuntimeError("Ningún experimento evaluado: revisar la etapa 5")
    print(f"\nMejor experimento según mAP@0.5:0.95 en {cfg['eval']['select_split']}: {best} ({best_map:.4f})")
    for k, module in [(6, "s06_desbalance"), (7, "s07_robustez"), (8, "s08_fondo_gradcam"), (9, "s09_monitoreo"),
                      (10, "s10_exportar_resultados")]:
        if k >= args.desde:
            sh(f"p3_fauna.scripts.{module}", *c, "--exp", best)
    print(f"\nPipeline de la Pregunta 3 terminado. Experimento final: {best}")


if __name__ == "__main__":
    main()
