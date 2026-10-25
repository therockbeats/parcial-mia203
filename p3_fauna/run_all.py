"""
Pipeline completo de la Pregunta 3.

    python -m p3_fauna.run_all [--desde N]

Etapas: inventario (1), conversión a YOLO (2), ablación con entrenamiento y evaluación de cada
experimento (5, que llama a 3 y 4) y, sobre el mejor experimento según mAP@0.5:0.95 en cis_val:
desbalance (6), robustez (7), dependencia del fondo con Grad-CAM (8), monitoreo ecológico (9) y
exportación (10). La verificación del entorno (0) se ejecuta por separado.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

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
    args = ap.parse_args()
    c = ["--config", args.config]
    for k, module in [(1, "s01_inventario"), (2, "s02_convertir_yolo"), (5, "s05_ablacion")]:
        if k >= args.desde:
            sh(f"p3_fauna.scripts.{module}", *c)
    cfg = load_config(args.config)
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
