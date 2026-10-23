"""
Pipeline completo de la Pregunta 2.

    python -m p2_inundaciones.run_all [--desde N] [--sin-lima] [--sin-fusion]

Etapas: tabla de chips (3), calidad de etiquetas débiles (4), etiquetas de Lima (6), ablación
(9), generalización (10), fallos del SAR (11), fusión óptica (12) y exportación (13). La descarga
(1), el inventario (2) y los chips de Lima desde Earth Engine (5) se ejecutan por separado.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common.config import REPO_ROOT, load_config
from p2_inundaciones.src.geo import lima_grid


def sh(module: str, *extra: str) -> None:
    cmd = [sys.executable, "-m", module, *extra]
    print("\n" + "=" * 80 + "\n>> " + " ".join(cmd) + "\n" + "=" * 80, flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=3)
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--sin-lima", action="store_true")
    ap.add_argument("--sin-fusion", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = ["--config", args.config]
    proc = Path(cfg["data"]["processed_dir"])
    lima_ready = not args.sin_lima and any((proc / "s1" / f"{ch['chip_id']}.npy").exists() for ch in lima_grid(cfg))
    steps = {
        3: ("p2_inundaciones.scripts.s03_preparar_chips", c),
        4: ("p2_inundaciones.scripts.s04_calidad_etiquetas", c),
        6: ("p2_inundaciones.scripts.s06_lima_etiquetas", c) if lima_ready else None,
        9: ("p2_inundaciones.scripts.s09_ablacion", c),
        10: ("p2_inundaciones.scripts.s10_generalizacion", c),
        11: ("p2_inundaciones.scripts.s11_fallos_sar", c + ([] if lima_ready else ["--splits", "test", "bolivia"])),
        12: None if args.sin_fusion else ("p2_inundaciones.scripts.s12_fusion_optica", c),
        13: ("p2_inundaciones.scripts.s13_exportar_resultados", c),
    }
    if not lima_ready:
        print("Sin chips de Lima: se omite la etapa 6 y Lima en las evaluaciones.")
    for k in sorted(steps):
        if k >= args.desde and steps[k] is not None:
            sh(steps[k][0], *steps[k][1])
    print("\nPipeline de la Pregunta 2 terminado.")


if __name__ == "__main__":
    main()
