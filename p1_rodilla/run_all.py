"""
Pipeline completo de la Pregunta 1 con UN SOLO COMANDO (requisito de reproducibilidad):

    python -m p1_rodilla.run_all

Ejecuta en orden: metadatos -> etiquetas (reglas) -> selección -> preprocesamiento ->
ablación (entrena y evalúa todos los experimentos) -> Grad-CAM del mejor -> desbalance y sesgo.
El inventario (PASO 1) y el LLM (PASO 4) se corren aparte porque son exploratorio/opcional.

Opciones:
    --desde 7     empieza en el paso 7 (si los anteriores ya están hechos)
    --smoke       prueba rápida de punta a punta con pocos estudios y 1 época
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common.config import REPO_ROOT, load_config


def sh(module: str, *extra: str) -> None:
    cmd = [sys.executable, "-m", module, *extra]
    print("\n" + "=" * 80 + "\n>> " + " ".join(cmd) + "\n" + "=" * 80, flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=2)
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    c = ["--config", args.config]
    steps = {
        2: ("p1_rodilla.scripts.s02_metadatos_series", c),
        3: ("p1_rodilla.scripts.s03_etiquetas_reglas", c),
        5: ("p1_rodilla.scripts.s05_seleccionar_estudios", c),
        6: ("p1_rodilla.scripts.s06_preprocesar", c),
        10: ("p1_rodilla.scripts.s10_ablacion", c + (["--smoke"] if args.smoke else [])),
    }
    for k in sorted(steps):
        if k >= args.desde:
            sh(steps[k][0], *steps[k][1])
    # Grad-CAM sobre el mejor experimento según AUC macro dorado
    cfg = load_config(args.config)
    best, best_auc = None, -1
    for e in cfg["ablation"]["experiments"]:
        p = Path(cfg["data"]["runs_dir"]) / e["name"] / "summary.json"
        if p.exists():
            auc = json.loads(p.read_text()).get("gold_macro_auc") or -1
            if auc == auc and auc > best_auc:   # auc == auc descarta NaN
                best, best_auc = e["name"], auc
    if best:
        sh("p1_rodilla.scripts.s09_gradcam", "--exp", best, *c)
    sh("p1_rodilla.scripts.s11_desbalance_y_sesgo", *c)
    sh("p1_rodilla.scripts.s13_comparacion_pareada", *c)
    sh("p1_rodilla.scripts.s12_exportar_resultados", *c)
    print(f"\nListo. Mejor experimento: {best} (AUC macro dorado = {best_auc:.4f})")


if __name__ == "__main__":
    main()
