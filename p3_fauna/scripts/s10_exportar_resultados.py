"""
Etapa 10. Exportación de resultados al repositorio.

Copia a ``p3_fauna/reports/resultados/<exp>/`` las métricas, el historial de entrenamiento, los
registros, la configuración y las matrices de confusión de cada experimento de la ablación, y del
experimento final además las carpetas de robustez, fondo (Grad-CAM) y monitoreo. Los pesos no se
copian.

Uso:
    python -m p3_fauna.scripts.s10_exportar_resultados
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p3_fauna.src.experimentos import best_experiment, exp_dir

FILES = ["metricas.csv", "por_especie.csv", "history.csv", "summary.json", "muestreo.json", "config.yaml",
         "train.log", "eval.log", "results.png", "confusion_cis_test.png", "confusion_trans_test.png"]
FOLDERS = {"robustez": ("*.csv", "*.png", "*.log"), "fondo": ("*.csv", "*.png", "*.jpg", "*.log"),
           "monitoreo": ("*.md", "*.csv", "*.log")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    ap.add_argument("--exp", default=None, help="experimento final (por defecto, el mejor de la ablación)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s10")
    final = args.exp or best_experiment(cfg)[0]
    dest_root = ensure_dir(Path(cfg["data"]["reports_dir"]) / "resultados")
    n = 0
    for e in cfg["ablation"]["experiments"]:
        src = exp_dir(cfg, e["name"])
        if not (src / "summary.json").exists():
            continue
        dest = ensure_dir(dest_root / e["name"])
        for f in FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dest / f)
                n += 1
        if e["name"] == final:
            for folder, patterns in FOLDERS.items():
                for pat in patterns:
                    for p in (src / folder).glob(pat):
                        shutil.copy2(p, ensure_dir(dest / folder) / p.name)
                        n += 1
    log.info(f"{n} archivos copiados a {dest_root} (experimento final: {final})")


if __name__ == "__main__":
    main()
