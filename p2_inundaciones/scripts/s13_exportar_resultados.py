"""
Etapa 13. Exportación de resultados al repositorio.

Copia a ``p2_inundaciones/reports/resultados/<corrida>/`` las métricas, el historial de
entrenamiento, los registros y la matriz de confusión de las corridas con la semilla base de la
ablación y de la fusión.

Uso:
    python -m p2_inundaciones.scripts.s13_exportar_resultados
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger

FILES = ["metricas.csv", "por_evento_test.csv", "history.csv", "summary.json", "confusion_test.png",
         "train.log", "eval.log", "config.yaml"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s13")
    runs = Path(cfg["data"]["runs_dir"])
    dest = ensure_dir(Path(cfg["data"]["reports_dir"]) / "resultados")
    names = [e["name"] for e in cfg["ablation"]["experiments"] + cfg["fusion"]["experiments"]]
    n = 0
    for name in names:
        src = runs / name
        if not (src / "summary.json").exists():
            continue
        out = ensure_dir(dest / name)
        for f in FILES:
            if (src / f).exists():
                shutil.copy2(src / f, out / f)
        n += 1
    log.info(f"{n} corridas exportadas a {dest}")


if __name__ == "__main__":
    main()
