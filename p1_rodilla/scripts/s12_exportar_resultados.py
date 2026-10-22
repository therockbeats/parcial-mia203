"""
PASO 12. Copia los resultados livianos de cada experimento (tablas y figuras) DENTRO del
repositorio, para versionarlos junto con el código y usarlos en el informe.

Copia de <runs_dir>/<exp>/:  metrics_gold.csv, metrics_val.csv, history.csv, summary.json,
roc_gold.png y gradcam/*.png   ->   p1_rodilla/reports/resultados/<exp>/
Los pesos (best.pt, cientos de MB) NO se copian: permanecen en runs_dir.

Uso:  python -m p1_rodilla.scripts.s12_exportar_resultados
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger

FILES = ["metrics_gold.csv", "metrics_val.csv", "history.csv", "summary.json", "roc_gold.png", "train.log", "eval.log"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    cfg = load_config(ap.parse_args().config)
    log = get_logger("s12")
    runs = Path(cfg["data"]["runs_dir"])
    dest_root = ensure_dir(Path(cfg["data"]["reports_dir"]) / "resultados")
    n = 0
    for exp in sorted(p for p in runs.iterdir() if p.is_dir()):
        dest = ensure_dir(dest_root / exp.name)
        for f in FILES:
            if (exp / f).exists():
                shutil.copy2(exp / f, dest / f)
                n += 1
        if (exp / "gradcam").exists():
            for png in (exp / "gradcam").glob("*.png"):
                shutil.copy2(png, ensure_dir(dest / "gradcam") / png.name)
                n += 1
    log.info(f"{n} archivos copiados a {dest_root}.")


if __name__ == "__main__":
    main()
