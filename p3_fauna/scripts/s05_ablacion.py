"""
Etapa 5. Ablación.

Entrena (etapa 3) y evalúa (etapa 4) cada experimento de ``ablation.experiments`` (los que pide
el enunciado: referencia y sin vacías) y, con ``--extras``, también ``ablation.extra_experiments``.
Los experimentos ya evaluados se omiten, los ya entrenados solo se evalúan y los interrumpidos
(con weights/last.pt) se reanudan desde la última época guardada, salvo con ``--rehacer``. Resume los resultados
en una tabla con la diferencia frente a la referencia y una figura de mAP en cámaras vistas y no
vistas.

Salidas en p3_fauna/reports/:
    tablas/ablacion.csv, figuras/ablacion_map.png

Uso:
    python -m p3_fauna.scripts.s05_ablacion
    python -m p3_fauna.scripts.s05_ablacion --solo B_yolov8s
    python -m p3_fauna.scripts.s05_ablacion --extras
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p3_fauna.src.experimentos import all_experiments, exp_dir, read_summary


def run(module: str, *extra: str) -> None:
    subprocess.run([sys.executable, "-m", module, *extra], check=True, cwd=REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p3_fauna" / "config.yaml"))
    ap.add_argument("--solo", nargs="*", default=None)
    ap.add_argument("--rehacer", action="store_true")
    ap.add_argument("--extras", action="store_true", help="incluir los experimentos adicionales")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s05")
    pool = all_experiments(cfg, extras=args.extras or bool(args.solo))
    exps = [e for e in pool if not args.solo or e["name"] in args.solo]
    for e in exps:
        if read_summary(cfg, e["name"]) and not args.rehacer:
            log.info(f"{e['name']}: ya evaluado; se omite")
            continue
        sets = [x for s in e.get("set", []) for x in ("--set", s)]
        d = exp_dir(cfg, e["name"])
        if (d / "entrenamiento_completo.json").exists() and not args.rehacer:
            log.info(f"{e['name']}: ya entrenado; solo se evalúa")
        elif (d / "weights" / "last.pt").exists() and not args.rehacer:
            log.info(f"{e['name']}: entrenamiento interrumpido; se reanuda desde weights/last.pt")
            run("p3_fauna.scripts.s03_entrenar", "--config", args.config, "--exp", e["name"], *sets, "--reanudar")
        else:
            run("p3_fauna.scripts.s03_entrenar", "--config", args.config, "--exp", e["name"], *sets)
        run("p3_fauna.scripts.s04_evaluar", "--config", args.config, "--exp", e["name"])

    rows = []
    required = {e["name"] for e in cfg["ablation"]["experiments"]}
    for e in all_experiments(cfg):
        s = read_summary(cfg, e["name"])
        if s:
            rows.append({"experimento": e["name"], "tipo": "enunciado" if e["name"] in required else "extra",
                         "cambio": "; ".join(e.get("set", [])) or "referencia",
                         **{k: v for k, v in s.items() if k not in ("exp", "deploy_conf")}})
    if not rows:
        log.info("Sin experimentos evaluados")
        return
    tab = pd.DataFrame(rows)
    ref = cfg["ablation"]["reference"]
    if ref in set(tab.experimento):
        r = tab.set_index("experimento").loc[ref]
        for k in ("map50_95_cis_test", "map50_95_trans_test"):
            if k in tab:
                tab[f"delta_{k}"] = tab[k] - r[k]
    tables = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")
    tab.to_csv(tables / "ablacion.csv", index=False)
    cols = [c for c in ["experimento", "map50_95_cis_val", "map50_cis_test", "map50_trans_test", "map50_95_cis_test",
                        "map50_95_trans_test", "brecha_map50_95_test", "fp_vacias_trans_test"] if c in tab]
    log.info("\n" + tab[cols].round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(tab)), 4))
    x = np.arange(len(tab))
    for off, split, lab in [(-0.2, "cis_test", "cámaras vistas (cis)"), (0.2, "trans_test", "cámaras nuevas (trans)")]:
        k = f"map50_95_{split}"
        if k in tab:
            ax.bar(x + off, tab[k], width=0.4, label=lab)
    ax.set_xticks(x, tab.experimento, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("Ablación: desempeño en cámaras vistas y no vistas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_dir(Path(cfg["data"]["reports_dir"]) / "figuras") / "ablacion_map.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
