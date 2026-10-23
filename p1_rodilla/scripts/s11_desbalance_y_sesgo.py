"""
Etapa 11. Desbalance de clases y sesgo lingüístico.

Prevalencia de cada hallazgo por partición (tabla y figura), AUC por hallazgo de los
experimentos de la ablación ordenado por prevalencia, y porcentaje de positivos de las
reglas por idioma del reporte.

Salidas:
    p1_rodilla/reports/tablas/prevalencia_pct.csv
    p1_rodilla/reports/tablas/auc_por_hallazgo_vs_prevalencia.csv
    p1_rodilla/reports/tablas/positivos_pct_por_idioma.csv
    p1_rodilla/reports/figuras/prevalencia.png

Uso:
    python -m p1_rodilla.scripts.s11_desbalance_y_sesgo
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p1_rodilla.src.labels_rules import POS


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    cfg = load_config(ap.parse_args().config)
    log = get_logger("s11")
    names = cfg["labels"]["names"]
    interim = Path(cfg["data"]["interim_dir"])
    tab_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")
    fig_dir = ensure_dir(Path(cfg["data"]["reports_dir"]) / "figuras")

    split = pd.read_csv(interim / "split.csv", dtype={"study_id": str})
    prev = pd.DataFrame({s: (g[names] == POS).mean() * 100 for s, g in split.groupby("split")}).round(1)
    prev.columns = [f"% positivos {c}" for c in prev.columns]
    prev.to_csv(tab_dir / "prevalencia_pct.csv")
    log.info("\nPrevalencia (%):\n" + prev.to_string())

    ax = prev.plot.barh(figsize=(8, 6))
    ax.set_xlabel("% de estudios positivos")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(fig_dir / "prevalencia.png", dpi=150)
    plt.close()

    # AUC por hallazgo de los experimentos de la ablación
    rows = []
    for e in cfg["ablation"]["experiments"]:
        p = Path(cfg["data"]["runs_dir"]) / e["name"] / "metrics_gold.csv"
        if p.exists():
            m = pd.read_csv(p).set_index("hallazgo")["auc"].rename(e["name"])
            rows.append(m)
    if rows:
        comp = pd.concat(rows, axis=1).round(3)
        train_prev = (split[split.split == "train"][names] == POS).mean()
        comp["prevalencia_train_%"] = (train_prev * 100).round(1)
        comp = comp.sort_values("prevalencia_train_%")
        comp.to_csv(tab_dir / "auc_por_hallazgo_vs_prevalencia.csv")
        log.info("\nAUC por hallazgo (por prevalencia creciente):\n" + comp.to_string())

    # Sesgo lingüístico
    lr = interim / "labels_rules.parquet"
    if lr.exists():
        rules = pd.read_parquet(lr)
        sub = rules[rules.study_id.isin(split.study_id)]
        by_lang = sub.groupby("language").apply(
            lambda g: pd.Series({"reportes": len(g), **{n: round((g[n] == POS).mean() * 100, 1) for n in names}}))
        by_lang = by_lang.sort_values("reportes", ascending=False)
        by_lang.to_csv(tab_dir / "positivos_pct_por_idioma.csv")
        log.info("\n% de positivos por idioma (reglas):\n" + by_lang.to_string())
    log.info(f"Tablas en {tab_dir} y figuras en {fig_dir}")


if __name__ == "__main__":
    main()
