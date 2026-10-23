"""
Etapa 4. Calidad de las etiquetas débiles y línea base clásica.

1. Compara la etiqueta de Otsu sobre VH (S1OtsuLabelHand) con la etiqueta experta (LabelHand)
   en los chips dorados: estima el ruido de las etiquetas débiles de Sentinel-1 y sirve como línea
   base de umbralización clásica en las particiones de prueba y de Bolivia.
2. Compara entre sí las dos fuentes débiles (Otsu sobre S1 e índice de agua de S2) en los chips
   débiles donde ambas existen.
3. Mide qué parte del agua etiquetada a mano es agua permanente según JRC.

Salidas:
    p2_inundaciones/reports/tablas/calidad_etiquetas_debiles.csv
    p2_inundaciones/reports/tablas/calidad_etiquetas_por_evento.csv
    p2_inundaciones/reports/tablas/acuerdo_fuentes_debiles.csv

Uso:
    python -m p2_inundaciones.scripts.s04_calidad_etiquetas
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p2_inundaciones.src.dataset import label_path, s1_path
from p2_inundaciones.src.io_s1f import load_chips
from p2_inundaciones.src.metrics import confusion, scores


def kappa(tp, fp, fn, tn) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return float("nan")
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def compare(cfg, chip_ids, src_pred: str, src_ref: str) -> pd.DataFrame:
    """Conteos por chip de ``src_pred`` frente a ``src_ref`` (agua = 1) en píxeles válidos de ambas."""
    rows = []
    for cid in chip_ids:
        pp, pr = label_path(cfg, src_pred, cid), label_path(cfg, src_ref, cid)
        if not (pp.exists() and pr.exists()):
            continue
        a, b = np.load(pp), np.load(pr)
        has = np.isfinite(np.load(s1_path(cfg, cid)).astype(np.float32)).all(axis=0)
        v = (a >= 0) & (b >= 0) & has
        rows.append({"chip_id": cid, **confusion(a == 1, b, v)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    s = scores(df.tp.sum(), df.fp.sum(), df.fn.sum(), df.tn.sum())
    s["kappa"] = kappa(df.tp.sum(), df.fp.sum(), df.fn.sum(), df.tn.sum())
    s["chips"] = len(df)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s04")
    chips = load_chips(cfg)
    hand = chips[chips.group == "hand"]
    tab = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas")

    rows, by_event = [], []
    for split in ["train", "val", "test", "bolivia"]:
        ids = hand.loc[hand.split == split, "chip_id"].tolist()
        c = compare(cfg, ids, "S1OtsuLabelHand", "LabelHand")
        if len(c):
            rows.append({"comparacion": "Otsu S1 vs experta", "particion": split, **summarize(c)})
            c["event"] = c.chip_id.str.rsplit("_", n=1).str[0]
            for ev, g in c.groupby("event"):
                by_event.append({"particion": split, "evento": ev, **summarize(g)})
    all_hand = compare(cfg, hand.chip_id.tolist(), "S1OtsuLabelHand", "LabelHand")
    if len(all_hand):
        rows.append({"comparacion": "Otsu S1 vs experta", "particion": "todos", **summarize(all_hand)})
    jrc = compare(cfg, hand.chip_id.tolist(), "JRCWaterHand", "LabelHand")
    if len(jrc):
        s = summarize(jrc)
        rows.append({"comparacion": "Agua permanente JRC vs experta", "particion": "todos", **s})
    res = pd.DataFrame(rows)
    res.to_csv(tab / "calidad_etiquetas_debiles.csv", index=False)
    pd.DataFrame(by_event).to_csv(tab / "calidad_etiquetas_por_evento.csv", index=False)

    weak = chips[(chips.group == "weak") & chips.get("has_S2IndexLabelWeak", False) & chips.get("has_S1OtsuLabelWeak", False)]
    w = compare(cfg, weak.chip_id.tolist(), "S1OtsuLabelWeak", "S2IndexLabelWeak")
    if len(w):
        pd.DataFrame([{"comparacion": "Otsu S1 vs índice S2 (chips débiles)", **summarize(w)}]).to_csv(
            tab / "acuerdo_fuentes_debiles.csv", index=False)
        log.info(f"Otsu S1 frente a índice S2 en {len(w)} chips débiles: IoU {summarize(w)['iou']:.3f}")
    cols = ["comparacion", "particion", "chips", "iou", "f1", "precision", "recall", "kappa", "exactitud"]
    log.info("\n" + res[cols].round(3).to_string(index=False))
    log.info(f"Guardado {tab / 'calidad_etiquetas_debiles.csv'}")


if __name__ == "__main__":
    main()
