"""
Etapa 12 (bono). Fusión de Sentinel-1 y Sentinel-2.

Entrena con los chips dorados de baja nubosidad (``data_filter.max_cloud_fraction``) una U-Net
solo con SAR (VV, VH) y otra con SAR más seis bandas ópticas (B2, B3, B4, B8, B11, B12), con las
semillas de la ablación, y compara sus ensambles en los mismos chips de validación, prueba y
Bolivia mediante bootstrap pareado.

Salidas:
    p2_inundaciones/reports/tablas/fusion_corridas.csv, fusion_resumen.csv
    p2_inundaciones/reports/figuras/fusion_iou.png

Uso:
    python -m p2_inundaciones.scripts.s12_fusion_optica [--solo-tabla]
"""
from __future__ import annotations

import argparse

from common.config import REPO_ROOT, load_config
from common.utils import get_logger
from p2_inundaciones.scripts.s09_ablacion import summarize, train_and_eval
from p2_inundaciones.src.io_s1f import load_chips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p2_inundaciones" / "config.yaml"))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--solo-tabla", action="store_true")
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()
    args.extra = []
    cfg = load_config(args.config)
    log = get_logger("s12")
    fu = cfg["fusion"]
    exps = fu["experiments"]
    mc = next(float(s.split("=", 1)[1]) for s in exps[0]["set"] if s.startswith("data_filter.max_cloud_fraction="))
    hand = load_chips(cfg)
    hand = hand[hand.group == "hand"]
    log.info("Chips dorados con nubosidad <= {:.0%}: ".format(mc) + ", ".join(
        f"{k} {int(((hand.split == k) & (hand.cloud_frac <= mc)).sum())}/{int((hand.split == k).sum())}"
        for k in ("train", "val", "test", "bolivia")))
    seeds = args.seeds or [int(s) for s in cfg["ablation"]["seeds"]]
    if not args.solo_tabla:
        train_and_eval(cfg, args, exps, seeds, log)
    cfg_f = load_config(args.config, [s for s in exps[0]["set"] if s.startswith("data_filter")])
    summarize(cfg_f, exps, [int(s) for s in cfg["ablation"]["seeds"]], exps[0]["name"], ["val", "test", "bolivia"], "fusion", log)


if __name__ == "__main__":
    main()
