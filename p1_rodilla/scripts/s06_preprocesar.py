"""
PASO 6. Preprocesamiento: DICOM -> volúmenes .npy listos para entrenar.

Para cada serie elegida en split.csv:
    1. Carga todos los cortes DICOM y los ordena espacialmente.
    2. Ventana de intensidad por percentiles (p_low..p_high) -> 0..255.
    3. Re-muestrea a n_slices cortes y redimensiona a size x size.
    4. Guarda <processed_dir>/<study_id>__<series_id>.npy (uint8, ~1.5 MB).

Hacer esto UNA vez evita leer y decodificar DICOM en cada época (lo más lento).
Usa varios procesos en paralelo; si se corta, al volver a correr salta lo ya hecho.

Uso:  python -m p1_rodilla.scripts.s06_preprocesar   [--workers 20]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.config import REPO_ROOT, load_config
from common.utils import Timer, ensure_dir, get_logger
from p1_rodilla.src.dicom_utils import load_series_volume, normalize_volume


def _one(args):
    series_dir, out_path, ext, n_slices, size, p_low, p_high = args
    if Path(out_path).exists():
        return out_path, "existente"
    try:
        vol = load_series_volume(Path(series_dir), ext)
        arr = normalize_volume(vol, n_slices, size, p_low, p_high)
        np.save(out_path, arr)
        return out_path, "ok"
    except Exception as e:
        return out_path, f"error: {str(e)[:150]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s06")
    pp, sch = cfg["preprocess"], cfg["schema"]
    images_root = Path(cfg["data"]["raw_dir"]) / sch["images_subdir"]
    out_dir = ensure_dir(cfg["data"]["processed_dir"])
    split = pd.read_csv(Path(cfg["data"]["interim_dir"]) / "split.csv", dtype={"study_id": str, "series_id": str})

    tasks = [(images_root / r.study_id / r.series_id, str(out_dir / f"{r.study_id}__{r.series_id}.npy"),
              sch.get("dicom_ext", ".dcm").lower(), pp["n_slices"], pp["size"], pp["p_low"], pp["p_high"])
             for r in split.itertuples()]
    log.info(f"Series a procesar: {len(tasks)}")
    results = []
    with Timer("Preprocesamiento", log), ProcessPoolExecutor(max_workers=args.workers or pp["num_workers"]) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="volúmenes"):
            results.append(f.result())
    res = pd.DataFrame(results, columns=["archivo", "estado"])
    log.info("\n" + res["estado"].str.split(":").str[0].value_counts().to_string())
    errs = res[res.estado.str.startswith("error")]
    if len(errs):
        p = Path(cfg["data"]["interim_dir"]) / "preprocess_errors.csv"
        errs.to_csv(p, index=False)
        log.warning(f"{len(errs)} series fallaron; detalle en {p}. Esos estudios se omitirán al entrenar.")

    # Control visual: mosaico con el corte central de 12 volúmenes al azar
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(out_dir.glob("*.npy"))
    if files:
        pick = np.random.default_rng(0).choice(len(files), size=min(12, len(files)), replace=False)
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for ax, i in zip(axes.ravel(), pick):
            v = np.load(files[i])
            ax.imshow(v[v.shape[0] // 2], cmap="gray")
            ax.set_title("…" + files[i].stem.split("__")[0][-8:], fontsize=8)
        for ax in axes.ravel():
            ax.axis("off")
        fig.suptitle("Control de calidad: corte central de volúmenes preprocesados")
        fig.tight_layout()
        fig_path = ensure_dir(Path(cfg["data"]["reports_dir"]) / "figuras") / "control_preprocesado.png"
        fig.savefig(fig_path, dpi=120)
        log.info(f"Revisa visualmente {fig_path}: deben verse rodillas sagitales nítidas y bien contrastadas")


if __name__ == "__main__":
    main()
