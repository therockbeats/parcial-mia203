"""
Etapa 0. Verificación del entorno para la Pregunta 3.

Comprueba PyTorch y la GPU, Ultralytics (inferencia de una red YOLOv8n sin pesos sobre una
imagen aleatoria), OpenCV y las rutas de datos de CCT-20.

Uso:
    python -m p3_fauna.scripts.s00_verificar_entorno
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> None:
    print("=" * 70)
    print("Verificación del entorno (Pregunta 3)")
    print("=" * 70)
    print(f"Sistema operativo : {platform.system()} {platform.release()}")
    print(f"Python            : {sys.version.split()[0]}")

    import torch

    print(f"PyTorch           : {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU               : {props.name} | {props.total_memory/1024**3:.0f} GB")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("GPU               : MPS (Apple)")
    else:
        print("GPU               : no se detecta; se usará CPU")

    for mod in ["numpy", "pandas", "matplotlib", "cv2", "PIL", "yaml", "ultralytics"]:
        try:
            m = __import__(mod)
            print(f"{mod:18s}: {getattr(m, '__version__', 'OK')}")
        except ImportError:
            print(f"{mod:18s}: FALTA")

    import numpy as np
    from ultralytics import YOLO

    from p3_fauna.src.yolo_io import ultralytics_device

    dev = ultralytics_device("auto")
    r = YOLO("yolov8n.yaml").predict(np.zeros((320, 320, 3), np.uint8), imgsz=320, device=dev, verbose=False)
    print(f"Prueba de YOLOv8n en {dev}: {len(r)} resultado(s) OK")

    from common.config import REPO_ROOT, load_config

    try:
        cfg = load_config(REPO_ROOT / "p3_fauna" / "config.yaml")
        raw = Path(cfg["data"]["raw_dir"])
        for sub in (cfg["data"]["images_subdir"], cfg["data"]["annotations_subdir"]):
            p = raw / sub
            print(f"{sub:30s}: {p} -> {'EXISTE' if p.exists() else 'NO EXISTE'}")
        for name, f in cfg["dataset"]["split_files"].items():
            p = raw / cfg["data"]["annotations_subdir"] / f
            print(f"  {name:12s}: {'OK' if p.exists() else 'FALTA'} ({f})")
        from p3_fauna.scripts.s01b_vacias_entrenamiento import find_metadata

        vac = raw / cfg["data"]["annotations_subdir"] / "train_vacias_annotations.json"
        if vac.exists():
            print(f"Vacías de train    : OK ({vac.name}, etapa 1b ya ejecutada)")
        else:
            try:
                print(f"Vacías de train    : pendiente; anotaciones del dataset completo -> {find_metadata(cfg).name}")
            except FileNotFoundError:
                print("Vacías de train    : FALTA el archivo 'Image-level annotations' del dataset completo en raw/")
    except FileNotFoundError as e:
        print(f"Rutas             : {e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
