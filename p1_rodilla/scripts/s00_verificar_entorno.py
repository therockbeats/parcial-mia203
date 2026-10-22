"""
PASO 0. Verificar que el entorno funciona en ESTA máquina.

Qué comprueba:
  1. Versiones de Python, PyTorch y librerías clave.
  2. Que la GPU se ve (CUDA en Windows, MPS en la Mac) y que puede multiplicar matrices.
  3. Que pydicom tiene decodificadores para DICOM comprimidos (JPEG Lossless / JPEG 2000).
  4. Que configs/paths.local.yaml existe y que la carpeta de datos crudos está donde dice el config.

Uso (desde la carpeta raíz del repositorio):
    python -m p1_rodilla.scripts.s00_verificar_entorno
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> None:
    print("=" * 70)
    print("PASO 0: verificación del entorno")
    print("=" * 70)
    print(f"Sistema operativo : {platform.system()} {platform.release()}")
    print(f"Python            : {sys.version.split()[0]}")

    # ---------- PyTorch y GPU ----------
    import torch

    print(f"PyTorch           : {torch.__version__}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"CUDA disponible   : SÍ ({n} GPU)  | CUDA de PyTorch: {torch.version.cuda}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} | {props.total_memory/1024**3:.0f} GB | "
                  f"capacidad de cómputo {props.major}.{props.minor}")
        print(f"   bf16 soportado : {torch.cuda.is_bf16_supported()}")
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("MPS (Apple)       : SÍ (GPU del chip M2 Max)")
        dev = torch.device("mps")
    else:
        print("GPU               : NO se detecta GPU, se usará CPU")
        dev = torch.device("cpu")

    # Prueba real de cálculo en el dispositivo: si la versión de PyTorch no
    # soporta la GPU (p. ej. RTX 5090 con CUDA < 12.8) aquí falla.
    a = torch.randn(2048, 2048, device=dev)
    b = torch.randn(2048, 2048, device=dev)
    c = (a @ b).sum().item()
    print(f"Prueba de cálculo en {dev}: OK (suma = {c:.1f})")

    import torchvision

    print(f"torchvision       : {torchvision.__version__}")

    # ---------- DICOM ----------
    import pydicom

    print(f"pydicom           : {pydicom.__version__}")
    try:
        from pydicom.pixels import get_decoder  # pydicom >= 3

        for uid, name in [("1.2.840.10008.1.2.4.70", "JPEG Lossless"),
                          ("1.2.840.10008.1.2.4.90", "JPEG 2000 Lossless"),
                          ("1.2.840.10008.1.2.4.91", "JPEG 2000")]:
            dec = get_decoder(uid)
            ok = dec.is_available
            print(f"   Decodificador {name:20s}: {'OK' if ok else 'FALTA (pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg)'}")
    except Exception as e:  # versiones antiguas de pydicom
        print(f"   No pude comprobar decodificadores ({e}). Actualiza pydicom a >= 3.0")

    # ---------- Otras librerías ----------
    for mod in ["numpy", "pandas", "pyarrow", "sklearn", "cv2", "yaml", "langdetect", "matplotlib"]:
        try:
            m = __import__(mod)
            print(f"{mod:18s}: {getattr(m, '__version__', 'OK')}")
        except ImportError:
            print(f"{mod:18s}: FALTA -> instala con pip/conda")

    # ---------- Rutas ----------
    from common.config import REPO_ROOT, load_config

    cfg_path = REPO_ROOT / "p1_rodilla" / "config.yaml"
    try:
        cfg = load_config(cfg_path)
        raw = Path(cfg["data"]["raw_dir"])
        print(f"data_root         : {cfg['_machine_paths']['data_root']}")
        print(f"Datos crudos P1   : {raw}  -> {'EXISTE' if raw.exists() else 'NO EXISTE (verificar configs/paths.local.yaml)'}")
    except FileNotFoundError as e:
        print(f"Rutas             : {e}")

    print("=" * 70)
    print("Entorno verificado si todos los elementos indican OK / SÍ / EXISTE.")


if __name__ == "__main__":
    main()
