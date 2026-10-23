"""
Etapa 0. Verificación del entorno para la Pregunta 2.

Comprueba PyTorch y la GPU, las librerías geoespaciales (rasterio, geopandas, pyogrio y el
controlador OpenFileGDB de GDAL), el cliente de Google Earth Engine y las rutas de datos.

Uso:
    python -m p2_inundaciones.scripts.s00_verificar_entorno
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> None:
    print("=" * 70)
    print("Verificación del entorno (Pregunta 2)")
    print("=" * 70)
    print(f"Sistema operativo : {platform.system()} {platform.release()}")
    print(f"Python            : {sys.version.split()[0]}")

    import torch

    print(f"PyTorch           : {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU               : {props.name} | {props.total_memory/1024**3:.0f} GB | bf16 {torch.cuda.is_bf16_supported()}")
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("GPU               : MPS (Apple)")
        dev = torch.device("mps")
    else:
        print("GPU               : no se detecta; se usará CPU")
        dev = torch.device("cpu")

    from p2_inundaciones.src.unet import UNet

    x = torch.randn(2, 2, 256, 256, device=dev)
    y = UNet().to(dev)(x)
    print(f"Prueba de U-Net en {dev}: salida {tuple(y.shape)} OK")

    for mod in ["numpy", "pandas", "pyarrow", "scipy", "matplotlib", "rasterio", "geopandas", "pyogrio", "shapely"]:
        try:
            m = __import__(mod)
            print(f"{mod:18s}: {getattr(m, '__version__', 'OK')}")
        except ImportError:
            print(f"{mod:18s}: FALTA")
    try:
        import pyogrio

        drivers = pyogrio.list_drivers()
        print(f"GDAL OpenFileGDB  : {'OK' if 'OpenFileGDB' in drivers else 'FALTA'}")
    except ImportError:
        pass
    try:
        import ee

        print(f"earthengine-api   : {ee.__version__}")
    except ImportError:
        print("earthengine-api   : FALTA (solo necesario para la etapa 5)")

    from common.config import REPO_ROOT, load_config

    try:
        cfg = load_config(REPO_ROOT / "p2_inundaciones" / "config.yaml")
        for key in ("raw_dir", "lima_dir"):
            p = Path(cfg["data"][key])
            print(f"{key:18s}: {p} -> {'EXISTE' if p.exists() else 'NO EXISTE'}")
    except FileNotFoundError as e:
        print(f"Rutas             : {e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
