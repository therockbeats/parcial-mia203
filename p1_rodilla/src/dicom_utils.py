"""
Funciones para leer series DICOM de resonancia magnética de rodilla.

Conceptos clave
---------------
* Un ESTUDIO (study) es un examen de un paciente. Contiene varias SERIES.
* Una SERIE es una secuencia de imágenes con el mismo plano y contraste
  (por ejemplo "SAG PD FS"). Cada archivo .dcm de la serie es un CORTE (instance).
* El PLANO se deduce de la etiqueta ImageOrientationPatient (IOP): seis números
  que dan las direcciones de las filas (r) y columnas (c) de la imagen en el espacio
  del paciente. La normal al corte es n = r × c (producto vectorial):
      n apunta sobre todo en x  -> plano SAGITAL
      n apunta sobre todo en y  -> plano CORONAL
      n apunta sobre todo en z  -> plano AXIAL
* Los cortes se ORDENAN proyectando ImagePositionPatient (IPP) sobre n.
  InstanceNumber no es confiable entre centros distintos.
* "Fat sat" (supresión grasa: FS, SPAIR, STIR...) en secuencias DP/T2 hace que el
  líquido y el edema brillen: es la secuencia más sensible para derrame, edema óseo,
  roturas de menisco y ligamentos. Por eso es la preferida.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

FS_PATTERN = re.compile(r"(\bfs\b|fat.?sat|fatsat|f\.s\.|spair|stir|spir|\bfs[_ ]|[_ ]fs\b|dixon|tirm|fat.?sup|\bfsat\b|_fs|fs_)", re.I)
PD_PATTERN = re.compile(r"(\bpd\b|proton|\bdp\b|_pd|pd_|\bpdw|pd.?w)", re.I)
T2_PATTERN = re.compile(r"(\bt2\b|t2w|_t2|t2_|t2\*|\bt2.)", re.I)
T1_PATTERN = re.compile(r"(\bt1\b|t1w|_t1|t1_|\bt1.)", re.I)


def plane_from_iop(iop) -> str:
    """Plano anatómico a partir de ImageOrientationPatient (6 números)."""
    try:
        v = np.asarray([float(x) for x in iop], dtype=float)
        r, c = v[:3], v[3:]
        n = np.cross(r, c)
        axis = int(np.argmax(np.abs(n)))
        return ["sagittal", "coronal", "axial"][axis]
    except Exception:
        return "unknown"


def normal_from_iop(iop) -> np.ndarray | None:
    try:
        v = np.asarray([float(x) for x in iop], dtype=float)
        n = np.cross(v[:3], v[3:])
        return n / (np.linalg.norm(n) + 1e-8)
    except Exception:
        return None


def classify_sequence(description: str, scan_options: str = "", tr=None, te=None) -> dict:
    """
    Clasifica la serie: ¿tiene supresión grasa? ¿es PD, T2 o T1?
    Usa primero el texto (SeriesDescription/ScanOptions) y, si no alcanza,
    los tiempos TR/TE (en ms):  T1: TR corto y TE corto; PD: TR largo y TE corto; T2: TE largo.
    """
    text = f"{description or ''} {scan_options or ''}"
    fat_sat = bool(FS_PATTERN.search(text)) or "FS" in str(scan_options).upper().split("\\")
    if PD_PATTERN.search(text):
        weighting = "PD"
    elif T2_PATTERN.search(text):
        weighting = "T2"
    elif T1_PATTERN.search(text):
        weighting = "T1"
    else:
        weighting = "unknown"
        try:
            tr_, te_ = float(tr), float(te)
            if tr_ < 900 and te_ < 30:
                weighting = "T1"
            elif te_ >= 60:
                weighting = "T2"
            elif tr_ >= 900:
                weighting = "PD"
        except (TypeError, ValueError):
            pass
    return {"fat_sat": fat_sat, "weighting": weighting}


def list_dicoms(series_dir: Path, ext: str = ".dcm") -> list[Path]:
    return sorted(p for p in Path(series_dir).iterdir() if p.suffix.lower() == ext)


def read_series_header(series_dir: Path, ext: str = ".dcm") -> dict:
    """Lee la cabecera del corte central de una serie (rápido: sin píxeles)."""
    import pydicom

    files = list_dicoms(series_dir, ext)
    out = {"n_slices": len(files)}
    if not files:
        return out
    ds = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True)
    desc = str(getattr(ds, "SeriesDescription", "") or getattr(ds, "ProtocolName", ""))
    so = getattr(ds, "ScanOptions", "")
    if isinstance(so, str):
        scan_opt = so
    else:  # pydicom devuelve una lista (MultiValue) cuando hay varias opciones
        try:
            scan_opt = "\\".join(str(x) for x in so)
        except TypeError:
            scan_opt = str(so)
    iop = getattr(ds, "ImageOrientationPatient", None)
    seq = classify_sequence(desc, scan_opt, getattr(ds, "RepetitionTime", None), getattr(ds, "EchoTime", None))
    ps = getattr(ds, "PixelSpacing", [None, None])
    out.update({
        "series_description": desc,
        "scan_options": scan_opt,
        "plane": plane_from_iop(iop) if iop is not None else "unknown",
        "fat_sat": seq["fat_sat"],
        "weighting": seq["weighting"],
        "tr": _f(getattr(ds, "RepetitionTime", None)),
        "te": _f(getattr(ds, "EchoTime", None)),
        "slice_thickness": _f(getattr(ds, "SliceThickness", None)),
        "pixel_spacing": _f(ps[0]) if ps else None,
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "cols": int(getattr(ds, "Columns", 0) or 0),
        "manufacturer": str(getattr(ds, "Manufacturer", "")),
        "field_strength": _f(getattr(ds, "MagneticFieldStrength", None)),
        "institution": str(getattr(ds, "InstitutionName", "")),
    })
    return out


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_series_volume(series_dir: Path, ext: str = ".dcm") -> np.ndarray:
    """
    Carga todos los cortes de una serie como un volumen (n_cortes, alto, ancho) en float32,
    ordenados espacialmente a lo largo de la normal del plano.
    """
    import pydicom

    files = list_dicoms(series_dir, ext)
    slices = []
    for f in files:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)
        # Escala real de intensidad si existe (RescaleSlope/Intercept)
        arr = arr * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        if arr.ndim == 3:  # algunos DICOM traen varios cuadros: tomamos el primero
            arr = arr[0]
        n = normal_from_iop(getattr(ds, "ImageOrientationPatient", None))
        ipp = getattr(ds, "ImagePositionPatient", None)
        if n is not None and ipp is not None:
            key = float(np.dot(np.asarray([float(v) for v in ipp]), n))
        else:
            key = float(getattr(ds, "InstanceNumber", len(slices)) or len(slices))
        slices.append((key, arr))
    if not slices:
        raise ValueError(f"Serie sin cortes: {series_dir}")
    # Todos los cortes deben tener el mismo tamaño; si alguno difiere, lo descartamos
    shapes = [s[1].shape for s in slices]
    common = max(set(shapes), key=shapes.count)
    slices = sorted([s for s in slices if s[1].shape == common], key=lambda t: t[0])
    return np.stack([s[1] for s in slices], axis=0)


def normalize_volume(vol: np.ndarray, n_slices: int, size: int, p_low: float, p_high: float) -> np.ndarray:
    """
    1) Ventana de intensidad por percentiles (robusta a valores extremos) -> rango 0..255.
    2) Re-muestreo a n_slices cortes uniformemente espaciados.
    3) Recorte cuadrado central y redimensión a size x size.
    Devuelve uint8 (n_slices, size, size): 24*256*256 = 1.5 MB por estudio.
    """
    import cv2

    lo, hi = np.percentile(vol, [p_low, p_high])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    idx = np.linspace(0, vol.shape[0] - 1, n_slices).round().astype(int)
    vol = vol[idx]
    _, h, w = vol.shape
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    vol = vol[:, top:top + s, left:left + s]
    out = np.stack([cv2.resize(sl, (size, size), interpolation=cv2.INTER_AREA) for sl in vol])
    return (out * 255).round().astype(np.uint8)
