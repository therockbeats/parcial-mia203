"""
Lectura de series DICOM de RM: metadatos, plano, secuencia y volumen ordenado.

* Plano: dirección dominante de la normal n = r x c, con r y c tomados de
  ImageOrientationPatient (x: sagital, y: coronal, z: axial).
* Orden de cortes: proyección de ImagePositionPatient sobre n.
* Secuencia: supresión grasa y ponderación (PD/T2/T1) a partir de SeriesDescription y
  ScanOptions; si no hay texto, por TR/TE.
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
    """Plano anatómico (sagittal, coronal, axial) a partir de ImageOrientationPatient."""
    try:
        v = np.asarray([float(x) for x in iop], dtype=float)
        r, c = v[:3], v[3:]
        n = np.cross(r, c)
        axis = int(np.argmax(np.abs(n)))
        return ["sagittal", "coronal", "axial"][axis]
    except Exception:
        return "unknown"


def normal_from_iop(iop) -> np.ndarray | None:
    """Normal unitaria al plano de corte, o None si la orientación no es válida."""
    try:
        v = np.asarray([float(x) for x in iop], dtype=float)
        n = np.cross(v[:3], v[3:])
        return n / (np.linalg.norm(n) + 1e-8)
    except Exception:
        return None


def classify_sequence(description: str, scan_options: str = "", tr=None, te=None) -> dict:
    """
    Supresión grasa y ponderación de una serie.

    Se usa el texto de SeriesDescription y ScanOptions; si no identifica la ponderación,
    se aplican los tiempos (ms): T1 si TR < 900 y TE < 30, T2 si TE >= 60, PD si TR >= 900.
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
    """Archivos DICOM de una serie, ordenados por nombre."""
    return sorted(p for p in Path(series_dir).iterdir() if p.suffix.lower() == ext)


def read_series_header(series_dir: Path, ext: str = ".dcm") -> dict:
    """Metadatos de la serie a partir de la cabecera del corte central (sin leer píxeles)."""
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
    else:  # MultiValue
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
    """Conversión a float tolerante a valores ausentes."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_series_volume(series_dir: Path, ext: str = ".dcm") -> np.ndarray:
    """
    Volumen (n_cortes, alto, ancho) en float32 con los cortes ordenados a lo largo de la
    normal al plano. Se descartan los cortes con dimensiones distintas a la mayoritaria.
    """
    import pydicom

    files = list_dicoms(series_dir, ext)
    slices = []
    for f in files:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)
        # RescaleSlope / RescaleIntercept
        arr = arr * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        if arr.ndim == 3:  # multiframe: primer cuadro
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
    shapes = [s[1].shape for s in slices]
    common = max(set(shapes), key=shapes.count)
    slices = sorted([s for s in slices if s[1].shape == common], key=lambda t: t[0])
    return np.stack([s[1] for s in slices], axis=0)


def normalize_volume(vol: np.ndarray, n_slices: int, size: int, p_low: float, p_high: float) -> np.ndarray:
    """
    Normalización de un volumen a uint8 (n_slices, size, size).

    Ventana de intensidad entre los percentiles ``p_low`` y ``p_high``, remuestreo uniforme
    a ``n_slices`` cortes, recorte cuadrado central y redimensión a ``size`` x ``size``.
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
