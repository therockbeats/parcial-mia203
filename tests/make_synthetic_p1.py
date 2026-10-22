"""
Genera un mini conjunto de datos FALSO con la misma estructura REAL de Kaggle
(verificada con el inventario del 22/10/2026: train.csv con StudyInstanceUID, Report y 12
columnas de etiqueta; train_series.csv con plano/supresión grasa/sensibilidad a líquido;
imágenes en train_series/<estudio>/<serie>/*.dcm),
para probar el pipeline de la Pregunta 1 de punta a punta en minutos (sin GPU).

    python tests/make_synthetic_p1.py --out /ruta/data/p1_rsna_knee/raw --studies 60

No reemplaza a los datos reales: solo verifica que el código corre sin errores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

LABELS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa", "pf_oa",
          "effusion", "synovitis", "baker_cyst", "bone_contusion", "fracture"]
# Nombres de columna tal como vienen en train.csv de Kaggle
KAGGLE_COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
               "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PHRASES = {
    "en": {"acl": ("Complete tear of the anterior cruciate ligament.", "The ACL is intact."),
           "effusion": ("Moderate joint effusion.", "No joint effusion."),
           "medial_meniscus": ("Tear of the posterior horn of the medial meniscus.", "Medial meniscus intact."),
           "fracture": ("Non-displaced fracture of the lateral tibial plateau.", "No fracture."),
           "baker_cyst": ("Small Baker cyst.", "No popliteal cyst.")},
    "es": {"acl": ("Rotura completa del ligamento cruzado anterior.", "LCA íntegro."),
           "effusion": ("Derrame articular moderado.", "Sin derrame articular."),
           "medial_meniscus": ("Rotura del cuerno posterior del menisco interno.", "Menisco interno íntegro."),
           "fracture": ("Fractura no desplazada de la meseta tibial.", "No se observan fracturas."),
           "baker_cyst": ("Quiste de Baker.", "Sin quiste poplíteo.")},
    "de": {"acl": ("Ruptur des vorderen Kreuzbandes.", "Vorderes Kreuzband intakt."),
           "effusion": ("Deutlicher Gelenkerguss.", "Kein Gelenkerguss."),
           "medial_meniscus": ("Riss im Innenmeniskus Hinterhorn.", "Innenmeniskus regelrecht."),
           "fracture": ("Fraktur des Tibiakopfes.", "Keine Fraktur."),
           "baker_cyst": ("Bakerzyste.", "Keine Bakerzyste.")},
}


def write_series(folder: Path, n_slices: int, size: int, plane: str, desc: str, signal: float, rng):
    folder.mkdir(parents=True, exist_ok=True)
    iop = {"sagittal": [0, 1, 0, 0, 0, -1], "coronal": [1, 0, 0, 0, 0, -1], "axial": [1, 0, 0, 0, 1, 0]}[plane]
    normal = np.cross(iop[:3], iop[3:])
    study_uid, series_uid = generate_uid(), generate_uid()
    order = rng.permutation(n_slices)   # archivos desordenados a propósito: el código debe ordenar por posición
    for k, s in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"{k}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.Modality, ds.SeriesDescription = "MR", desc
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.ImageOrientationPatient = [str(v) for v in iop]
        ds.ImagePositionPatient = [str(v) for v in (normal * s * 3.0)]
        ds.InstanceNumber = int(k + 1)
        ds.RepetitionTime, ds.EchoTime = "3000", "35"
        ds.SliceThickness, ds.PixelSpacing = "3.0", ["0.4", "0.4"]
        ds.Rows = ds.Columns = size
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        img = rng.normal(500, 80, (size, size)) + signal * 400 * np.exp(-((np.arange(size) - size / 2) ** 2) / 200)[None, :]
        ds.PixelData = np.clip(img, 0, 4095).astype(np.uint16).tobytes()
        ds.save_as(folder / f"{k}.dcm", enforce_file_format=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--studies", type=int, default=60)
    ap.add_argument("--gold", type=int, default=14)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(args.out)
    rows, series_rows = [], []
    for i in range(args.studies):
        sid = f"{100000 + i}"
        truth = {l: int(rng.random() < 0.3) for l in LABELS}
        lang = ["en", "es", "de"][i % 3]
        text = " ".join(PHRASES[lang][l][0 if truth[l] else 1] for l in PHRASES[lang])
        row = {"StudyInstanceUID": sid, "Report": text}
        row.update({k: (float(truth[l]) if i < args.gold else np.nan) for l, k in zip(LABELS, KAGGLE_COLS)})
        rows.append(row)
        # (plano, descripción DICOM, supresión grasa, sensible a líquido); una descripción anonimizada a propósito
        for j, (plane, desc, fs, fluid) in enumerate([("sagittal", "DummySeriesDesc!", 1, 1),
                                                      ("coronal", "COR T1", 0, 0),
                                                      ("axial", "AX PD FS", 1, 1)]):
            ser = f"{sid}{j}"
            write_series(out / "train_series" / sid / ser, 18, 96, plane, desc, truth["effusion"], rng)
            series_rows.append({"StudyInstanceUID": sid, "SeriesInstanceUID": ser, "Fluid_Sensitive": fluid,
                                "Fat_Suppression": fs, "Anatomical_Plane": plane.capitalize()})
    pd.DataFrame(rows).to_csv(out / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(out / "train_series.csv", index=False)
    print(f"Datos sintéticos en {out}")


if __name__ == "__main__":
    main()
