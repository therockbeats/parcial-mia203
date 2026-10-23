# P3 · Detección de fauna en cámaras trampa (Caltech Camera Traps, CCT-20)

Estado: **en desarrollo**.

Estructura prevista (igual que la P1):
```
p3_fauna/
├── config.yaml        parámetros
├── run_all.py         pipeline completo
├── scripts/           conversión COCO → YOLO, entrenamiento, evaluación, reporte de monitoreo
├── src/
├── notebooks/
└── reports/
```
Datos: `<data_root>/p3_cct20/raw/` (imágenes y anotaciones COCO de LILA BC).
