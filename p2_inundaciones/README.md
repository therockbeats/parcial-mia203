# P2 · Segmentación de inundaciones (Sen1Floods11 + Lima 2017)

Estado: **en desarrollo**.

Estructura prevista (igual que la P1):
```
p2_inundaciones/
├── config.yaml        parámetros (rutas con {data_root}, {runs_root})
├── run_all.py         pipeline completo
├── scripts/           s00 ... sNN
├── src/               dataset Sentinel-1 (VV, VH), U-Net propia, métricas IoU/F1
├── notebooks/
└── reports/           tablas y figuras para el informe
```
Ubicación de los datos: `D:\MIA203\data\p2_sen1floods11\raw\` (subconjunto hand-labeled
de Sen1Floods11) y la delimitación Copernicus EMSN038 de Lima 2017.
