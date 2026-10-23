# P2 · Segmentación de inundaciones (Sen1Floods11 + Lima 2017)

```
p2_inundaciones/
├── config.yaml        parámetros (rutas con {data_root}, {runs_root})
├── run_all.py         pipeline completo
├── scripts/           etapas s00 a s13
├── src/               lectura de chips, U-Net, pérdidas, métricas, ensambles y capas auxiliares
├── snap/              grafo de SNAP para procesar escenas Sentinel-1 GRD fuera de Earth Engine
└── reports/           tablas, figuras y resultados para el informe
```

Datos:
- Sen1Floods11 v1.1: `<data_root>/p2_sen1floods11/raw/v1.1/` (etapa 1).
- Copernicus EMS EMSN038: `<data_root>/p2_lima/ems/EMSN038_UTM18S_v0.gdb`.
- Sentinel-1 de Lima: Earth Engine (etapa 5) o GeoTIFF propios (etapa 5b).
