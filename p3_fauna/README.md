# P3 · Detección de fauna en cámaras trampa (Caltech Camera Traps, CCT-20)

```
p3_fauna/
├── config.yaml        parámetros (rutas con {data_root}, {runs_root})
├── run_all.py         pipeline completo
├── scripts/           etapas s00 a s10
├── src/               anotaciones COCO Camera Traps, formato YOLO, muestreo, métricas, Grad-CAM y reglas de monitoreo
├── notebooks/
├── reports/           tablas, figuras y resultados para el informe
└── GUIA_ESTUDIO.md    explicación de las decisiones del pipeline
```

Datos (LILA BC, sección *CCT20 benchmark*):
- Imágenes (~6 GB): `<data_root>/p3_cct20/raw/eccv_18_all_images_sm/`
- Metadatos de particiones: `<data_root>/p3_cct20/raw/eccv_18_annotation_files/`
  (`train`, `cis_val`, `trans_val`, `cis_test`, `trans_test` `_annotations.json`)

La etapa 0 comprueba que estén todos los archivos:
```bash
python -m p3_fauna.scripts.s00_verificar_entorno
```

Pipeline completo: `python -m p3_fauna.run_all`; prueba con datos sintéticos: `python tests/smoke_test_p3.py`.
