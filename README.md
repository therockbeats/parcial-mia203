# Examen Parcial · MIA-203 Redes Neuronales y Aprendizaje Profundo

Maestría en Inteligencia Artificial · Facultad de Ingeniería Industrial y de Sistemas ·
Universidad Nacional de Ingeniería · Semestre 2026-II

Repositorio del grupo con la implementación, los experimentos y los resultados de las tres
preguntas del examen parcial.

| Pregunta | Tarea | Arquitectura | Datos | Carpeta | Estado |
|---|---|---|---|---|---|
| P1 | Clasificación multietiqueta de 12 hallazgos en resonancia magnética de rodilla con supervisión débil | CNN 2.5D con agregación por atención: encoder propio y ResNet-50 preentrenada | RSNA Knee Abnormality Detection (Kaggle) | `p1_rodilla/` | experimentos completados |
| P2 | Segmentación de inundaciones en radar Sentinel-1 y generalización a Lima 2017 | U-Net propia (salto estándar, con atención o sin salto) | Sen1Floods11 + Sentinel-1 Lima 2017 + Copernicus EMSN038 | `p2_inundaciones/` | en desarrollo |
| P3 | Detección de fauna en cámaras trampa | YOLOv8/v9 | Caltech Camera Traps (LILA BC) | `p3_fauna/` | en desarrollo |

---

## Pregunta 1 · Clasificación multietiqueta de anomalías de rodilla

### Problema
Cada estudio de resonancia magnética de rodilla debe clasificarse respecto de 12 hallazgos no
excluyentes: lesión del ligamento cruzado anterior, lesión del ligamento colateral medial,
daño de menisco medial y lateral, osteoartrosis medial, lateral y patelofemoral, derrame
articular, sinovitis, quiste de Baker, contusión ósea y fractura. El conjunto contiene 4,407
estudios (24,371 series, 819,635 imágenes DICOM, 530.6 GB) de 16 centros; solo 58 estudios
tienen etiquetas de radiólogo, mientras que todos tienen su reporte en texto libre en
aproximadamente doce idiomas.

### Pipeline

| Etapa | Script | Descripción |
|---|---|---|
| Inventario | `s01_inventario` | Estructura de la descarga, esquema de tablas y muestra de cabeceras DICOM |
| Metadatos | `s02_metadatos_series` | Plano, supresión grasa, sensibilidad a líquido, ponderación y parámetros de adquisición por serie; concordancia entre la cabecera DICOM y la anotación oficial |
| Etiquetas débiles | `s03_etiquetas_reglas` | Extracción multilingüe basada en reglas con detección de negación e incertidumbre; acuerdo (kappa de Cohen, sensibilidad, especificidad) contra las 58 etiquetas doradas, global y por idioma |
| Análisis de errores | `s03b_errores_reglas` | Falsos positivos y negativos de las reglas en la partición de desarrollo de los estudios dorados |
| Etiquetas débiles (alternativa) | `s04_etiquetas_llm` | Extracción con un modelo de lenguaje local (Ollama) con salida restringida a un esquema JSON |
| Selección | `s05_seleccionar_estudios` | Una serie por estudio (sagital DP/T2 con supresión grasa como primera opción) y subconjunto estratificado de 500 estudios con un mínimo de positivos por hallazgo; partición entrenamiento / validación / dorado |
| Preprocesamiento | `s06_preprocesar` | Ordenamiento espacial de cortes, ventana de intensidad por percentiles, remuestreo a 24 cortes y 256×256 px |
| Entrenamiento | `s07_entrenar` | AdamW, warm-up y decaimiento coseno, precisión mixta bf16, BCE o focal ponderadas, early stopping por AUC macro en validación |
| Evaluación | `s08_evaluar` | AUC-ROC por hallazgo y macro con IC 95 % por bootstrap; precisión, recall y F1 con umbrales elegidos en validación |
| Interpretabilidad | `s09_gradcam` | Grad-CAM sobre el corte de mayor peso de atención |
| Ablación | `s10_ablacion` | Seis configuraciones (tabla abajo) |
| Desbalance y sesgo | `s11_desbalance_y_sesgo` | Prevalencias, AUC por hallazgo según prevalencia y tasa de positivos por idioma |
| Exportación | `s12_exportar_resultados` | Copia de métricas, curvas y mapas a `p1_rodilla/reports/` |
| Comparación pareada | `s13_comparacion_pareada` | Diferencia de AUC macro frente a la referencia con IC 95 % por bootstrap pareado |
| Semillas | `s14_semillas` | Corridas adicionales con otras semillas; media y desviación estándar del AUC macro y ensamble de semillas comparado con la referencia |
| Escala de datos | `s15_escala_datos` | Entrenamiento con subconjuntos anidados de 425 a todos los estudios con etiqueta débil, validación y conjunto dorado fijos; curva de AUC frente al tamaño |

### Arquitectura
Cada volumen se representa como una secuencia de cortes; cada corte, junto con sus dos
vecinos como canales (representación 2.5D), pasa por un encoder convolucional 2D compartido.
Los vectores de los cortes se agregan con atención con compuerta (aprendizaje de múltiples
instancias, Ilse et al., 2018) y una capa lineal produce 12 logits con activación sigmoide
independiente. Se comparan dos encoders:

- **Modelo A:** red residual diseñada e inicializada desde cero (~2.8 M parámetros).
- **Modelo B:** ResNet-50 preentrenada en ImageNet con ajuste fino completo.

### Experimentos de ablación

| Experimento | Factor que varía |
|---|---|
| `A_scratch_bce` | encoder desde cero |
| `B_resnet50_bce` | referencia |
| `C_resnet50_focal` | pérdida focal |
| `D_resnet18_bce` | profundidad del encoder |
| `E_resnet50_ignorar_no_mencionados` | tratamiento de hallazgos no mencionados en el reporte |
| `F_resnet50_sobremuestreo` | sobremuestreo de hallazgos poco frecuentes |

Todos los hiperparámetros están en `p1_rodilla/config.yaml`.

---

## Pregunta 2 · Mapeo de inundaciones con Sentinel-1

### Problema
Segmentar, píxel a píxel, el agua de inundación en imágenes de radar Sentinel-1 (VV, VH). El
conjunto Sen1Floods11 contiene 446 chips de 512 × 512 px con etiqueta experta (11 eventos) y
4,385 chips con etiquetas débiles generadas por umbralización (Otsu sobre VH de Sentinel-1 o
índices de agua de Sentinel-2). La generalización se evalúa en una región propia de Lima
(marzo de 2017): Sentinel-1B del 17/03/2017 y la delineación de inundación de Copernicus EMS
EMSN038 (WorldView-2 del 18 y 21/03/2017).

### Pipeline

| Etapa | Script | Descripción |
|---|---|---|
| Entorno | `s00_verificar_entorno` | PyTorch y GPU, librerías geoespaciales, controlador OpenFileGDB y Earth Engine |
| Descarga | `s01_descargar_sen1floods11` | Descarga del bucket público `gs://sen1floods11` (v1.1) mediante su API JSON |
| Inventario | `s02_inventario` | Archivos por carpeta, particiones oficiales, chips por evento y cabeceras |
| Chips | `s03_preparar_chips` | Caché en .npy (S1 en dB, etiquetas, bandas S2), fracción de agua, sin dato y nubes |
| Etiquetas débiles | `s04_calidad_etiquetas` | Otsu S1 frente a la etiqueta experta (ruido y línea base), Otsu S1 frente a índice S2, agua permanente JRC |
| Lima S1 | `s05_lima_s1_gee` | Escenas posterior y previa desde Earth Engine (`COPERNICUS/S1_GRD`), ángulo de incidencia y SRTM sobre la grilla de chips |
| Lima S1 (alternativa) | `s05b_lima_s1_geotiff` | Chips desde GeoTIFF propios (grafo SNAP en `p2_inundaciones/snap/`) |
| Lima etiquetas | `s06_lima_etiquetas` | Rasterización de EMSN038 (inundación, zonas ignoradas, origen del polígono y cobertura del suelo) |
| Entrenamiento | `s07_entrenar` | Presupuesto fijo de pasos, AdamW, calentamiento y coseno, bf16, mejor modelo por IoU de validación |
| Evaluación | `s08_evaluar` | Umbral elegido en validación; IoU, F1, precisión, sensibilidad, omisión, comisión e IC 95 % en prueba, Bolivia y Lima |
| Ablación | `s09_ablacion` | Diez configuraciones con tres semillas, ensambles y comparación pareada con la referencia |
| Generalización | `s10_generalizacion` | Brecha frente a Bolivia y Lima y cuantificación de hipótesis en Lima |
| Fallos del SAR | `s11_fallos_sar` | Categorías de error (bordes, superficies oscuras, retrodispersión alta) con ejemplos visuales |
| Fusión óptica | `s12_fusion_optica` | SAR frente a SAR + Sentinel-2 en chips de baja nubosidad |
| Exportación | `s13_exportar_resultados` | Copia de métricas y figuras a `p2_inundaciones/reports/` |

### Arquitectura
U-Net implementada desde cero en PyTorch: bloques de dos convoluciones 3 × 3 con normalización
por lotes, cuatro submuestreos (32 a 512 canales), convoluciones transpuestas y un logit por
píxel. Las conexiones de salto son configurables: estándar, filtradas por compuertas de atención
(Attention U-Net) o ausentes.

### Experimentos de ablación

| Experimento | Factor que varía |
|---|---|
| `G_bcedice` | solo chips dorados |
| `W_otsu` | solo etiquetas débiles de Otsu sobre S1 |
| `W_s2index` | solo etiquetas débiles del índice de agua de S2 |
| `GW_bcedice` | referencia: dorados y débiles (Otsu S1) mezclados al 50 % por lote, BCE + Dice |
| `WG_ajuste` | preentrenamiento con débiles y ajuste fino con dorados |
| `GW_bce`, `GW_dice` | función de pérdida |
| `GW_sin_skip`, `GW_atencion` | diseño de las conexiones de salto |
| `GW_prof3` | profundidad de la red |

Pipeline completo de la Pregunta 2: `python -m p2_inundaciones.run_all`; prueba con datos
sintéticos: `python tests/smoke_test_p2.py`.

---

## Requerimientos

- Python 3.11 y conda (Miniforge).
- Entrenamiento: GPU NVIDIA con CUDA 12.8 o superior y PyTorch ≥ 2.7.
- Desarrollo y pruebas: también funciona en Apple Silicon (MPS) o CPU.

```bash
conda env create -f environment.yml        # Windows/Linux con GPU NVIDIA
conda env create -f environment-mac.yml    # macOS (Apple Silicon)
conda activate mia203
```

### Datos
Los datos no se incluyen en el repositorio por su tamaño y por las reglas de uso de la
competencia. Se descargan desde Kaggle (*RSNA Knee Abnormality Detection*) y se ubican en
`<data_root>/p1_rsna_knee/raw/`, con la estructura original:

```
raw/train.csv
raw/train_series.csv
raw/train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
```

La ruta `data_root` (y `runs_root` para checkpoints) se define por máquina:
```bash
cp configs/paths.example.yaml configs/paths.local.yaml   # y editar las rutas
```

## Reproducción

Pipeline completo de la Pregunta 1:
```bash
python -m p1_rodilla.run_all
```

Etapas individuales (por ejemplo, un experimento concreto):
```bash
python -m p1_rodilla.scripts.s07_entrenar --exp B_resnet50_bce
python -m p1_rodilla.scripts.s08_evaluar  --exp B_resnet50_bce
```

Verificación del entorno y prueba de extremo a extremo con datos sintéticos (sin datos reales):
```bash
python -m p1_rodilla.scripts.s00_verificar_entorno
python tests/smoke_test_p1.py
```

Las semillas son fijas (`seed` en el config) y cada experimento guarda su configuración
efectiva junto a sus resultados.

## Estructura

```
├── common/                  configuración, semillas, dispositivo y registro
├── configs/                 plantilla de rutas por máquina
├── p1_rodilla/
│   ├── config.yaml          parámetros de la Pregunta 1
│   ├── run_all.py           pipeline completo
│   ├── scripts/             etapas s00 a s15
│   ├── src/                 DICOM, etiquetado, dataset, modelos, pérdidas, métricas, Grad-CAM
│   └── reports/             tablas, figuras y resultados exportados
├── p2_inundaciones/         Pregunta 2: config, run_all, scripts s00 a s13, src (U-Net, datos, métricas), snap
├── p3_fauna/                Pregunta 3
├── informe/                 artículos en formato NeurIPS
├── presentacion/            diapositivas
└── tests/                   datos sintéticos y prueba de extremo a extremo
```

## Integrantes

| Nombre | Código | Responsabilidad principal |
|---|---|---|
| Jose Miguel Alvarado Moreno | 20266834E | P1: pipeline de datos, arquitectura 2.5D y entrenamiento en GPU. P2: pipeline de Sen1Floods11 y de la región de Lima, U-Net y experimentos |
| (completar) | | |
| (completar) | | |
| (completar) | | |
| (completar) | | |

---

## Declaración sobre el uso de herramientas de asistencia

En conformidad con los lineamientos académicos del curso, se utilizaron modelos de lenguaje de gran escala (LLMs) como herramientas de consulta técnica, apoyo en la depuración de scripts y revisión de estilo en la documentación técnica. El diseño metodológico, la formulación de reglas clínicas, la estructuración de la arquitectura 2.5D, la ejecución de los experimentos en GPU y el análisis crítico de las métricas fueron realizados y validados íntegramente por los autores del trabajo.
