# Examen Parcial · MIA-203 Redes Neuronales y Aprendizaje Profundo

Maestría en Inteligencia Artificial · Facultad de Ingeniería Industrial y de Sistemas ·
Universidad Nacional de Ingeniería · Semestre 2026-II

Repositorio del grupo con la implementación, los experimentos y los resultados de las tres
preguntas del examen parcial.

| Pregunta | Tarea | Arquitectura | Datos | Carpeta | Estado |
|---|---|---|---|---|---|
| P1 | Clasificación multietiqueta de 12 hallazgos en resonancia magnética de rodilla con supervisión débil | CNN 2.5D con agregación por atención: encoder propio y ResNet-50 preentrenada | RSNA Knee Abnormality Detection (Kaggle) | `p1_rodilla/` | en ejecución |
| P2 | Segmentación de inundaciones en radar Sentinel-1 y generalización a Lima 2017 | U-Net propia | Sen1Floods11 + Copernicus EMSN038 | `p2_inundaciones/` | en desarrollo |
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
| Etiquetas débiles (alternativa) | `s04_etiquetas_llm` | Extracción con un modelo de lenguaje local (Ollama) con salida restringida a un esquema JSON |
| Selección | `s05_seleccionar_estudios` | Una serie por estudio (sagital DP/T2 con supresión grasa como primera opción) y subconjunto estratificado de 500 estudios con un mínimo de positivos por hallazgo; partición entrenamiento / validación / dorado |
| Preprocesamiento | `s06_preprocesar` | Ordenamiento espacial de cortes, ventana de intensidad por percentiles, remuestreo a 24 cortes y 256×256 px |
| Entrenamiento | `s07_entrenar` | AdamW, warm-up y decaimiento coseno, precisión mixta bf16, BCE o focal ponderadas, early stopping por AUC macro en validación |
| Evaluación | `s08_evaluar` | AUC-ROC por hallazgo y macro con IC 95 % por bootstrap; precisión, recall y F1 con umbrales elegidos en validación |
| Interpretabilidad | `s09_gradcam` | Grad-CAM sobre el corte de mayor peso de atención |
| Ablación | `s10_ablacion` | Seis configuraciones (tabla abajo) |
| Desbalance y sesgo | `s11_desbalance_y_sesgo` | Prevalencias, AUC por hallazgo según prevalencia y tasa de positivos por idioma |
| Exportación | `s12_exportar_resultados` | Copia de métricas, curvas y mapas a `p1_rodilla/reports/` |

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
│   ├── scripts/             etapas s00 a s12
│   ├── src/                 DICOM, etiquetado, dataset, modelos, pérdidas, métricas, Grad-CAM
│   └── reports/             tablas, figuras y resultados exportados
├── p2_inundaciones/         Pregunta 2
├── p3_fauna/                Pregunta 3
├── informe/                 artículos en formato NeurIPS
├── presentacion/            diapositivas
└── tests/                   datos sintéticos y prueba de extremo a extremo
```

## Integrantes

| Nombre | Código | Responsabilidad principal |
|---|---|---|
| Jose Miguel Alvarado Moreno | 20266834E | P1: Pipeline de datos, arquitectura 2.5D y entrenamiento GPU |
| (completar) | | |
| (completar) | | |
| (completar) | | |
| (completar) | | |

---

## Declaración sobre el uso de herramientas de asistencia

Durante el desarrollo de este trabajo se utilizó un asistente de inteligencia artificial
generativa (Claude, de Anthropic) como herramienta de apoyo. El asistente se empleó para
proponer la estructura del repositorio, generar una primera versión del código del pipeline
de la Pregunta 1 y de su configuración, y redactar documentación de apoyo para la
preparación del entorno.

El equipo es responsable de la verificación y adaptación del código a los datos reales, la
ejecución de todos los experimentos, la interpretación y el análisis de los resultados, y la
redacción del informe. Todos los integrantes revisaron el código y asumen la responsabilidad
por su contenido.
