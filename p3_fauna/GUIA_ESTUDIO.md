# Guía de estudio — Pregunta 3: fauna en cámaras trampa con YOLO

Mapa rápido del código: los parámetros están en `config.yaml`, cada etapa en `scripts/sNN_*.py` (se ejecutan con `python -m p3_fauna.scripts.<etapa>`) y la lógica reutilizable en `src/`. Las rutas a los datos salen de `configs/paths.local.yaml`, igual que en P1 y P2.

Esta guía explica **por qué** el código está hecho así. La idea es que puedas rehacer un
proyecto parecido (otro dataset, otro detector) sin mirar el código. Una advertencia práctica:
el enunciado sanciona copiar cuadernos públicos, así que úsalo como material para entender y
escribe tu propia versión cuando te toque entregar.

---

## 1. Cómo descomponer el enunciado

Antes de escribir código, convierte cada entregable en una pregunta medible y en un archivo:

| # | Lo que piden | Pregunta que responde | Dónde |
|---|---|---|---|
| 1 | YOLO con decisiones documentadas | ¿Qué elegí y por qué? | `config.yaml`, `s02_convertir_yolo`, `s03_entrenar`, `s05_ablacion` |
| 2 | mAP@0.5 y @0.5:0.95 por especie | ¿Qué tan bien detecta y qué especies fallan? | `s04_evaluar`, `s06_desbalance` |
| 3 | Brecha cámaras vistas vs nuevas | ¿Generaliza o memoriza el fondo? | `s04_evaluar` + `s08_fondo_gradcam` |
| 4 | Oclusión y tamaño | ¿Dónde se rompe? | `s07_robustez` |
| 5 | Reporte ecológico | ¿Sirve para un biólogo? | `s09_monitoreo` (reglas en `src/monitoreo.py`) |
| 6 | Uso responsable | ¿Cuáles son los costos de equivocarse? | informe |

Regla general: **todo número del informe debe salir de un script**, y todo hiperparámetro de un archivo de configuración. Así cumples el requisito de "un único comando": `python -m p3_fauna.run_all`.

---

## 2. Conceptos que debes dominar

### 2.1 Detección vs clasificación
Un clasificador responde "¿qué hay en la imagen?". Un detector responde "¿qué hay y **dónde**?"
con cajas `(x, y, w, h)` + clase + confianza. En cámaras trampa la detección importa porque
permite **contar individuos** y porque obliga al modelo a mirar al animal (en teoría).

### 2.2 YOLOv8 en una página
- **Una sola etapa**: una pasada por la red produce todas las cajas (no hay propuesta de regiones como en Faster R-CNN). Por eso es rápido.
- **Backbone** (CSPDarknet con bloques C2f) → extrae rasgos. El índice 9 es el SPPF, fin del backbone.
- **Neck** (PAN-FPN) → mezcla rasgos de varias escalas. Salidas P3 (stride 8, objetos pequeños, capa 15), P4 (capa 18), P5 (stride 32, objetos grandes, capa 21).
- **Head desacoplada y sin anclas** (*anchor-free*): cada celda predice directamente la distancia a los 4 bordes de la caja y, por separado, la probabilidad de cada clase.
- **Pérdida** = caja (CIoU + Distribution Focal Loss) + clase (BCE). La asignación de qué predicción "le toca" a cada objeto real la hace el *Task-Aligned Assigner*.
- **Postproceso**: NMS (supresión de no-máximos) elimina cajas duplicadas.

### 2.3 Métricas
- **IoU** = área de intersección / área de unión entre caja predicha y real.
- **AP** de una clase = área bajo la curva precisión-recall, contando como acierto las predicciones con IoU ≥ umbral.
- **mAP@0.5**: promedio de AP sobre clases con IoU ≥ 0.5 (tolerante con la localización).
- **mAP@0.5:0.95**: promedio de AP en umbrales 0.50, 0.55 … 0.95 (exige cajas precisas).
- Se calculan con `conf=0.001` porque la curva PR necesita *todas* las predicciones. Para el despliegue (monitoreo) se usa un umbral real (`deploy_conf=0.35`), que debes justificar mirando la curva PR o F1.

Pregunta típica de examen: *¿por qué el mAP puede ser alto y el sistema inútil?* → porque el mAP promedia clases por igual y no pesa las imágenes vacías; con 70 % de vacías, la tasa de falsos positivos por imagen vacía es lo que decide el trabajo humano. Por eso aquí se evalúa con **todas** las vacías incluidas.

---

## 3. Datos: lo que hay que saber de CCT20

- El enunciado describe el Caltech Camera Traps **completo** (243,100 imágenes, 140 cámaras). El benchmark del paper original (**CCT20**) es un subconjunto: ~57,868 imágenes de 20 cámaras, 15 especies, con cajas. Es el que tiene sentido usar en 5 semanas, y hay que **decirlo en el informe**.
- CCT20 viene en cinco splits: `train`, `cis_val`, `cis_test` (cámaras vistas) y `trans_val`, `trans_test` (cámaras nunca vistas). Es exactamente lo que pide el entregable 3.
- Formato **COCO Camera Traps** (JSON): `images` (con `location`, `date_captured`, `seq_id`), `annotations` (con `bbox` en píxeles `[x, y, w, h]`) y `categories`. Las vacías tienen la categoría `empty` sin caja.
- Las imágenes del benchmark están reducidas a ≤1024 px. `s02_convertir_yolo` lee el tamaño real y reescala las cajas si no coincide con el JSON (evita un error silencioso muy común).

### Formato YOLO
Un `.txt` por imagen, una línea por objeto: `clase xc yc w h`, todo normalizado a [0, 1].
**Imagen sin `.txt` = fondo (background)**: así se manejan las vacías explícitamente, como pide el enunciado.

### ¿Cuántas vacías meter en entrenamiento?
- 0 % → el modelo nunca ve "nada" y dispara sobre ramas y sombras.
- 70 % (todas) → la mayoría de los lotes no enseña nada sobre especies; entrenamiento lento.
- Elegimos 25 % respecto a las imágenes con animal (configurable). **Esto es una ablación natural** para el informe (experimento `C_sin_vacias` frente a `B_yolov8s`; `s06_desbalance` compara mAP *y* falsos positivos en vacías). Puedes agregar más proporciones como experimentos en `config.yaml`.

---

## 4. Decisiones de entrenamiento (cómo justificarlas)

| Decisión | Valor | Justificación |
|---|---|---|
| Modelo | YOLOv8s preentrenado en COCO | COCO ya incluye animales; "s" cabe en una T4 gratuita. Comparar con "n" y "m" si hay tiempo. |
| Resolución | 640 | Las imágenes son ≤1024; los animales lejanos son pequeños. 1024 mejora pequeños pero cuadruplica costo → ablación. |
| Mosaic + close_mosaic | 1.0 / últimas 10 épocas | Mosaic combina 4 imágenes: más contextos y fondos distintos por imagen (ayuda contra memorizar fondos). Se apaga al final para ajustarse a imágenes reales. |
| flipud | 0 | Los animales no aparecen de cabeza. |
| HSV | s=0.5, v=0.4 | Hay imágenes diurnas a color e infrarrojas nocturnas casi grises. |
| MixUp | 0.1 | Regularización ligera. |
| Optimizador | AdamW, lr 1e-3, coseno | Estable para fine-tuning. |
| Early stopping | patience 15 sobre **cis_val** | Nunca seleccionar con trans: eso filtraría información del test de generalización. |

**Error conceptual clave:** si eliges hiperparámetros mirando `trans_test`, tu experimento de generalización deja de ser válido.

---

## 5. Desbalance de clases

Dos niveles: (a) vacías vs con fauna, (b) especies comunes vs raras.

**Repeat Factor Sampling** (LVIS, Gupta et al. 2019): para cada clase `c`, `f_c` = fracción de imágenes de entrenamiento que la contienen; `r_c = max(1, sqrt(t / f_c))` con `t = 0.1`. Cada imagen se repite `max_c r_c` veces por época. Ventaja: no toca la pérdida ni el código de Ultralytics; solo la lista de imágenes.

Alternativas que puedes mencionar o probar: pesos por clase en la pérdida, *focal loss* (Lin et al. 2017), aumento dirigido (copy-paste de animales raros sobre fondos vacíos).

**Cómo reportarlo:** tabla de prevalencia por split + gráfico AP vs nº de cajas de entrenamiento (escala log) + correlación de Spearman. Si la correlación es alta y positiva, el desbalance explica buena parte del rendimiento por especie.

---

## 6. Generalización a cámaras nuevas: el corazón del problema

**Por qué pasa:** cada cámara es fija. Si los coyotes casi siempre salen en la cámara 38, la forma más fácil de bajar la pérdida es aprender "este tronco → coyote". Es *shortcut learning* (Geirhos et al. 2020), documentado específicamente para cámaras trampa por Beery et al. (2018, *Recognition in Terra Incognita*).

**Qué medir:**
1. La brecha: mAP(cis_test) − mAP(trans_test), global y por especie.
2. **Evidencia** de que la causa es el fondo, no solo "datos distintos". `s08_fondo_gradcam` da tres:
   - **Grad-CAM** (Selvaraju et al. 2017), implementado a mano: se toma la probabilidad máxima de la clase real sobre todas las anclas, se deriva respecto a las activaciones de una capa, se promedian los gradientes por canal (importancia) y se combina con las activaciones. Se calcula en la capa 9 (backbone, como pide el enunciado) y en la 15 (P3, más resolución).
   - **Energy pointing**: (energía del mapa dentro de las cajas / energía total) ÷ (área de las cajas / área total). >1 = mira al animal. Comparar distribuciones cis vs trans.
   - **Prueba de enmascarado**: se borra el animal con desenfoque y se vuelve a predecir. Si en cámaras conocidas el modelo sigue diciendo "coyote" sin coyote, es evidencia directa de que usa el contexto.

**Cómo escribirlo:** no basta con mostrar mapas bonitos. Presenta 3-4 ejemplos cualitativos *y* las métricas agregadas; discute casos en que el mapa sí cae en el animal pero igual falla (p. ej., noche/IR).

**Mitigaciones para la discusión:** más cámaras distintas en entrenamiento (más importante que más imágenes), recortar el animal con un detector genérico (MegaDetector) y clasificar el recorte, aumento de fondos (pegar animales sobre vacías de otras cámaras), sustracción de fondo usando la secuencia.

---

## 7. Robustez: oclusión y tamaño

- **Tamaño** es fácil: área relativa de la caja; umbrales 1 % y 10 % del área de imagen.
- **Oclusión no viene anotada en CCT.** Opciones honestas:
  1. Proxy automático: caja truncada por el borde + solape con otras cajas (lo que hace el código).
  2. Anotar a mano una muestra (200-300 cajas, se reparte entre los 5 integrantes en una tarde) en `p3_fauna/reports/tablas/oclusion_manual.csv` con columnas `image_id,ann_idx,occlusion`.
  3. Reportar el **acuerdo** entre proxy y anotación manual: así conviertes una limitación en un resultado.
- Se reportan dos métricas por grupo: recall por caja (sirve aunque una imagen tenga cajas de grupos distintos) y mAP sobre las imágenes "puras" del grupo.

---

## 8. Módulo de monitoreo ecológico

Lo que importa aquí es **traducir detecciones a unidades que usa un ecólogo**:

- **Secuencia**: la cámara dispara ráfagas (2-10 fotos). Contar fotos infla la abundancia.
- **MaxN**: nº máximo de individuos visibles en un solo fotograma de la secuencia → estimación conservadora de cuántos animales había.
- **Evento independiente**: misma especie, misma cámara, separados por > 30 min. Es el criterio estándar en la literatura de fototrampeo.
- **RAI** (índice de abundancia relativa) = eventos independientes / días-cámara × 100.
- **Tasa de vacías**: lo que ahorra el sistema al biólogo.

Buena práctica que suma puntos: calcular el mismo reporte con las **etiquetas reales** de esa cámara y compararlo (el código lo hace). Así muestras si los errores del detector cambian las conclusiones ecológicas, que es lo que realmente importa.

---

## 9. Ideas para la discusión de uso responsable (entregable 6)

- **Falso negativo** de especie amenazada: la especie "desaparece" del registro, se subestima su presencia y puede retirarse protección. Costo alto e invisible (nadie revisa lo que el modelo descartó como vacío).
- **Falso positivo**: minutos de un especialista. Costo bajo y visible.
- Consecuencia: para especies sensibles conviene un **umbral más bajo** (más recall) y revisión humana de todo lo que el modelo marque, más un muestreo aleatorio de las imágenes "vacías" para estimar cuántas se pierden.
- **Cambio de región**: CCT es del suroeste de EE. UU.; en la Amazonía o los Andes las especies ni siquiera existen en las clases del modelo (problema de *conjunto abierto*), los fondos son selva densa (más oclusión) y las cámaras distintas. Sin validación local, el modelo puede reportar con confianza una especie equivocada. Mencionar: validar con una muestra etiquetada local, reentrenar o usar un detector genérico + clasificador local.

---

## 10. Estructura del informe (formato NeurIPS, 6-10 páginas)

1. **Motivación** (½ pág.): problema de escala en fototrampeo.
2. **Trabajos relacionados** (≥5 referencias, abajo).
3. **Datos** — CCT20, splits cis/trans, prevalencias, manejo de vacías.
4. **Metodología** — YOLOv8, decisiones de la tabla §4, RFS, pruebas de fondo, módulo de monitoreo.
5. **Experimentos y resultados** — tabla global, tabla por especie, AP vs frecuencia, brecha cis/trans, ablaciones, robustez, Grad-CAM, reporte de cámara.
6. **Discusión crítica y limitaciones** — proxy de oclusión, subconjunto CCT20 vs dataset completo, una sola semilla (si no hubo tiempo para varias), sesgo geográfico.
7. **Uso responsable** (½ pág.).

### Referencias sugeridas
- Beery, Van Horn, Perona (2018). *Recognition in Terra Incognita*. ECCV. — dataset y splits cis/trans.
- Norouzzadeh et al. (2018). *Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning*. PNAS.
- Beery, Morris, Yang (2019). *Efficient Pipeline for Camera Trap Image Review* (MegaDetector). arXiv:1907.06772.
- Redmon et al. (2016). *You Only Look Once*. CVPR. / Jocher et al. (2023). *Ultralytics YOLOv8* (software).
- Selvaraju et al. (2017). *Grad-CAM*. ICCV.
- Gupta, Dollár, Girshick (2019). *LVIS*. CVPR. — Repeat Factor Sampling.
- Lin et al. (2017). *Focal Loss for Dense Object Detection*. ICCV.
- Geirhos et al. (2020). *Shortcut learning in deep neural networks*. Nature Machine Intelligence.
- Schneider et al. (2020). *Three critical factors affecting automated image species recognition performance for camera traps*. Ecology and Evolution.

(Verifica cada referencia en Google Scholar antes de citarla.)

---

## 11. Cómo reutilizar esto en proyectos parecidos

El esqueleto sirve para casi cualquier proyecto de detección aplicada:

1. **Convertidor** del formato del dataset a YOLO + CSV de metadatos (dónde, cuándo, quién).
2. **Configuración única** y entrenamiento sin lógica propia.
3. **Evaluación estratificada**: global → por clase → por subgrupo (tamaño, condición, dominio).
4. **Prueba de atajos**: ¿qué pasa si quito el objeto? ¿dónde mira el modelo?
5. **Capa de negocio basada en reglas** que convierte detecciones en la unidad que usa el usuario final.

Cambia el dataset (p. ej. otro de LILA BC con especies sudamericanas, o defectos industriales, o tráfico) y casi todo se mantiene.

### Preguntas para autoevaluarte
1. ¿Por qué no se usa `trans_val` para early stopping?
2. ¿Qué pasa con el mAP si quitas todas las vacías del test? ¿Y con la utilidad real?
3. Si el Grad-CAM cae sobre el animal pero el modelo falla en trans, ¿qué otras causas hay? (iluminación IR, pose, especies ausentes del train en esas cámaras)
4. ¿Por qué MaxN y no la suma de individuos por foto?
5. ¿Qué umbral de confianza usarías para una especie en peligro y por qué?
