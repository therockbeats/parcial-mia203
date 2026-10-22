# Inventario de datos: RSNA Knee Abnormality Detection

Carpeta analizada: `D:\MIA203\data\p1_rsna_knee\raw`  
Archivos totales: **819,640**

## Contenido del primer nivel

- `sample_submission.csv` (archivo)
- `test.csv` (archivo)
- `test_series` (carpeta)
- `test_series.csv` (archivo)
- `train.csv` (archivo)
- `train_series` (carpeta)
- `train_series.csv` (archivo)

## Archivos por extensión

| Extensión | Cantidad | GB |
|---|---:|---:|
| .dcm | 819,635 | 530.63 |
| .csv | 5 | 0.01 |

## Estructura de carpetas (por profundidad)

| Nivel | Carpetas | Archivos | Ejemplos de archivos |
|---:|---:|---:|---|
| 1 | 2 | 5 | `sample_submission.csv`<br>`test.csv`<br>`test_series.csv` |
| 2 | 4,410 | 0 |  |
| 3 | 24,386 | 0 |  |
| 4 | 0 | 819,635 | `train_series\1.2.826.0.1.3680043.8.498.99957611207313098696146908027229590966\1.2.826.0.1.3680043.8.498.95722278506316542017897506192254108185\1.2.826.0.1.3680043.8.498.10885448200590891750539833800174501870.dcm`<br>`train_series\1.2.826.0.1.3680043.8.498.99957611207313098696146908027229590966\1.2.826.0.1.3680043.8.498.95722278506316542017897506192254108185\1.2.826.0.1.3680043.8.498.10983321128787543962293516472740643471.dcm`<br>`train_series\1.2.826.0.1.3680043.8.498.99957611207313098696146908027229590966\1.2.826.0.1.3680043.8.498.95722278506316542017897506192254108185\1.2.826.0.1.3680043.8.498.11472540896589347626621593998333988493.dcm` |

## Tablas (CSV / Parquet / JSON)

### `sample_submission.csv`  (3 filas)

| Columna | Tipo | No nulos | Únicos | Rango | Ejemplos |
|---|---|---:|---:|---|---|
| `StudyInstanceUID` | str | 3 | 3 |  | 1.2.826.0.1.3680043.8.498.1004703505754442731801857912163527… · 1.2.826.0.1.3680043.8.498.1006286178314531262933225097745699… · 1.2.826.0.1.3680043.8.498.1006751470707257228026348154849759… |
| `ACL` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |
| `MCL` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |
| `Medial Meniscus` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |
| `Lateral Meniscus` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |
| `Medial OA` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |
| `Lateral OA` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |
| `PF OA` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |
| `Effusion` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |
| `Synovitis` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |
| `Baker's` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |
| `Contusion` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |
| `Fracture` | float64 | 3 | 1 | 0.5 a 0.5 | 0.5 |

### `test.csv`  (3 filas)

| Columna | Tipo | No nulos | Únicos | Rango | Ejemplos |
|---|---|---:|---:|---|---|
| `StudyInstanceUID` | str | 3 | 3 |  | 1.2.826.0.1.3680043.8.498.1004703505754442731801857912163527… · 1.2.826.0.1.3680043.8.498.1006286178314531262933225097745699… · 1.2.826.0.1.3680043.8.498.1006751470707257228026348154849759… |

### `test_series.csv`  (15 filas)

| Columna | Tipo | No nulos | Únicos | Rango | Ejemplos |
|---|---|---:|---:|---|---|
| `StudyInstanceUID` | str | 15 | 3 |  | 1.2.826.0.1.3680043.8.498.1004703505754442731801857912163527… · 1.2.826.0.1.3680043.8.498.1006286178314531262933225097745699… · 1.2.826.0.1.3680043.8.498.1006751470707257228026348154849759… |
| `SeriesInstanceUID` | str | 15 | 15 |  | 1.2.826.0.1.3680043.8.498.1158065644225911125567556260515590… · 1.2.826.0.1.3680043.8.498.1781150261403063166451762251890664… · 1.2.826.0.1.3680043.8.498.3056539559504594240408102206248975… · 1.2.826.0.1.3680043.8.498.3285649454181684580594785030465366… |
| `Fluid_Sensitive` | int64 | 15 | 2 | 0 a 1 | 0 · 1 |
| `Fat_Suppression` | int64 | 15 | 2 | 0 a 1 | 0 · 1 |
| `Anatomical_Plane` | str | 15 | 3 |  | Axial · Sagittal · Coronal |

### `train.csv`  (4,407 filas)

| Columna | Tipo | No nulos | Únicos | Rango | Ejemplos |
|---|---|---:|---:|---|---|
| `StudyInstanceUID` | str | 4,407 | 4,407 |  | 1.2.826.0.1.3680043.8.498.1000487322909905386909332429219581… · 1.2.826.0.1.3680043.8.498.1000494592747265602719979207565239… · 1.2.826.0.1.3680043.8.498.1000927869260663157354006290990913… · 1.2.826.0.1.3680043.8.498.1000963920317075027417470743435662… |
| `Report` | str | 4,407 | 4,276 |  | Técnica: RMN de la rodilla. Resultados: Rotura de menisco in… · [DATE]: * MR Knie Rechts 15ch AA Klinische Inlichtingen: [DA… · Hallazgos: No hay alteraciones en significativas de la médul… ·  In the medial compartment, the meniscus is not torn and the… |
| `ACL` | float64 | 58 | 2 | 0 a 1 | 0.0 · 1.0 |
| `MCL` | float64 | 58 | 2 | 0 a 1 | 0.0 · 1.0 |
| `Medial Meniscus` | float64 | 58 | 2 | 0 a 1 | 0.0 · 1.0 |
| `Lateral Meniscus` | float64 | 58 | 2 | 0 a 1 | 0.0 · 1.0 |
| `Medial OA` | float64 | 58 | 2 | 0 a 1 | 0.0 · 1.0 |
| `Lateral OA` | float64 | 58 | 2 | 0 a 1 | 0.0 · 1.0 |
| `PF OA` | float64 | 58 | 2 | 0 a 1 | 1.0 · 0.0 |
| `Effusion` | float64 | 58 | 2 | 0 a 1 | 1.0 · 0.0 |
| `Synovitis` | float64 | 58 | 2 | 0 a 1 | 0.0 · 1.0 |
| `Baker's` | float64 | 58 | 2 | 0 a 1 | 0.0 · 1.0 |
| `Contusion` | float64 | 58 | 2 | 0 a 1 | 0.0 · 1.0 |
| `Fracture` | float64 | 58 | 2 | 0 a 1 | 0.0 · 1.0 |

### `train_series.csv`  (24,371 filas)

| Columna | Tipo | No nulos | Únicos | Rango | Ejemplos |
|---|---|---:|---:|---|---|
| `StudyInstanceUID` | str | 24,371 | 4,407 |  | 1.2.826.0.1.3680043.8.498.1000487322909905386909332429219581… · 1.2.826.0.1.3680043.8.498.1000494592747265602719979207565239… · 1.2.826.0.1.3680043.8.498.1000927869260663157354006290990913… · 1.2.826.0.1.3680043.8.498.1000963920317075027417470743435662… |
| `SeriesInstanceUID` | str | 24,371 | 24,371 |  | 1.2.826.0.1.3680043.8.498.1234311019503621348345409171541233… · 1.2.826.0.1.3680043.8.498.1382122974499722064157529192742654… · 1.2.826.0.1.3680043.8.498.2308483653672259527582869029316873… · 1.2.826.0.1.3680043.8.498.4073420610245872309615468714739047… |
| `Fluid_Sensitive` | int64 | 24,371 | 2 | 0 a 1 | 1 · 0 |
| `Fat_Suppression` | int64 | 24,371 | 2 | 0 a 1 | 1 · 0 |
| `Anatomical_Plane` | str | 24,371 | 3 |  | Sagittal · Axial · Coronal |

## Muestra de cabeceras DICOM

| carpeta | n_archivos | SeriesDescription | ImageOrientationPatient | ScanOptions | RepetitionTime | EchoTime | SliceThickness | PixelSpacing | Rows | Columns | Manufacturer | MagneticFieldStrength | TransferSyntax | prueba_pixeles | error_cabecera |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train_series\1.2.826.0.1.3680043.8.498.103514609113871068791… | 22 | STIR | [0.97632777690887, 0.20293751358985, 0.07483636587858, -0.21… | OTHER | 2986.72216796875 | 30 | 3.5 | [0.33203125, 0.33203125] | 512 | 512 | Philips Medical Systems | 1.5 | Explicit VR Little Endian | OK shape=(512, 512) dtype=uint16 min=0 max=666 |  |
| train_series\1.2.826.0.1.3680043.8.498.101708986158676730286… | 40 | RT.Tra_T1W_TSE | [0.98052859306335, -0.1944152116775, -0.0276844967156, 0.195… | OTHER | 545.468994140625 | 11 | 3 | [0.22727273404598, 0.22727273404598] | 704 | 704 | Philips | 3 | Explicit VR Little Endian | OK shape=(704, 704) dtype=uint16 min=0 max=1376 |  |
| train_series\1.2.826.0.1.3680043.8.498.102738613267537074366… | 32 | pd_tse_fs_tra_320 | [0.98952248560119, 0.10698160659187, 0.09695455812215, -0.11… | FS | 2920 | 20 | 3.5 | [0.234375, 0.234375] | 640 | 640 | SIEMENS | 3 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.102394181828192708608… | 21 | AXI DER | [0.98155111074447, -0.1474914848804, -0.1216701865196, 0.161… | OTHER | 3152.43286132812 | 28 | 4 | [0.29469498991966, 0.29469498991966] | 528 | 528 | Philips | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.103123507141647390811… | 27 | t2_pd_fs_tra L | [0.94832366006779, 0.31730464187534, 0, -0.3173046418753, 0.… | FS | 3480 | 32 | 3 | [0.625, 0.625] | 256 | 256 | SIEMENS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.102548283650339847089… | 22 | SAG T2 | [-0.104535, 0.994521, 0, -0, -0, -1] | ['FAST_GEMS', 'FC_FREQ_AX_GEMS', 'FC', 'SAT_GEMS', 'EDR_GEMS… | 4257 | 83.84 | 4 | [0.293, 0.293] | 512 | 512 | GE MEDICAL SYSTEMS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.101627203742911801969… | 20 | COR T1 | [0.99001848697662, -0.0319242589175, -0.1372744441032, -0.13… | OTHER | 529.737609863281 | 17 | 4.324 | [0.2734375, 0.2734375] | 640 | 640 | Philips | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.103142400786858041107… | 34 | RT. PD_TSE_SAG | [0.0801163, 0.996786, -3.17885E-8, -0.0585359, 0.00470477, -… | ['SP', 'PER'] | 2020.0 | 35.0 | 3.0 | [0.507812, 0.507812] | 384 | 384 | Siemens Healthineers | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.102093870476934482824… | 24 | pd_tse_fs_tra_np | [0.99205436334426, -0.0857418226579, -0.0920678011953, 0.086… | FS | 3840 | 14 | 4 | [0.390625, 0.390625] | 384 | 384 | SIEMENS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.103747917348803106435… | 30 | pd_tse_fs_cor_320_p4_s2_DRB | [0.973613, 0.218935, 0.0643863, 0.0352495, 0.134477, -0.9902… | ['FS', 'PER'] | 2300 | 32 | 3 | [0.2734, 0.2734] | 640 | 540 | Siemens Healthineers | 3 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.102401664400440875541… | 36 | pd_tse_fs_cor_384_p2_s2 | [0.998493, 0.0446472, 0.0319228, 0.0303198, 0.0361333, -0.99… | ['FS', 'PER'] | 2850 | 33 | 3 | [0.390625, 0.390625] | 384 | 384 | Siemens Healthineers | 3 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.103123507141647390811… | 27 | t1_tse_cor_320 L | [0.94763316284536, 0.31936090661775, -1.9684297e-008, -0.035… |  | 520 | 11 | 3 | [0.5, 0.5] | 320 | 320 | SIEMENS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.103872274721106434143… | 35 | pd_tse_COR_384 | [0.98817, -0.153152, -0.00802859, -0.00910133, -0.00630468, … | PER | 3030 | 44 | 3 | [0.416667, 0.416667] | 384 | 384 | Siemens Healthineers | 3 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.101575496629170622739… | 24 | COR PD R | [0.98557543754577, -0.1585438251495, -0.0592024251818, -0.05… | OTHER | 2200.0 | 40.0 | 4.5 | [0.30357143282890, 0.30357143282890] | 560 | 560 | Philips Medical Systems | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.100049459274726560271… | 30 | pd_tse_fs_cor_320_p4_s2_DRB | [0.999541, -0.000289172, -0.0302993, -0.0300255, 0.124995, -… | ['FS', 'PER'] | 2300 | 32 | 3 | [0.25, 0.25] | 640 | 540 | Siemens Healthineers | 3 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.103872274721106434143… | 36 | pd_tse_fs_sag_384 | [0.159834, 0.987144, -5.2514e-09, -4.22391e-10, -5.2514e-09,… | ['FS', 'PER'] | 3720 | 44 | 3 | [0.416667, 0.416667] | 384 | 384 | Siemens Healthineers | 3 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.100906318110169398193… | 35 | DummySeriesDesc! | [-0.3696313500404, 0.92916375398635, -0.0052369376644, 0.002… | FS | 3143.98803710937 | 35 | 3 | [0.31124749779701, 0.31124749779701] | 512 | 512 | Philips Medical Systems | 3 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.101811196647283350209… | 40 | Ax T2 FSE FS LEFT | [0.99965, -0.00000, 0.02658, -0.00000, 1.00000, 0.00000] | ['FAST_GEMS', 'FC_SLICE_AX_GEMS', 'FC', 'SAT_GEMS', 'TRF_GEM… | 5884 | 65.24 | 3 | [0.5859, 0.5859] | 256 | 256 | GE MEDICAL SYSTEMS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.101581786689623798579… | 32 | Axial PD FatSat | [1, 0, 0, 0, 1, 0] | ['SP', 'FS', 'FC'] | 2124 | 36 | 4 | [0.33482142857143, 0.33482142857143] | 448 | 448 | FUJIFILM Healthcare Corporation | 1.16 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.100592006271753204819… | 16 | DP SPAIR_CS | [0.99842762947082, -0.0332781411707, -0.0451091192662, -0.03… | FS | 1800 | 18 | 4 | [0.30745044350624, 0.30745044350624] | 560 | 560 | Philips Healthcare | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.101839398418908950370… | 21 | DummySeriesDesc! | [-0.01296, 0.99969, 0.02148, 0.01894, 0.02173, -0.99958] | ['FAST_GEMS', 'SAT_GEMS', 'NPW', 'NP', 'EDR_GEMS', 'TRF_GEMS… | 4452 | 62.448 | 3.5 | [0.1465, 0.1465] | 1024 | 1024 | GE MEDICAL SYSTEMS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.100295200339579680440… | 16 | LT_t2_tse_fs_cor_obl_ACL | [0.99950975079015, 0.03130907337194, -3.1225556e-008, 0.0159… | FS | 4400 | 70 | 2 | [0.234375, 0.234375] | 640 | 640 | SIEMENS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.102996186116261909271… | 16 | COR T1 | [0.99611622095108, 0.05230011045932, -0.0708316937088, -0.07… | OTHER | 605.414428710937 | 17 | 3.5 | [0.32196968793869, 0.32196968793869] | 528 | 528 | Philips Medical Systems | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.101975650049929377228… | 26 | SAG DP | [0.22500629723072, 0.97434759140014, -0.0043542566709, 0.011… | OTHER | 2844.56909179687 | 10 | 3.5 | [0.265625, 0.265625] | 640 | 640 | Philips | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.100969196892724243881… | 32 | pd_tse_fs_cor_320 | [0.99790275609421, -0.0647308997279, -6.127376e-009, 0.00116… | FS | 4210 | 32 | 3 | [0.5625, 0.5625] | 320 | 320 | SIEMENS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.100969196892724243881… | 30 | pd_tse_fs_sag_320 | [-0.0714970736579, 0.99744080950117, 2.861513e-009, 0.005219… | FS | 3950 | 32 | 3 | [0.5625, 0.5625] | 320 | 320 | SIEMENS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.102292235464124419248… | 28 | RT.COR PDW_TSE_SPAIR | [0.97217488288879, -0.1398718059062, -0.1879146248102, -0.18… | FS | 3178.12182617187 | 30 | 3 | [0.33203125, 0.33203125] | 512 | 512 | Philips | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.101237517397675860257… | 24 | L- CORPD FSE FS | [0.9950996996970, 0.0836287416251, 0.0527524524277, 0.051298… | ['FAST_GEMS', 'FC_FREQ_AX_GEMS', 'FC', 'SAT_GEMS', 'EDR_GEMS… | 1500 | 35.392 | 4 | [0.6641, 0.6641] | 256 | 256 | GE MEDICAL SYSTEMS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.101240636074039423301… | 33 | T1W_TSE | [0.05507993698120, 0.99076759815216, 0.12387818098068, -0.03… | OTHER | 631.271179199218 | 17 | 3 | [0.30388787388801, 0.30388787388801] | 528 | 528 | Philips Medical Systems | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.101811196647283350209… | 25 | Sag PD FSE | [0.12279, 0.99226, 0.01860, 0.01772, 0.01654, -0.99971] | ['FAST_GEMS', 'FC_FREQ_AX_GEMS', 'FC', 'NPW', 'NP', 'TRF_GEM… | 2975 | 17.664 | 4 | [0.293, 0.293] | 512 | 512 | GE MEDICAL SYSTEMS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.100212885571718667946… | 20 | L- Sag PD FSE FS | [0.0932094238332, 0.9950128826276, 0.0355157248799, 0.087583… | ['FAST_GEMS', 'FC_FREQ_AX_GEMS', 'FC', 'SAT_GEMS', 'NPW', 'N… | 1500 | 33.984 | 3 | [0.332, 0.332] | 512 | 512 | GE MEDICAL SYSTEMS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.103520119385749887278… | 50 | 3D_WATSc | [0.99824106693267, -0.0032142577692, -0.0591978952288, 0.005… | OTHER | 20 | 5.153 | 3 | [0.29296875, 0.29296875] | 512 | 512 | Philips | 3 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.103514609113871068791… | 16 | PDW_SPIR | [0.99548900127410, 0.05913405865430, 0.07419408857822, 0.064… | FS | 1545.69201660156 | 30 | 3.5 | [0.32196968793869, 0.32196968793869] | 528 | 528 | Philips Medical Systems | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.101215588540842711542… | 31 | R- CORPD FSE FS | [0.9967022482972, -0.0422670012661, 0.0692685270761, 0.07358… | ['FAST_GEMS', 'FC_FREQ_AX_GEMS', 'FC', 'SAT_GEMS', 'NPW', 'N… | 2235 | 31.944 | 4 | [0.332, 0.332] | 512 | 512 | GE MEDICAL SYSTEMS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.102564414551078604742… | 22 | L- CORPD FSE FS | [0.9999762415337, 0.0058600034630, 0.0036299762595, 0.003516… | ['FAST_GEMS', 'FC_FREQ_AX_GEMS', 'FC', 'SAT_GEMS', 'EDR_GEMS… | 1513 | 31.296 | 4 | [0.332, 0.332] | 512 | 512 | GE MEDICAL SYSTEMS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.102996186116261909271… | 18 | PDW_TSE_SPAIR_Tra | [0.99854433536529, -0.0067568677477, -0.0535125322639, 0.016… | FS | 2500.00024414062 | 30 | 3 | [0.3125, 0.3125] | 320 | 320 | Philips Medical Systems | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.102433004296079356146… | 38 | pd_tse_tra_d | [0.997008, 4.897e-12, -0.077296, -4.882e-12, 1, 3.78e-13] | PER | 3000 | 46 | 3 | [0.1562, 0.1562] | 960 | 960 | Siemens Healthineers | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.101240636074039423301… | 33 | PDW_SPAIR | [0.99771767854690, -0.0510839968919, -0.0441577024757, -0.03… | FS | 4392.5654296875 | 30 | 3 | [0.3125, 0.3125] | 512 | 512 | Philips Medical Systems | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.103061591133248115387… | 30 | ax STIR_sm_AIRS | [0.960607, -0.253804, -0.113214, 0.263867, 0.960814, 0.08491… | ['SAT_GEMS', 'NPW', 'SEQ_GEMS', 'TRF_GEMS', 'SP'] | 4167 | 52.48 | 3 | [0.31248, 0.31248] | 640 | 640 | GE MEDICAL SYSTEMS | 1.5 | Explicit VR Little Endian |  |  |
| train_series\1.2.826.0.1.3680043.8.498.103228530841229204083… | 30 | DummySeriesDesc! | [0.99212950468063, -0.1252152919769, 0.00044675890239, 0.125… | FS | 2204.02807617187 | 30 | 4 | [0.3125, 0.3125] | 512 | 512 | Philips Medical Systems | 3 | Explicit VR Little Endian |  |  |

## Siguiente paso

Con este reporte, corrige la sección `schema:` de `p1_rodilla/config.yaml` (nombre de columnas, archivo de reportes, carpeta de imágenes) y sube los cambios.