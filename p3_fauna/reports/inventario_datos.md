# Inventario de datos: Caltech Camera Traps (CCT-20)

Imágenes: `C:\Users\USUARIO\MIA203\data\p3_cct20\raw\eccv_18_all_images_sm`
Anotaciones: `C:\Users\USUARIO\MIA203\data\p3_cct20\raw\eccv_18_annotation_files`

Archivos por partición: `train`: train_annotations.json, train_vacias_annotations.json; `cis_val`: cis_val_annotations.json; `trans_val`: trans_val_annotations.json; `cis_test`: cis_test_annotations.json; `trans_test`: trans_test_annotations.json

## Imágenes por partición

| particion | imagenes | con_caja | vacias | ambiguas | pct_vacias | cajas | camaras | en_disco |
|---|---|---|---|---|---|---|---|---|
| train | 16553 | 12099 | 3000 | 1454 | 18.10 | 12617 | 116 | 16553 |
| cis_val | 3484 | 1665 | 995 | 824 | 28.60 | 1763 | 10 | 3484 |
| trans_val | 1725 | 1533 | 61 | 131 | 3.50 | 1673 | 1 | 1725 |
| cis_test | 15827 | 12696 | 1180 | 1951 | 7.50 | 13264 | 10 | 15827 |
| trans_test | 23275 | 18033 | 1778 | 3464 | 7.60 | 18786 | 9 | 23275 |

Ambiguas: imágenes con etiqueta de animal sin caja; se excluyen de la detección. La partición train de CCT-20 no trae vacías; las de `train_vacias_annotations.json` provienen del Caltech Camera Traps completo, de cámaras ajenas a CCT-20 y nunca de cámaras trans (etapa 1b).

## Cámaras (cis = vistas en entrenamiento, trans = no vistas)

- `train`: 116 cámaras, 116 compartidas con entrenamiento
- `cis_val`: 10 cámaras, 10 compartidas con entrenamiento
- `trans_val`: 1 cámaras, 0 compartidas con entrenamiento
- `cis_test`: 10 cámaras, 10 compartidas con entrenamiento
- `trans_test`: 9 cámaras, 0 compartidas con entrenamiento

## Cajas por especie (15 especies)

| especie | train | cis_val | trans_val | cis_test | trans_test | total |
|---|---|---|---|---|---|---|
| opossum | 2514 | 358 | 425 | 4010 | 4663 | 11970 |
| raccoon | 1030 | 126 | 136 | 981 | 4803 | 7076 |
| rabbit | 2278 | 366 | 70 | 1559 | 722 | 4995 |
| coyote | 1371 | 200 | 45 | 1182 | 1902 | 4700 |
| bobcat | 684 | 92 | 793 | 772 | 1979 | 4320 |
| cat | 1170 | 169 | 70 | 1528 | 1366 | 4303 |
| dog | 769 | 106 | 80 | 725 | 961 | 2641 |
| car | 668 | 41 | 0 | 791 | 1113 | 2613 |
| squirrel | 1037 | 150 | 0 | 510 | 782 | 2479 |
| bird | 560 | 82 | 5 | 752 | 180 | 1579 |
| skunk | 214 | 24 | 49 | 162 | 273 | 722 |
| rodent | 264 | 46 | 0 | 146 | 26 | 482 |
| deer | 44 | 2 | 0 | 140 | 0 | 186 |
| badger | 9 | 1 | 0 | 4 | 15 | 29 |
| fox | 5 | 0 | 0 | 2 | 1 | 8 |
