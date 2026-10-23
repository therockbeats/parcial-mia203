# Inventario de datos: Sen1Floods11 v1.1

Carpeta analizada: `D:\MIA203\data\p2_sen1floods11\raw\v1.1`

## Archivos por carpeta

| Carpeta | Archivos .tif |
|---|---:|
| `data/flood_events/HandLabeled/JRCWaterHand` | 446 |
| `data/flood_events/HandLabeled/LabelHand` | 446 |
| `data/flood_events/HandLabeled/S1Hand` | 446 |
| `data/flood_events/HandLabeled/S1OtsuLabelHand` | 446 |
| `data/flood_events/HandLabeled/S2Hand` | 446 |
| `data/flood_events/WeaklyLabeled/S1OtsuLabelWeak` | 4,384 |
| `data/flood_events/WeaklyLabeled/S1Weak` | 4,384 |
| `data/flood_events/WeaklyLabeled/S2IndexLabelWeak` | 4,384 |

## Particiones oficiales (chips dorados)

| Partición | Chips | Sin S1Hand | Sin LabelHand |
|---|---:|---:|---:|
| train | 252 | 0 | 0 |
| val | 89 | 0 | 0 |
| test | 90 | 0 | 0 |
| bolivia | 15 | 0 | 0 |

## Chips por evento

| Evento | bolivia | debil | test | train | val | total |
|---|---:|---:|---:|---:|---:|---:|
| Bolivia | 15 | 224 | 0 | 0 | 0 | 239 |
| Colombia | 0 | 534 | 0 | 0 | 0 | 534 |
| Ghana | 0 | 181 | 11 | 31 | 11 | 234 |
| India | 0 | 467 | 14 | 40 | 14 | 535 |
| Mekong | 0 | 1353 | 6 | 18 | 6 | 1383 |
| Nigeria | 0 | 109 | 4 | 10 | 4 | 127 |
| Pakistan | 0 | 249 | 6 | 16 | 6 | 277 |
| Paraguay | 0 | 316 | 14 | 39 | 14 | 383 |
| Somalia | 0 | 129 | 6 | 15 | 5 | 155 |
| Spain | 0 | 146 | 6 | 18 | 6 | 176 |
| Sri-Lanka | 0 | 190 | 9 | 24 | 9 | 232 |
| USA | 0 | 486 | 14 | 41 | 14 | 555 |

## Muestra de cabeceras

| Archivo | Bandas | Tipo | Tamaño | CRS | Píxel (grados) | Nodata |
|---|---:|---|---|---|---|---|
| `Bolivia_103757_S1Hand.tif` | 2 | float32 | 512x512 | EPSG:4326 | 8.9832e-05 | nan |
| `Bolivia_103757_LabelHand.tif` | 1 | int16 | 512x512 | EPSG:4326 | 8.9832e-05 | None |
| `Bolivia_103757_S1OtsuLabelHand.tif` | 1 | int16 | 512x512 | EPSG:4326 | 8.9832e-05 | -1.0 |
| `Bolivia_103757_JRCWaterHand.tif` | 1 | uint8 | 512x512 | EPSG:4326 | 8.9832e-05 | None |
| `Bolivia_103757_S2Hand.tif` | 13 | int16 | 512x512 | EPSG:4326 | 8.9832e-05 | 0.0 |
| `Bolivia_1009032_S1Weak.tif` | 2 | float32 | 512x512 | EPSG:4326 | 8.9832e-05 | nan |
| `Bolivia_1009032_S1OtsuLabelWeak.tif` | 1 | int16 | 512x512 | EPSG:4326 | 8.9832e-05 | -1.0 |
| `Bolivia_1009032_S2IndexLabelWeak.tif` | 1 | int16 | 512x512 | EPSG:4326 | 8.9832e-05 | None |
