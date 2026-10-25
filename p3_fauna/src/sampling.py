"""Lista de entrenamiento por experimento: imágenes vacías como fondo y Repeat Factor Sampling."""
from __future__ import annotations

import math
import random

import pandas as pd


def repeat_factors(train_meta: pd.DataFrame, threshold: float) -> dict[str, float]:
    """r_c = max(1, sqrt(t / f_c)), con f_c la fracción de imágenes que contienen la especie c."""
    n = len(train_meta)
    species = train_meta.loc[train_meta.n_boxes > 0, "species"].str.split(";")
    freq = species.explode().value_counts()
    return {c: max(1.0, math.sqrt(threshold / (f / n))) for c, f in freq.items()}


def build_train_list(meta: pd.DataFrame, sampling: dict, seed: int) -> tuple[list[str], dict]:
    """
    Devuelve la lista de rutas (con repeticiones) y un resumen.

    meta: tabla images_meta.csv filtrada a la partición de entrenamiento.
    """
    rng = random.Random(seed)
    animal = meta[meta.n_boxes > 0]
    empty = meta[meta.n_boxes == 0]
    n_empty = min(len(empty), int(round(len(animal) * float(sampling["empty_ratio_train"]))))
    empty_sel = empty.sample(n=n_empty, random_state=seed) if n_empty else empty.iloc[:0]
    used = pd.concat([animal, empty_sel])

    factors: dict[str, float] = {}
    if sampling.get("repeat_factor_sampling", False):
        factors = repeat_factors(used, float(sampling["rfs_threshold"]))
    entries: list[str] = []
    for path, sp in zip(used.image, used.species):
        r = max((factors.get(c, 1.0) for c in sp.split(";")), default=1.0) if sp != "empty" else 1.0
        k = int(r) + (1 if rng.random() < r - int(r) else 0)
        entries += [path] * max(k, 1)
    rng.shuffle(entries)
    summary = {"imagenes_con_animal": len(animal), "imagenes_vacias": n_empty,
               "entradas_lista": len(entries), "factores_rfs": factors}
    return entries, summary
