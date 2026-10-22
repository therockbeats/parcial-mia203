"""
PASO 5. Selección de la serie por estudio y del subconjunto estratificado (reducción de escala).

1. Para cada estudio elige UNA serie según selection.series_priority del config
   (por defecto: sagital DP/T2 con supresión grasa; si no hay, sagital DP/T2; si no, cualquier
   sagital; si no, coronal DP/T2 con supresión grasa).
2. Los estudios con etiqueta dorada forman el conjunto "gold" (evaluación limpia).
3. Del resto, arma un subconjunto de n_studies estudios (recomendado 300-600) garantizando
   al menos min_pos_per_label positivos por hallazgo: primero se toman positivos de los
   hallazgos más raros, luego se completa al azar.
4. Separa una parte para validación interna (early stopping y elección de umbrales),
   cuidando que cada hallazgo tenga positivos en validación.

Salidas:
    <interim_dir>/split.csv   columnas: study_id, series_id, split(train/val/gold), plano,
                              y un código por hallazgo (1 pos, 0 neg, 2 incierto, -1 no mencionado)
    p1_rodilla/reports/tablas/prevalencia_por_split.csv

Uso:  python -m p1_rodilla.scripts.s05_seleccionar_estudios
"""
from __future__ import annotations

from pathlib import Path

import argparse

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p1_rodilla.src.data_io import load_gold
from p1_rodilla.src.labels_rules import NEG, POS


def pick_series(meta: pd.DataFrame, priority: list[dict], min_slices: int) -> pd.DataFrame:
    """
    Devuelve una fila por estudio con la serie preferida.

    Cada regla de `priority` es un diccionario como
        {plane: "sagittal", fat_sat: true, fluid_sensitive: true}
    y se prueban en orden: la primera regla que encuentre alguna serie gana.
    Un valor null indica que el atributo no se usa como filtro.
    """
    m = meta.copy()
    if "error" in m.columns:
        m = m[m["error"].isna()]
    m = m[m["n_slices"] >= min_slices]
    if "fluid_sensitive" not in m.columns:
        m["fluid_sensitive"] = m["weighting"].isin(["PD", "T2"])
    chosen = []
    for sid, g in m.groupby("study_id"):              # g = todas las series de un estudio
        for rank, rule in enumerate(priority):
            cand = g[g["plane"] == rule["plane"]]
            for key in ("fat_sat", "fluid_sensitive"):
                if rule.get(key) is not None:
                    cand = cand[cand[key] == rule[key]]
            if len(cand):
                # Si hay varias: primero sensibles a líquido, luego la de más cortes
                best = cand.sort_values(["fluid_sensitive", "n_slices"], ascending=False).iloc[0]
                chosen.append({"study_id": sid, "series_id": best["series_id"], "plane": best["plane"],
                               "fat_sat": bool(best["fat_sat"]), "fluid_sensitive": bool(best["fluid_sensitive"]),
                               "weighting": best.get("weighting", ""), "n_slices": int(best["n_slices"]),
                               "series_rule": rank, "series_description": best.get("series_description", "")})
                break
    return pd.DataFrame(chosen)


def load_weak(cfg: dict, interim) -> pd.DataFrame:
    names, source = cfg["labels"]["names"], cfg["labels"].get("source", "rules")
    rules = pd.read_parquet(interim / "labels_rules.parquet")[["study_id"] + names]
    if source == "rules":
        return rules
    llm_path = interim / "labels_llm.parquet"
    if not llm_path.exists():
        raise FileNotFoundError("labels.source pide LLM pero no existe labels_llm.parquet (PASO 4)")
    llm = pd.read_parquet(llm_path)[["study_id"] + names]
    if source == "llm":
        return llm
    # "combined": LLM donde exista, reglas en el resto
    comb = rules.set_index("study_id")
    comb.update(llm.set_index("study_id"))
    return comb.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s05")
    sel, names = cfg["selection"], cfg["labels"]["names"]
    rng = np.random.default_rng(cfg.get("seed", 42))
    interim = ensure_dir(cfg["data"]["interim_dir"])

    meta = pd.read_parquet(interim / "series_meta.parquet")
    meta["study_id"], meta["series_id"] = meta["study_id"].astype(str), meta["series_id"].astype(str)
    chosen = pick_series(meta, sel["series_priority"], sel["min_slices"])
    log.info(f"Estudios con serie válida: {len(chosen):,}")
    log.info("Regla de serie usada (0 = primera preferencia):\n" + chosen["series_rule"].value_counts().to_string())

    # ---------- Conjunto dorado ----------
    gold = load_gold(cfg)
    gold_sel = chosen[chosen.study_id.isin(gold.study_id)].merge(gold, on="study_id")
    for n in names:  # dorado: 1 -> POS, 0 -> NEG
        gold_sel[n] = np.where(gold_sel[n] == 1, POS, NEG)
    gold_sel["split"] = "gold"
    log.info(f"Estudios dorados con serie válida: {len(gold_sel)} de {len(gold)}")

    # ---------- Subconjunto con etiquetas débiles ----------
    weak = load_weak(cfg, interim)
    pool = chosen[~chosen.study_id.isin(gold.study_id)].merge(weak, on="study_id")
    log.info(f"Estudios candidatos con etiqueta débil: {len(pool):,}")
    target, min_pos = int(sel["n_studies"]), int(sel["min_pos_per_label"])
    picked: set[str] = set()
    prevalence = {n: int((pool[n] == POS).sum()) for n in names}
    for n in sorted(names, key=lambda k: prevalence[k]):           # del más raro al más común
        pos_ids = pool.loc[pool[n] == POS, "study_id"].tolist()
        rng.shuffle(pos_ids)
        have = sum(1 for s in picked if s in set(pos_ids))
        for s in pos_ids:
            if have >= min_pos or len(picked) >= target:
                break
            if s not in picked:
                picked.add(s)
                have += 1
    rest = [s for s in pool.study_id if s not in picked]
    rng.shuffle(rest)
    picked |= set(rest[: max(0, target - len(picked))])
    subset = pool[pool.study_id.isin(picked)].copy()

    # ---------- Train / val ----------
    frac = float(sel["val_fraction"])
    val_ids: set[str] = set()
    for n in sorted(names, key=lambda k: prevalence[k]):
        pos_ids = [s for s in subset.loc[subset[n] == POS, "study_id"] if s not in val_ids]
        need = int(np.ceil(frac * (subset[n] == POS).sum())) - sum(1 for s in val_ids if s in set(subset.loc[subset[n] == POS, "study_id"]))
        rng.shuffle(pos_ids)
        val_ids |= set(pos_ids[: max(0, need)])
    others = [s for s in subset.study_id if s not in val_ids]
    rng.shuffle(others)
    n_val = int(round(frac * len(subset)))
    val_ids |= set(others[: max(0, n_val - len(val_ids))])
    subset["split"] = np.where(subset.study_id.isin(val_ids), "val", "train")

    cols = ["study_id", "series_id", "split", "plane", "fat_sat", "fluid_sensitive", "weighting", "n_slices",
            "series_description"] + names
    split = pd.concat([subset[cols], gold_sel[cols]], ignore_index=True)
    out = interim / "split.csv"
    split.to_csv(out, index=False)
    log.info(f"Guardado {out}:\n" + split["split"].value_counts().to_string())

    prev = split.groupby("split")[names].apply(lambda d: (d == POS).sum()).T
    prev["descripcion"] = "positivos por hallazgo"
    p = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas") / "prevalencia_por_split.csv"
    prev.to_csv(p)
    log.info("\nPositivos por hallazgo y split:\n" + prev.drop(columns="descripcion").to_string())


if __name__ == "__main__":
    main()
