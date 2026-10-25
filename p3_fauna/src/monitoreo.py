"""
Reglas del módulo de monitoreo ecológico.

R1  Fotograma: una especie está presente si hay al menos una detección con conf >= deploy_conf.
R2  Secuencia (ráfaga de la cámara, ``seq_id``): presencia si aparece en algún fotograma;
    individuos = MaxN, el máximo de individuos de esa especie en un solo fotograma.
R3  Evento independiente: secuencias de la misma especie separadas por menos de
    ``independence_minutes`` pertenecen al mismo evento.
R4  Índice de abundancia relativa: RAI = eventos independientes / días-cámara x 100.
R5  Fotograma vacío: sin detecciones tras R1.
"""
from __future__ import annotations

import pandas as pd


def build_report(frames: pd.DataFrame, dets: pd.DataFrame, indep_min: int, period: str) -> dict:
    """
    frames: image, date_captured, seq_id (una fila por fotograma de la cámara).
    dets:   image, species (una fila por detección; species nulo si el fotograma no tiene).
    """
    frames = frames.copy()
    frames["ts"] = pd.to_datetime(frames.date_captured, errors="coerce")
    frames["seq_id"] = frames.seq_id.fillna(frames.image)
    per_frame = (dets.dropna(subset=["species"]).groupby(["image", "species"]).size().rename("n").reset_index()
                 .merge(frames[["image", "ts", "seq_id"]], on="image"))
    seq = (per_frame.groupby(["seq_id", "species"])
           .agg(maxN=("n", "max"), ts=("ts", "min"), fotogramas=("image", "nunique")).reset_index()
           .sort_values(["species", "ts"]))
    gap = seq.groupby("species").ts.diff()
    seq["nuevo_evento"] = gap.isna() | (gap > pd.Timedelta(minutes=indep_min))
    seq["evento_id"] = seq.groupby("species").nuevo_evento.cumsum()
    events = (seq.groupby(["species", "evento_id"])
              .agg(inicio=("ts", "min"), fin=("ts", "max"), maxN=("maxN", "max"),
                   secuencias=("seq_id", "nunique")).reset_index())

    days = max(1.0, (frames.ts.max() - frames.ts.min()) / pd.Timedelta(days=1)) if frames.ts.notna().any() else 1.0
    events["periodo"] = events.inicio.dt.to_period(period).astype(str)
    by_period = (events.groupby(["periodo", "species"])
                 .agg(eventos=("evento_id", "size"), maxN=("maxN", "max"), secuencias=("secuencias", "sum"))
                 .reset_index())
    by_species = (events.groupby("species")
                  .agg(eventos=("evento_id", "size"), maxN=("maxN", "max"), secuencias=("secuencias", "sum"))
                  .reset_index())
    by_species["fotogramas"] = by_species.species.map(per_frame.groupby("species").image.nunique())
    by_species["RAI_100_dias"] = by_species.eventos / days * 100
    events["franja"] = pd.cut(events.inicio.dt.hour, [-1, 5, 11, 17, 23],
                              labels=["madrugada (0-5 h)", "mañana (6-11 h)", "tarde (12-17 h)", "noche (18-23 h)"])
    by_hour = events.pivot_table(index="species", columns="franja", values="evento_id", aggfunc="size",
                                 fill_value=0, observed=False)
    with_fauna = per_frame.image.nunique()
    summary = {"fotogramas": len(frames), "fotogramas_con_fauna": with_fauna,
               "tasa_vacias": 1 - with_fauna / max(len(frames), 1),
               "especies": by_species.species.nunique(), "lista_especies": ", ".join(sorted(by_species.species)),
               "dias_camara": round(days, 1), "desde": str(frames.ts.min()), "hasta": str(frames.ts.max())}
    return {"summary": summary, "by_species": by_species, "by_period": by_period, "by_hour": by_hour, "events": events}


def df_to_md(df: pd.DataFrame) -> str:
    if df.index.name is not None:
        df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def report_markdown(camera: str, pred: dict, ref: dict | None, conf: float) -> tuple[str, pd.DataFrame | None]:
    s = pred["summary"]
    md = [f"# Reporte de actividad · cámara {camera}", "",
          f"Detecciones con confianza >= {conf}; eventos independientes separados por más de la ventana configurada.", "",
          "## Resumen", "",
          f"- Fotogramas: {s['fotogramas']:,} ({s['fotogramas_con_fauna']:,} con fauna; vacíos {s['tasa_vacias']:.1%})",
          f"- Periodo: {s['desde']} a {s['hasta']} ({s['dias_camara']} días-cámara)",
          f"- Especies detectadas ({s['especies']}): {s['lista_especies']}", "",
          "## Actividad por especie", "", df_to_md(pred["by_species"]), "",
          "## Actividad por periodo", "", df_to_md(pred["by_period"]), "",
          "## Eventos por franja horaria", "", df_to_md(pred["by_hour"])]
    cmp = None
    if ref is not None:
        cmp = (pred["by_species"][["species", "eventos", "maxN"]]
               .merge(ref["by_species"][["species", "eventos", "maxN"]], on="species", how="outer",
                      suffixes=("_modelo", "_referencia")).fillna(0).sort_values("species"))
        md += ["", "## Validación con las etiquetas de referencia", "",
               f"- Fotogramas vacíos: modelo {s['tasa_vacias']:.1%}, referencia {ref['summary']['tasa_vacias']:.1%}",
               f"- Especies de referencia: {ref['summary']['lista_especies']}", "", df_to_md(cmp)]
    return "\n".join(md) + "\n", cmp
