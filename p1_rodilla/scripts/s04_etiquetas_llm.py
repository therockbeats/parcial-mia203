"""
PASO 4 (OPCIONAL, recomendado). Etiquetas débiles con un LLM LOCAL (Ollama).

Por qué: las reglas fallan con negaciones complejas ("sin evidencia de rotura del LCA,
aunque hay edema"), sinónimos y los ~12 idiomas. Un modelo de lenguaje multilingüe
entiende el reporte completo. Corre 100 % local (Ollama), sin enviar datos a internet.

Se procesan solo los estudios necesarios (dorados + subconjunto seleccionado en el
PASO 5) para que tome minutos y no horas. Si prefieres etiquetar todo, usa --all.

Requisitos:
    1. Instalar Ollama para Windows: https://ollama.com/download
    2. En una terminal:   ollama pull qwen2.5:14b-instruct
       (cabe de sobra en una RTX 5090; alternativa más ligera: qwen2.5:7b-instruct)
    3. Ollama queda escuchando en http://localhost:11434

Salidas:
    <interim_dir>/labels_llm.parquet      (se va guardando: si se corta, se reanuda)
    p1_rodilla/reports/tablas/acuerdo_llm_vs_dorado.csv

Uso:
    python -m p1_rodilla.scripts.s04_etiquetas_llm            # dorados + seleccionados
    python -m p1_rodilla.scripts.s04_etiquetas_llm --all      # todos los reportes
"""
from __future__ import annotations

from pathlib import Path

import argparse
import json
import urllib.request

import pandas as pd
from tqdm import tqdm

from common.config import REPO_ROOT, load_config
from common.utils import ensure_dir, get_logger
from p1_rodilla.src.data_io import load_gold, load_reports
from p1_rodilla.src.labels_common import agreement_report
from p1_rodilla.src.labels_rules import NEG, NM, POS, UNC

FINDINGS_DESC = {
    "acl": "anterior cruciate ligament (ACL) tear or injury (partial or complete)",
    "mcl": "medial collateral ligament (MCL) tear, sprain or injury",
    "medial_meniscus": "medial meniscus tear or damage",
    "lateral_meniscus": "lateral meniscus tear or damage",
    "medial_oa": "osteoarthritis / cartilage degeneration of the MEDIAL tibiofemoral compartment",
    "lateral_oa": "osteoarthritis / cartilage degeneration of the LATERAL tibiofemoral compartment",
    "pf_oa": "patellofemoral osteoarthritis or chondropathy (patella / trochlea cartilage)",
    "effusion": "joint effusion (more than physiological fluid)",
    "synovitis": "synovitis or synovial thickening",
    "baker_cyst": "Baker's (popliteal) cyst",
    "bone_contusion": "bone contusion / bone bruise / traumatic bone marrow edema",
    "fracture": "fracture (any bone of the knee)",
}
STATUS = {"present": POS, "absent": NEG, "uncertain": UNC, "not_mentioned": NM}

SYSTEM = (
    "You are an expert musculoskeletal radiologist. You read knee MRI reports written in any language "
    "(English, Spanish, German, Turkish, Greek, Russian, Croatian, French, Dutch, ...). For each finding, "
    "decide if the report states it is present, absent, uncertain (possible/suspected/cannot be excluded) "
    "or not_mentioned. Answer ONLY with the JSON object requested."
)


def build_prompt(report: str) -> str:
    lines = "\n".join(f'- "{k}": {v}' for k, v in FINDINGS_DESC.items())
    return (f"Findings:\n{lines}\n\nReport:\n\"\"\"\n{report[:6000]}\n\"\"\"\n\n"
            'Return JSON: {"<finding>": "present|absent|uncertain|not_mentioned", ...} with all 12 keys.')


def ask_ollama(host: str, model: str, prompt: str, temperature: float, timeout: int) -> dict:
    schema = {"type": "object",
              "properties": {k: {"type": "string", "enum": list(STATUS)} for k in FINDINGS_DESC},
              "required": list(FINDINGS_DESC)}
    body = {"model": model, "stream": False, "format": schema,
            "options": {"temperature": temperature},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        content = json.loads(r.read())["message"]["content"]
    return json.loads(content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "p1_rodilla" / "config.yaml"))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("s04")
    llm, names = cfg["llm"], cfg["labels"]["names"]
    interim = ensure_dir(cfg["data"]["interim_dir"])

    reports = load_reports(cfg)
    if not args.all:
        wanted = set()
        try:
            wanted |= set(load_gold(cfg).study_id)
        except KeyError:
            pass
        split = interim / "split.csv"
        if split.exists():
            wanted |= set(pd.read_csv(split, dtype={"study_id": str}).study_id)
        else:
            log.warning("Aún no existe split.csv (PASO 5). Se etiquetarán solo los dorados.")
        reports = reports[reports.study_id.isin(wanted)]
    log.info(f"Reportes a etiquetar con {llm['model']}: {len(reports):,}")

    out = interim / "labels_llm.parquet"
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["study_id"] + names)
    todo = reports[~reports.study_id.isin(set(done.study_id))]
    rows = done.to_dict("records")
    for i, (sid, text) in enumerate(tqdm(zip(todo.study_id, todo.report), total=len(todo), desc="LLM")):
        try:
            ans = ask_ollama(llm["host"], llm["model"], build_prompt(text), llm["temperature"], llm["timeout_s"])
            row = {"study_id": sid, **{k: STATUS.get(str(ans.get(k, "not_mentioned")), NM) for k in names}}
        except Exception as e:
            log.warning(f"{sid}: {e}")
            continue
        rows.append(row)
        if i % 25 == 0:  # guardado parcial para poder reanudar
            pd.DataFrame(rows).to_parquet(out, index=False)
    llm_df = pd.DataFrame(rows)
    llm_df.to_parquet(out, index=False)
    log.info(f"Guardado {out}")

    try:
        gold = load_gold(cfg)
        agr = agreement_report(llm_df, gold, names)
        p = ensure_dir(Path(cfg["data"]["reports_dir"]) / "tablas") / "acuerdo_llm_vs_dorado.csv"
        agr.to_csv(p, index=False)
        log.info("\nAcuerdo LLM vs dorado:\n" + agr.round(3).to_string(index=False))
    except KeyError as e:
        log.warning(str(e))


if __name__ == "__main__":
    main()
