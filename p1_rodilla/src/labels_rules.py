"""
Extracción de etiquetas débiles a partir de reportes radiológicos multilingües (reglas).

Procedimiento
-------------
1. Normalización: minúsculas, sin diacríticos y sin secciones de antecedentes o indicación.
2. Segmentación en cláusulas por puntuación y conectores de contraste.
3. Búsqueda de términos por hallazgo en en, es, de, tr, el, ru, bg, hr/bs/sr, fr y nl.
4. Estado de cada cláusula, en orden de prioridad: incierto, negado (tras eliminar falsas
   negaciones), normal sin lesión (negativo) y positivo.
5. Combinación por estudio: POS > UNC > NEG; sin menciones, NM.

Reglas de gravedad
------------------
* Ligamentos: esguince grado 1, señal intersticial o degeneración mucoide sin rotura completa: NEG.
* Meniscos: degeneración o señal grado 1-2 (Stoller) sin rotura explícita: UNC.
* Artrosis: positiva si se declara explícitamente (OA, artrosis, pinzamiento); la condropatía
  o los osteofitos aislados solo si el grado es alto.
* Derrame: leve, mínimo o escaso sin calificativo de moderado o abundante: NEG.
* Contusión: el edema óseo sin el término "contusión" solo con contexto traumático.
* Sinovitis: edema de la grasa de Hoffa como indicador (Hoffa-sinovitis, MOAKS).

Códigos de estado: POS = 1, NEG = 0, UNC = 2 (incierto), NM = -1 (no mencionado).
"""
from __future__ import annotations

import re
import unicodedata

POS, NEG, UNC, NM = 1, 0, 2, -1
STATUS_NAME = {POS: "positivo", NEG: "negativo", UNC: "incierto", NM: "no_mencionado"}


def strip_accents(text: str) -> str:
    """Elimina diacríticos (p. ej., 'lesión' -> 'lesion', 'ρήξη' -> 'ρηξη')."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


# Encabezados de antecedentes o indicación clínica; su contenido no describe hallazgos del estudio.
CLINICAL_HEADER = re.compile(
    r"(antecedentes clinicos|antecedentes|indicacion clinica|clinical (history|information|indication)|indication|"
    r"history|anamnese|klinische angaben|fragestellung|klinische inlichtingen|diagnostische vraagstelling|"
    r"endikasyon|klinik bilgi|клинични данни|anamneza)\s*:[ \t]*(\n[ \t]*)?[^.\n]{0,200}"
)


def normalize(text: str) -> str:
    """Minúsculas, sin diacríticos, sin secciones clínicas previas y con espacios simples."""
    text = strip_accents(text.casefold())
    text = text.replace("ı", "i").replace("œ", "oe").replace("ß", "ss")
    text = CLINICAL_HEADER.sub(" ", text)
    return re.sub(r"\s+", " ", text)


def rx(*parts: str) -> re.Pattern:
    """Compila la disyunción de patrones, sin diacríticos."""
    return re.compile("|".join(strip_accents(p) for p in parts))


# ----------------------------------------------------------------------------
# Vocabulario multilingüe (en, es, de, tr, el, ru, bg, hr/bs/sr, fr, nl)
# ----------------------------------------------------------------------------
NEGATION = rx(
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"negative for", r"\babsen", r"free of", r"ruled out", r"\bnor\b", r"\bnone\b",
    r"\bsin\b", r"ausencia", r"descart", r"\bningun", r"\bni\b",
    r"\bkein", r"\bohne\b", r"\bnicht\b", r"ausgeschlossen", r"\bfehlt",
    r"\byok\b", r"izlenmedi", r"saptanmadi", r"gorulmedi", r"izlenmemis", r"saptanmamis", r"gozlenmedi", r"rastlanmadi", r"\bdegil\b",
    r"χωρις", r"\bδεν\b", r"ουδεν", r"ανευ", r"απουσια",
    r"\bнет\b", r"\bбез\b", r"\bне\b", r"отсутств", r"\bняма\b", r"липсва",
    r"\bbez\b", r"\bnema\b", r"\bnije\b", r"\bnisu\b",
    r"\bpas\b", r"\bsans\b", r"\baucun",
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b",
)
UNCERTAIN = rx(
    r"possibl", r"probabl", r"suspect", r"suspicio", r"cannot be excluded", r"can't be excluded", r"not excluded",
    r"may represent", r"\blikely\b", r"questionabl", r"equivocal", r"\bsuggest",
    r"posible", r"sospech", r"no se descarta", r"no puede descartarse", r"sugest", r"dudos",
    r"verdacht", r"moglich", r"wahrscheinlich", r"fraglich", r"nicht ausgeschlossen", r"\bv\.a\.",
    r"olasi", r"muhtemel", r"supheli", r"ekarte edilemedi", r"dislanamadi",
    r"πιθαν", r"υποψ", r"δεν αποκλειεται",
    r"возможн", r"вероятн", r"подозр", r"не исключ", r"суспектн", r"съмнени",
    r"podria", r"sugier",
    r"moguc", r"vjerojatn", r"suspekt", r"ne iskljucuje", r"ne moze se iskljuc",
    r"suspicion", r"ne peut etre exclu", r"evoqu",
    r"mogelijk", r"waarschijnlijk", r"verdenking", r"niet uit te sluiten",
)
NORMAL = rx(
    r"intact", r"\bnormal", r"unremarkable", r"preserved", r"\bintegr", r"conservad", r"indemne",
    r"intakt", r"regelrecht", r"unauffallig", r"olagan", r"saglam",
    r"ακεραι", r"φυσιολογικ",
    r"интакт", r"без особенност", r"без особености", r"\bб\.о\.", r"норм", r"сохран", r"запазен", r"съхранен",
    r"uredn", r"ocuvan", r"normaln",
    r"preserve", r"normaal", r"\bgaaf",
)
LESION = rx(
    r"\btear", r"\btorn\b", r"ruptur", r"disrupt", r"\blesion", r"injur", r"sprain", r"strain", r"\bpartial", r"discontinu",
    r"desgarr", r"rotura", r"\broto\b", r"esguince", r"distension",
    r"\briss\b", r"gerissen", r"lasion", r"zerrung", r"teilruptur", r"\bdistorsion",
    r"y[i]rt[i][kg]", r"lezyon", r"hasar", r"burkul",
    r"ρηξη", r"κακωσ", r"βλαβη", r"διαστρεμμ",
    r"разрыв", r"повреждени", r"надрыв", r"растяжени",
    r"скъсван", r"скъсан", r"руптур", r"лезия", r"увреда", r"разкъсван",
    r"puknu", r"ozljed", r"lezij", r"distenzij",
    r"dechirure", r"entorse", r"fissur",
    r"scheur", r"ruptuur", r"letsel", r"laesie", r"verstuiking",
)
DEGENERATION = rx(r"degenera", r"dejenera", r"εκφυλ", r"дегенерат", r"degenerescence", r"degeneratie", r"mukoid", r"mucoid",
                  r"meniskopati", r"мукоидн")
# Señal meniscal de grado 1-2 (Stoller): alteración intrasustancia sin extensión a la superficie.
MENISCAL_LOW_GRADE = rx(r"grad\w* (i|ii|1|2)[ab]?\b", r"\b(i|ii|1|2)[ab]? (stepen|степен|derece)", r"стадий", r"\b2[abаб]\b",
                        r"степен 2", r"\bii[abаб]\b")
TEAR = rx(r"\btear", r"\btorn\b", r"ruptur", r"desgarr", r"rotura", r"\briss\b", r"y[i]rt[i][kg]", r"ρηξη", r"разрыв",
          r"скъсван", r"руптур", r"puknu", r"dechirure", r"scheur", r"ruptuur", r"bucket", r"kova sap")
# Lesión ligamentaria de bajo grado (esguince grado 1, señal intersticial, degeneración mucoide)
LIGAMENT_LOW_GRADE = rx(r"grad\w* (1|i)\b", r"\b(1|i) (stepen|степен|derece)", r"interstiti", r"intersticial",
                        r"mucoid", r"mukoid", r"мукоидн")
# Falsas negaciones ("no desplazada", "sin cambios"): se eliminan antes de buscar negaciones.
PSEUDO_NEGATION = rx(
    r"\bno desplazad\w*", r"non[- ]?displac\w*", r"undisplaced", r"nicht disloz\w*", r"nicht verschoben",
    r"sin cambios\w*", r"no (significant )?change\w*", r"unchanged", r"keine (wesentliche )?anderung\w*",
    r"sin desplazamiento", r"without displacement", r"ohne dislokation", r"sans deplacement",
    r"non deplace\w*", r"niet verplaatst", r"без смещения", r"bez pomaka", r"deplase olmayan",
)
CONTRAST = re.compile(r"\b(but|however|pero|sin embargo|aber|jedoch|ancak|fakat|αλλα|однако|\bно\b|\bali\b|mais|maar)\b")

ANATOMY = {
    "acl": rx(r"anterior cruciate", r"\bacl\b", r"\blca\b", r"cruzado anterior", r"vorder\w* kreuzband", r"\bvkb\b",
              r"on capraz", r"προσθιο\w* χιαστ", r"передн\w* крестообразн", r"prednj\w* krizn", r"croise anterieur",
              r"croise anterieur", r"voorste kruisband", r"\bvkb\b", r"anterior capraz", r"предн\w* кръстн"),
    "mcl": rx(r"medial collateral", r"\bmcl\b", r"colateral medial", r"colateral interno", r"lateral interno", r"\blcm\b",
              r"innenband", r"mediale\w* kollateral", r"ic yan bag", r"εσω πλαγι", r"медиальн\w* коллатеральн",
              r"большеберцов\w* коллатеральн", r"medijaln\w* kolateraln", r"collateral interne", r"collateral medial",
              r"lateral interne", r"\blli\b", r"mediale collaterale", r"mediale band", r"medial kollateral",
              r"медиал\w* колатерал"),
    "meniscus": rx(r"menisc", r"menisk", r"μηνισκ", r"мениск", r"meniscus", r"menisku"),
}
MEDIAL = rx(r"medial", r"intern", r"innen", r"\bic\b", r"\bεσω", r"медиал", r"внутренн", r"medijaln", r"mediaal")
LATERAL = rx(r"lateral", r"extern", r"aussen", r"\bdis\b", r"\bεξω", r"латерал", r"наружн", r"lateraln", r"lateraal")
# Lateralidad de estructuras patelofemorales: no se asigna a los compartimentos femorotibiales
PATELLAR_SIDE = rx(r"(medial|lateral)\w* (patellar )?facet\w*", r"faceta (medial|lateral|interna|externa)\w*",
                   r"(medial|lateral)\w* (patell|retinac)\w*", r"(medial|lateral)\w* patellafacet\w*")
PF = rx(r"patell", r"femoropatel", r"patelofemor", r"rotul", r"retropatell", r"kniescheib", r"επιγονατιδ", r"πατελ",
        r"надколен", r"пател", r"ivera", r"trochle", r"troclea", r"troklea", r"трохле", r"τροχιλ")
# Artrosis declarada explícitamente (o pinzamiento del espacio articular)
OA = rx(r"osteoarthr", r"(?<!hem)arthros", r"(?<!hem)artros", r"arthrot", r"gonarthr", r"gonartr", r"osteoartrit", r"artroz",
        r"οστεοαρθρ", r"(?<!αιμ)αρθρωσ", r"остеоартр", r"(?<!хем)артроз", r"arthrose", r"artrose", r"\boa\b",
        r"tricompart", r"trikompart", r"joint space narrow", r"eklem aralig\w* (belirgin )?daral", r"pinzamiento",
        r"gelenkspaltverschmaler")
# Daño condral u osteofitos: solo cuentan como artrosis si el grado es alto
CHONDRAL = rx(r"chondropath", r"condropat", r"chondromalac", r"condromalac", r"kondromalaz", r"хондромаляц", r"хондромалац",
              r"хондропат", r"χονδρομαλακ", r"knorpelschad", r"chondrosis",
              r"kikirdak", r"cartilage loss", r"cartilage thinning", r"cartilage defect", r"chondral (ulcer|defect|lesion)",
              r"ulceras? condral", r"хрущял\w* дефект", r"дефект\w* в хрущял", r"osteophyt", r"osteofit", r"остеофит", r"οστεοφυτ")
HIGH_GRADE = rx(r"grad\w* (iii|iv|3|4)\b", r"high[- ]grade", r"full[- ]thickness", r"espesor total", r"more than 50",
                r"\b(iii|iv)\b", r"grado (tres|cuatro)", r"\bdenud", r"bone on bone", r"степен (3|4)", r"grade 3", r"grade 4")
TRICOMPARTMENTAL = rx(r"tricompart", r"trikompart", r"tri-compart", r"трехкомпартмент", r"three compartment",
                      r"all (three|3) compartm", r"tres compartimentos", r"todos los compartimentos")
SIMPLE = {
    "effusion": rx(r"effusion", r"derrame", r"erguss", r"efuzyon", r"eklem\w*[^,]{0,40}sivi artisi", r"sivi artisi[^,]{0,40}eklem",
                   r"eklem sivisi\w* art", r"συλλογ", r"выпот", r"излив", r"izljev", r"epanchement", r"hydrops", r"gewrichtsvocht",
                   r"intra-articulair vocht", r"opzetting\w* (van )?(de )?suprapatell"),
    "synovitis": rx(r"synovit", r"sinovit", r"синовит", r"συνοβιτ", r"αρθρογυμνιτ", r"synovial thicken", r"engrosamiento sinovial",
                    r"synovialitis", r"synoviale verdikking", r"sinovijaln\w* zadeb", r"synovial hypertroph",
                    r"hypertrophy of the synovium"),
    "baker_cyst": rx(r"baker", r"popliteal cyst", r"quiste popliteo", r"poplitealzyste", r"popliteal kist", r"μπεικερ",
                     r"бейкер", r"бекер", r"подколенн\w* кист", r"bakerov", r"kyste poplite", r"popliteale cyste"),
    "bone_contusion": rx(r"contusion", r"bone bruise", r"bruis", r"kontusion", r"kontuzyon", r"θλαση", r"μωλωπ",
                         r"ушиб", r"kontuzij", r"contusion osseuse", r"botkneuzing", r"botcontusie", r"контузи", r"kissing"),
    "fracture": rx(r"fractur", r"fraktur", r"k[i]r[i]k", r"καταγμ", r"перелом", r"фрактур", r"prijelom", r"prelom", r"fractuur",
                   r"\bbreuk", r"avulsion", r"knochern\w* ausriss", r"avulsion osea", r"segond", r"subchondral\w* impression"),
}
# Edema de médula ósea: inespecífico salvo contexto traumático.
MARROW_EDEMA = rx(r"marrow (o)?edema", r"knochenmark\w*odem", r"knochenodem", r"kemik iligi\w* odem", r"οιδημα\w* (του )?(οστικ\w* )?μυελ",
                  r"отек костного мозга", r"edem kostane srzi", r"oedeme osseux", r"beenmergoedeem", r"botoedeem", r"edema oseo",
                  r"костно-?\s?мозъч\w* едем", r"костномозъч\w* едем")
COMPLETE = rx(r"complet", r"\bfull", r"\btotal", r"komplet", r"vollstandig", r"volledig", r"пълн", r"полн", r"πληρ", r"potpun")
TRAUMA = rx(r"traumat", r"posttraum", r"травмат", r"посттравмат", r"τραυματ", r"κακωσ", r"\bimpact", r"pivot", r"kissing")
# Hoffa-sinovitis (MOAKS): edema de la grasa de Hoffa como indicador de sinovitis sin contraste
HOFFA = rx(r"hoffa", r"infrapatellar fat pad (o)?edema", r"infrapatellar yag yastik\w* \w* ?odem")
# Gradación del derrame
MILD = rx(r"\bmild\b", r"minimal", r"\bsmall\b", r"\bslight", r"\btrace", r"\bsome\b", r"\bleve\b", r"\bminim", r"pequen",
          r"escaso", r"discret", r"gering", r"\bklein", r"\bwenig", r"hafif", r"az miktar", r"μικρ", r"ελαχ", r"небольш",
          r"незначит", r"минимал", r"малък", r"\bmali\b", r"\bblag", r"minimaal", r"\bklein", r"faible", r"\bpeu\b")
MODERATE_LARGE = rx(r"hemarthros", r"hemartros", r"хемартроз", r"αιμαρθρ", r"moderat", r"\blarge", r"marked", r"significant", r"extensive", r"abundant", r"massive", r"moderad",
                    r"importante", r"grande", r"\bmittel", r"ausgepragt", r"deutlich", r"\bgross", r"belirgin", r"yaygin",
                    r"\borta\b", r"ικαν", r"μετρι", r"μεγαλ", r"умерен", r"выраж", r"значит", r"изразен", r"umjeren",
                    r"izrazit", r"matig", r"uitgesproken", r"\bfors", r"\bveel\b", r"important")
PHYSIOLOGICAL = rx(r"physiolog", r"fisiolog", r"fizyolojik", r"φυσιολογικ\w* ποσ", r"физиологич", r"fizioloski", r"physiologique", r"fysiologisch")

LABELS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa", "pf_oa",
          "effusion", "synovitis", "baker_cyst", "bone_contusion", "fracture"]


def split_clauses(text: str) -> list[str]:
    """Divide el texto normalizado en cláusulas por puntuación y conectores de contraste."""
    parts = re.split(r"[.;\n\r•]+|\s-\s", text)
    out = []
    for p in parts:
        out.extend(CONTRAST.split(p))
    return [p.strip() for p in out if p and p.strip() and not CONTRAST.fullmatch(p.strip())]


def clause_status(clause: str, has_lesion: bool | None = None) -> int:
    """Estado de un hallazgo mencionado en la cláusula (incierto > negado > normal > positivo)."""
    if UNCERTAIN.search(clause):
        return UNC
    if NEGATION.search(PSEUDO_NEGATION.sub(" ", clause)):
        return NEG
    if has_lesion is False:
        return NEG if NORMAL.search(clause) else NM
    return POS


def combine(statuses: list[int]) -> int:
    """Estado del estudio a partir de los estados de sus cláusulas (POS > UNC > NEG > NM)."""
    if POS in statuses:
        return POS
    if UNC in statuses:
        return UNC
    if NEG in statuses:
        return NEG
    return NM


def extract_labels(report: str, evidence: dict[str, list[str]] | None = None) -> dict[str, int]:
    """
    Estados ``{hallazgo: código}`` de un reporte.

    Si se proporciona ``evidence``, se completa con las cláusulas que determinaron cada
    estado, en el formato ``"<estado>: <cláusula>"``.
    """
    found: dict[str, list[int]] = {k: [] for k in LABELS}
    n_prev = {k: 0 for k in LABELS}
    for clause in split_clauses(normalize(report)):
        lesion = bool(LESION.search(clause))
        # --- ligamentos ---
        for lab in ("acl", "mcl"):
            if ANATOMY[lab].search(clause):
                if LIGAMENT_LOW_GRADE.search(clause) and not COMPLETE.search(clause):
                    st = NEG
                else:
                    st = clause_status(clause, lesion)
                if st != NM:
                    found[lab].append(st)
        # --- meniscos (lado medial / lateral) ---
        if ANATOMY["meniscus"].search(clause):
            if (DEGENERATION.search(clause) or MENISCAL_LOW_GRADE.search(clause)) and not TEAR.search(clause):
                st = NEG if NEGATION.search(PSEUDO_NEGATION.sub(" ", clause)) else UNC
            elif lesion:
                st = clause_status(clause, True)
            else:
                st = clause_status(clause, False)
            if st != NM:
                side = PATELLAR_SIDE.sub(" ", clause)
                med, lat = bool(MEDIAL.search(side)), bool(LATERAL.search(side))
                if not med and not lat:  # sin lateralidad: negación a ambos, lesión como incierta
                    side_st = st if st == NEG else UNC
                    found["medial_meniscus"].append(side_st)
                    found["lateral_meniscus"].append(side_st)
                else:
                    if med:
                        found["medial_meniscus"].append(st)
                    if lat:
                        found["lateral_meniscus"].append(st)
        # --- osteoartrosis por compartimento ---
        oa_explicit, chondral = bool(OA.search(clause)), bool(CHONDRAL.search(clause))
        if oa_explicit or chondral:
            st = clause_status(clause)
            if not oa_explicit and st == POS and not HIGH_GRADE.search(clause):
                st = NM
            if st != NM:
                side = PATELLAR_SIDE.sub(" ", clause)
                comps = []
                if TRICOMPARTMENTAL.search(clause):
                    comps = ["medial_oa", "lateral_oa", "pf_oa"]
                else:
                    if MEDIAL.search(side):
                        comps.append("medial_oa")
                    if LATERAL.search(side):
                        comps.append("lateral_oa")
                    if PF.search(clause):
                        comps.append("pf_oa")
                if comps:
                    for c in comps:
                        found[c].append(st)
                else:
                    for c in ("medial_oa", "lateral_oa", "pf_oa"):
                        found[c].append(NEG if st == NEG else UNC)
        # --- hallazgos simples ---
        for lab, pat in SIMPLE.items():
            if pat.search(clause):
                if lab == "effusion" and PHYSIOLOGICAL.search(clause):
                    found[lab].append(NEG)
                    continue
                st = clause_status(clause)
                if lab == "effusion" and st == POS and MILD.search(clause) and not MODERATE_LARGE.search(clause):
                    st = NEG
                found[lab].append(st)
        # --- edema óseo sin mención de contusión ---
        if MARROW_EDEMA.search(clause) and not SIMPLE["bone_contusion"].search(clause):
            st = clause_status(clause)
            if st == NEG or TRAUMA.search(clause):
                found["bone_contusion"].append(st)
        # --- Hoffa-sinovitis ---
        if HOFFA.search(clause) and not SIMPLE["synovitis"].search(clause):
            st = clause_status(clause, bool(re.search(r"(o)?edem|odem|infiltrat|signal", clause)))
            if st != NM:
                found["synovitis"].append(st)
        if evidence is not None:
            for k in LABELS:
                for st in found[k][n_prev[k]:]:
                    evidence.setdefault(k, []).append(f"{STATUS_NAME[st]}: {clause}")
                n_prev[k] = len(found[k])
    return {k: combine(v) for k, v in found.items()}


def detect_language(text: str) -> str:
    """Código ISO 639-1 del idioma del reporte (langdetect), o 'unknown'."""
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text[:2000])
    except Exception:
        return "unknown"
