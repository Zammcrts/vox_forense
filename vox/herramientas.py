# -*- coding: utf-8 -*-
"""Herramientas del sistema. Cada una es una funcion pura (fn_*) mas un envoltorio @tool de CrewAI.
Los agentes intercambian un `case_id`; los datos viven en vox.casos (no se pasan cifras por el LLM)."""
import json, math, os
from . import casos as K
from . import trazador as T
from . import cliente_api as API
from .config import RANGOS, RUTA_ESCALAMIENTOS, cargar_umbrales
from .contract_net import TASK_TYPES, TAREA_DE_RASGO, contract_net, build_registry, subtareas_por_defecto
from . import ontologia as O
from .politica import decidir

NEXT_ACTION = {"AWARDED": "DELEGAR_A_GANADORES", "PARTIAL": "DELEGAR_ASIGNADAS_Y_MARCAR_BRECHA", "NO_BIDS": "ESCALAR_A_ANALISTA_HUMANO"}


def _fmt(facets):
    op = {"minExclusive": ">", "maxExclusive": "<", "minInclusive": ">=", "maxInclusive": "<="}
    return " y ".join("%s %.6g" % (op[k], v) for k, v in facets.items())


def case_id_de(audio_path):
    return os.path.basename(audio_path)


def _bloqueado(case_id):
    """El Validador Semantico frena la operacion: tras un rechazo las demas herramientas no se ejecutan."""
    if getattr(T.de(case_id), "cancelado", False) and not T.suplencia_activa.get():
        return dict(case_id=case_id, ok=False, bloqueado=True, error="Operacion cancelada: la Crew excedio su tiempo y el sistema completo el caso por suplencia.")
    if K.caso(case_id).get("rechazado") or getattr(T.de(case_id), "rechazado", False):
        return dict(case_id=case_id, ok=False, bloqueado=True, error="Operacion bloqueada: el Validador Semantico rechazo la solicitud por inconsistencia con la ontologia.")
    return None


def _j(x): return json.dumps(x, ensure_ascii=False, default=float)


def _rasgos_de(ag, cfg):
    return [r["feature"] for r in cfg["reglas"][ag]]


# ------------------------------------------------------------------ Validador semantico (T1)
@T.trazada("validar_consistencia_semantica")
def fn_validar_consistencia_semantica(case_id):
    """Valida con HermiT (RAM, efimero) lo que la solicitud DECLARA sobre el audio. Si contradice un axioma de la ontologia,
    rechaza la operacion con justificacion logica y bloquea las demas herramientas."""
    c = K.caso(case_id); decl = c.get("declaracion")
    if not decl:
        K.registrar(case_id, "validacion", validacion=dict(consistente=True, justificacion="Sin declaraciones que validar."))
        return dict(case_id=case_id, ok=True, consistente=True, sin_declaracion=True, transporte="local")
    try:
        if API.disponible(): r = API.post("/ontologia/validar", {"declaracion": decl}); r["transporte"] = "http"
        else: r = O.validar_declaracion(decl); r["transporte"] = "local"
    except Exception as e:                                                                   # noqa: BLE001
        return dict(case_id=case_id, ok=False, error="No se pudo validar la solicitud: %s" % e)
    K.registrar(case_id, "validacion", validacion=r)
    if r["consistente"]:
        T.emitir(case_id, "RULE_FIRED", "agente_validador", "T1", regla_id="HermiT:consistencia", hechos_premisa=r["aserciones"], conclusion="consistente")
        return dict(case_id=case_id, ok=True, consistente=True, transporte=r["transporte"], justificacion=r["justificacion"])
    expl = ["Veredicto: SOLICITUD RECHAZADA por inconsistencia semantica (regla V1).", "Justificacion logica: " + r["justificacion"],
            "La operacion se detuvo antes de analizar el audio; no se emitio ningun veredicto sobre su autenticidad."]
    K.registrar(case_id, "rechazo", rechazado=True, explicacion="\n".join(expl),
                veredicto=dict(veredicto="RECHAZADO", regla="V1", motivo=r["justificacion"], n_indicios=None, n_naturalidad=None, contradictoria=False))
    tr = T.de(case_id)
    if tr: tr.rechazado = True
    T.emitir(case_id, "CONFLICT", "agente_validador", "T1", reglas_en_conflicto=r["conflicto_minimo"], axiomas=[a["formal"] for a in r["axiomas_violados"]])
    T.emitir(case_id, "REJECTION", "agente_validador", "T1", motivo="INCONSISTENCIA_SEMANTICA", justificacion=r["justificacion"])
    return dict(case_id=case_id, ok=True, consistente=False, rechazado=True, transporte=r["transporte"], conflicto_minimo=r["conflicto_minimo"],
                axiomas_violados=r["axiomas_violados"], justificacion=r["justificacion"])


# ------------------------------------------------------------------ Perceptor
@T.trazada("extraer_y_validar_rasgos")
def fn_extraer_y_validar_rasgos(audio_path, rasgos=None):
    """Extrae (o recibe ya calculados) los rasgos acusticos, los valida y los guarda como hechos del caso."""
    cid = case_id_de(audio_path)
    if (b := _bloqueado(cid)): return b
    try:
        if rasgos is None:
            from .features import extraer
            rasgos = extraer(audio_path)
    except Exception as e:                      # noqa: BLE001
        K.registrar(cid, "error_percepcion", error=str(e)); return dict(case_id=cid, ok=False, error=str(e))
    facts, faltantes, avisos = {}, [], []
    for k, v in rasgos.items():
        if k in ("file", "label"): continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            faltantes.append(k); continue
        v = float(v); facts[k] = v
        if k in RANGOS and not (RANGOS[k][0] <= v <= RANGOS[k][1]): avisos.append("%s=%.6g fuera de rango %s" % (k, v, RANGOS[k]))
    K.registrar(cid, "percepcion", audio_path=audio_path, facts=facts, faltantes=faltantes, avisos=avisos)
    return dict(case_id=cid, ok=True, n_rasgos=len(facts), faltantes=faltantes, avisos=avisos, duracion_s=facts.get("duration_s"))


# ------------------------------------------------------------------ Manager (Contract Net)
@T.trazada("contract_net_allocate")
def fn_contract_net_allocate(subtasks_json, deadline_ms=2000):
    if getattr(T.de(), "rechazado", False): return dict(status="ERROR", ok=False, bloqueado=True, error="Operacion bloqueada por el Validador Semantico.")
    if getattr(T.de(), "cancelado", False) and not T.suplencia_activa.get(): return dict(status="ERROR", ok=False, bloqueado=True, error="Operacion cancelada por tiempo.")
    try:
        subtasks = json.loads(subtasks_json)["subtasks"]
        if deadline_ms <= 0 or not subtasks: raise ValueError("deadline_ms debe ser > 0 y subtasks no vacio")
        cfps, ajustes = [], []
        por_defecto = {x["task_type"]: x["required_features"] for x in subtareas_por_defecto()["subtasks"]}
        for s in subtasks:
            if s["task_type"] not in TASK_TYPES: raise ValueError("task_type invalido: %s" % s["task_type"])
            # un LLM puede inventar nombres de rasgos: se conservan solo los que pertenecen a ese tipo de tarea; si no queda ninguno, se usan los del catalogo
            validos = [f for f in (s.get("required_features") or []) if TAREA_DE_RASGO.get(f) == s["task_type"]]
            if validos != list(s.get("required_features") or []):
                ajustes.append({"task_type": s["task_type"], "recibido": s.get("required_features"), "usado": validos or por_defecto.get(s["task_type"], [])})
            cfps.append({**s, "required_features": validos or por_defecto.get(s["task_type"], []), "deadline_ms": deadline_ms})
        result = contract_net(cfps, build_registry()); result["next_action"] = NEXT_ACTION[result["status"]]
        if ajustes: result["ajustes_de_entrada"] = ajustes
        tr = T.de()
        if tr:
            tr.licitacion = result
            for c in cfps: tr.evento("CFP", "agente_manager", "T3", {"task_type": c["task_type"], "required_features": c["required_features"], "deadline_ms": c["deadline_ms"]})
            for l in result["log"]:
                for b in l.get("bids", []): tr.evento("BID", b["agent_id"], "T3", {k: b[k] for k in ("task_type", "utility", "competence", "coverage", "load", "est_latency_ms")})
            for tt, w in result["awards"].items():
                tr.evento("AWARD", "agente_manager", "T3", {"task_type": tt, **w})
                tr.evento("DELEGATION", "agente_manager", "T3", {"tarea": tt, "agente_destino": w["agent_id"], "motivo": "mayor utilidad Contract Net (%s)" % w["utility"]})
            if result["status"] in ("PARTIAL", "NO_BIDS"): tr.evento("ESCALATION", "agente_manager", "T3", {"motivo": result["status"], "sin_asignar": result["unassigned"]})
    except (KeyError, ValueError, TypeError) as exc:
        result = {"status": "ERROR", "error": str(exc), "next_action": "CORREGIR_ENTRADA_Y_REINTENTAR"}
    return result


# ------------------------------------------------------------------ Especialistas
def _hallazgos(case_id, ag, clave):
    if (b := _bloqueado(case_id)): return b
    cfg = cargar_umbrales(); c = K.caso(case_id); facts = c.get("facts")
    if facts is None: return dict(case_id=case_id, ok=False, error="El caso no tiene hechos: ejecuta primero la percepcion.")
    out = []
    for f in _rasgos_de(ag, cfg):
        r = O.regla_de(f, cfg); auc = next(x["auc"] for x in cfg["reglas"][ag] if x["feature"] == f)
        v = facts.get(f); estado = O.evaluar_rasgo(f, v, cfg)
        out.append(dict(rasgo=f, valor=None if v is None else round(v, 6), estado=estado,
                        clase_ontologica=(r["clase_indicio"] if estado == "indicio" else r["clase_naturalidad"] if estado == "naturalidad" else None),
                        peso_evidencia=round(abs(auc - .5) * 2, 3), tecnica_sospechada=r["tecnica"] if estado == "indicio" else None))
    K.registrar(case_id, "hallazgos_" + clave, **{"hallazgos_" + clave: out})
    for h in out:
        if h["estado"] in ("indicio", "naturalidad"):
            T.emitir(case_id, "RULE_FIRED", "agente_" + clave, "T4" if clave == "espectral" else "T5", regla_id=h["clase_ontologica"],
                     hechos_premisa=[{h["rasgo"]: h["valor"]}], conclusion=h["estado"], peso_evidencia=h["peso_evidencia"])
    return dict(case_id=case_id, ok=True, agente=clave, hallazgos=out,
                n_indicios=sum(h["estado"] == "indicio" for h in out), n_naturalidad=sum(h["estado"] == "naturalidad" for h in out),
                sin_evidencia=[h["rasgo"] for h in out if h["estado"] in ("sin_evidencia", "sin_valor")])


@T.trazada("buscar_patrones_espectrales")
def fn_buscar_patrones_espectrales(case_id): return _hallazgos(case_id, "E", "espectral")
@T.trazada("buscar_patrones_prosodicos")
def fn_buscar_patrones_prosodicos(case_id): return _hallazgos(case_id, "Pr", "prosodico")


# ------------------------------------------------------------------ Razonador
@T.trazada("razonar_con_ontologia")
def fn_razonar_con_ontologia(case_id, resultado_lote=None):
    """Inserta el caso como individuos de la ontologia, ejecuta HermiT y guarda las clases inferidas.
    `resultado_lote` permite reutilizar un razonamiento por lotes ya hecho (modo --lote)."""
    if (b := _bloqueado(case_id)): return b
    cfg = cargar_umbrales(); c = K.caso(case_id); facts = c.get("facts")
    if facts is None: return dict(case_id=case_id, ok=False, error="El caso no tiene hechos.")
    usados = {f: facts[f] for f in cfg["ontologia"] if f in facts}
    if resultado_lote: inf, meta = resultado_lote, {"almacen": "lote"}
    elif API.disponible():                       # Reasoning Offloading: la inferencia se pide a la API asincrona (httpx)
        try: r = API.post("/ontologia/razonar", {"nombre": case_id, **usados}); inf, meta = r["resultado"], dict(r["meta"], transporte="http")
        except Exception as e: return dict(case_id=case_id, ok=False, error="Fallo la llamada HTTP a la API: %s" % e)       # noqa: BLE001
    else:
        res, meta = O.razonar_efimero({case_id: usados}); inf = res[case_id]; meta["transporte"] = "local"
    if not inf.get("consistente", False):
        K.registrar(case_id, "ontologia_inconsistente", ontologia=inf); return dict(case_id=case_id, ok=False, error="Ontologia inconsistente con los hechos del caso.")
    previos = (c.get("hallazgos_espectral") or []) + (c.get("hallazgos_prosodico") or [])
    disc = []
    for h in previos:
        if h["estado"] == "sin_valor": continue
        r = inf["rasgos"].get(h["rasgo"], {})
        if (h["estado"] == "indicio") != r.get("indicio", False) or (h["estado"] == "naturalidad") != r.get("naturalidad", False):
            disc.append(h["rasgo"])
    K.registrar(case_id, "razonamiento", ontologia=inf, discrepancias_agentes=disc)
    T.emitir(case_id, "RULE_FIRED", "agente_razonador", "T6", regla_id="HermiT", hechos_premisa=sorted(usados), conclusion=inf["clases_audio"], meta_razonamiento=meta)
    if "AudioConEvidenciaContradictoria" in inf["clases_audio"] or disc:
        T.emitir(case_id, "CONFLICT", "agente_razonador", "T6", reglas_en_conflicto=inf["clases_audio"], discrepancias_con_especialistas=disc)
    return dict(case_id=case_id, ok=True, transporte=meta.get("transporte"), consistente=True, clases_audio=inf["clases_audio"], n_indicios=inf["n_indicios"],
                n_naturalidad=inf["n_naturalidad"], tecnicas_sospechadas=inf["tecnicas"], discrepancias_con_especialistas=disc)


@T.trazada("decidir_veredicto")
def fn_decidir_veredicto(case_id):
    c = K.caso(case_id); inf = c.get("ontologia")
    if c.get("rechazado"): return dict(case_id=case_id, ok=True, **c["veredicto"])
    if not inf: return dict(case_id=case_id, ok=False, error="Falta el razonamiento ontologico.")
    d = decidir(inf["n_indicios"], inf["n_naturalidad"])
    d.update(n_indicios=inf["n_indicios"], n_naturalidad=inf["n_naturalidad"], contradictoria="AudioConEvidenciaContradictoria" in inf["clases_audio"])
    K.registrar(case_id, "veredicto", veredicto=d)
    if d["contradictoria"]: T.emitir(case_id, "CONFLICT_RESOLVED", "agente_razonador", "T6", criterio=d["regla"], resultado=d["veredicto"])
    T.emitir(case_id, "VERDICT", "agente_razonador", "T6", audio_id=case_id, veredicto=d["veredicto"], regla=d["regla"], n_indicios=d["n_indicios"], n_naturalidad=d["n_naturalidad"])
    return dict(case_id=case_id, ok=True, **d)


# ------------------------------------------------------------------ Explicador / Auditor
@T.trazada("construir_explicacion")
def fn_construir_explicacion(case_id):
    c = K.caso(case_id); v = c.get("veredicto"); inf = c.get("ontologia")
    if c.get("rechazado"): return dict(case_id=case_id, ok=True, explicacion=c["explicacion"])
    if not v or not inf: return dict(case_id=case_id, ok=False, error="Faltan veredicto u ontologia: no se puede explicar sin respaldo.")
    cfg = cargar_umbrales(); hs = (c.get("hallazgos_espectral") or []) + (c.get("hallazgos_prosodico") or [])
    lineas = []
    for h in hs:
        if h["estado"] in ("indicio", "naturalidad"):
            r = O.regla_de(h["rasgo"], cfg)
            lineas.append("%s = %s -> %s (clase %s; indicio si valor %s, naturalidad si valor %s)" % (h["rasgo"], h["valor"], h["estado"], h["clase_ontologica"],
                          _fmt(r["umbral_indicio"]), _fmt(r["umbral_naturalidad"])))
    sin = [h["rasgo"] for h in hs if h["estado"] not in ("indicio", "naturalidad")]
    etiqueta = {"SINTETICO": "voz SINTETICA", "AUTENTICO": "voz AUTENTICA", "ESCALAR": "ESCALADO a analista humano"}[v["veredicto"]]
    txt = ["Veredicto: %s (regla %s)." % (etiqueta, v["regla"]), "Motivo: " + v["motivo"],
           "Hechos y reglas aplicadas:"] + ["  - " + l for l in lineas]
    if sin: txt.append("Rasgos sin evidencia (zona intermedia o sin valor): " + ", ".join(sin) + ".")
    if v["contradictoria"]:
        txt.append("La ontologia clasifico el audio como AudioConEvidenciaContradictoria: hay indicios de sintesis y rasgos de naturalidad a la vez."
                   + ("" if v["veredicto"] == "ESCALAR" else " El conflicto se resolvio por la regla %s (%d indicios frente a %d rasgos de naturalidad)." % (v["regla"], v["n_indicios"], v["n_naturalidad"])))
    if inf["tecnicas"]: txt.append("Tecnicas de sintesis sospechadas (hipotesis de la ontologia, no validadas): " + ", ".join(inf["tecnicas"]) + ".")
    if c.get("faltantes"): txt.append("Rasgos no disponibles: " + ", ".join(c["faltantes"]) + ".")
    txt.append("Limite: umbrales calibrados con un corpus propio; no se garantiza generalizacion a otros generadores o grabaciones.")
    K.registrar(case_id, "explicacion", explicacion="\n".join(txt))
    return dict(case_id=case_id, ok=True, explicacion="\n".join(txt))


@T.trazada("escalar_a_analista")
def fn_escalar_a_analista(case_id, motivo):
    reg = dict(case_id=case_id, motivo=motivo, veredicto=K.caso(case_id).get("veredicto"))
    if not getattr(T.de(case_id), "persistente", False):      # con Postgres el escalamiento va a la tabla `escalamientos`; sin BD, a un .jsonl
        RUTA_ESCALAMIENTOS.parent.mkdir(parents=True, exist_ok=True)
        with open(RUTA_ESCALAMIENTOS, "a", encoding="utf-8") as f: f.write(_j(reg) + "\n")
    K.registrar(case_id, "escalamiento", escalado=True, motivo_escalamiento=motivo)
    T.emitir(case_id, "ESCALATION", "agente_explicador", "T7", motivo=motivo)
    return dict(case_id=case_id, ok=True, escalado=True, registro=str(RUTA_ESCALAMIENTOS))


# ================================================================== envoltorios CrewAI
try:
    from crewai.tools import tool
except ImportError:           # permite usar el modo directo sin instalar crewai
    tool = None

if tool is not None:
    @tool("validar_consistencia_semantica")
    def validar_consistencia_semantica(case_id: str) -> str:
        """Valida con el razonador HermiT (en RAM, efimero) lo que la solicitud declara sobre el audio (etiqueta, hablante, tecnica de sintesis, rasgos).
        Si contradice un axioma de la ontologia, RECHAZA la operacion con justificacion logica y bloquea el resto del flujo. Usa httpx hacia la API cuando VOX_API_URL esta definida.
        Args: case_id: identificador del caso. Returns: JSON con consistente, conflicto minimo, axiomas violados y justificacion."""
        return _j(fn_validar_consistencia_semantica(case_id))

    @tool("extraer_y_validar_rasgos")
    def extraer_y_validar_rasgos(audio_path: str) -> str:
        """Extrae y valida el vector de rasgos acusticos de un audio y lo guarda como hechos del caso.
        Args: audio_path: ruta del archivo de audio (mp3, m4a, wav...). Returns: JSON con case_id, rasgos validos, faltantes y avisos."""
        return _j(fn_extraer_y_validar_rasgos(audio_path))

    @tool("contract_net_allocate")
    def contract_net_allocate(subtasks_json: str, deadline_ms: int = 2000) -> str:
        """Asigna subtareas de analisis forense al agente especialista mas competente con el protocolo Contract Net.
        Args: subtasks_json: JSON {"subtasks":[{"task_type":"SPECTRAL|PROSODIC|PLATEAU_STATS","required_features":["..."]}]};
        deadline_ms: plazo maximo por subtarea en ms (> 0). Returns: JSON con status, awards, unassigned, log y next_action."""
        return _j(fn_contract_net_allocate(subtasks_json, deadline_ms))

    @tool("buscar_patrones_espectrales")
    def buscar_patrones_espectrales(case_id: str) -> str:
        """Evalua los rasgos espectrales del caso contra las clases de la ontologia (indicio de sintesis / naturalidad) con su peso de evidencia.
        Args: case_id: identificador del caso devuelto por la percepcion. Returns: JSON con hallazgos."""
        return _j(fn_buscar_patrones_espectrales(case_id))

    @tool("buscar_patrones_prosodicos")
    def buscar_patrones_prosodicos(case_id: str) -> str:
        """Evalua los rasgos prosodicos (jitter, variabilidad de F0...) del caso contra las clases de la ontologia.
        Args: case_id: identificador del caso. Returns: JSON con hallazgos."""
        return _j(fn_buscar_patrones_prosodicos(case_id))

    @tool("razonar_con_ontologia")
    def razonar_con_ontologia(case_id: str) -> str:
        """Inserta el caso en la ontologia OWL, ejecuta el razonador HermiT y devuelve las clases inferidas, el conteo de indicios y de rasgos de naturalidad.
        Args: case_id: identificador del caso. Returns: JSON con clases_audio, n_indicios, n_naturalidad y discrepancias."""
        return _j(fn_razonar_con_ontologia(case_id))

    @tool("decidir_veredicto")
    def decidir_veredicto(case_id: str) -> str:
        """Aplica la politica de decision (SINTETICO / AUTENTICO / ESCALAR) sobre lo inferido por la ontologia. Es determinista: no la sustituyas por tu criterio.
        Args: case_id: identificador del caso. Returns: JSON con veredicto, regla y motivo."""
        return _j(fn_decidir_veredicto(case_id))

    @tool("construir_explicacion")
    def construir_explicacion(case_id: str) -> str:
        """Construye la cadena percepcion -> hechos -> reglas -> conclusion con respaldo en la traza del caso.
        Args: case_id: identificador del caso. Returns: JSON con la explicacion."""
        return _j(fn_construir_explicacion(case_id))

    @tool("escalar_a_analista")
    def escalar_a_analista(case_id: str, motivo: str) -> str:
        """Registra el caso para revision de un analista forense humano (usar cuando el veredicto sea ESCALAR).
        Args: case_id: identificador del caso; motivo: razon del escalamiento. Returns: JSON de confirmacion."""
        return _j(fn_escalar_a_analista(case_id, motivo))
