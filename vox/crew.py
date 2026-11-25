# -*- coding: utf-8 -*-
"""Ensamblado de la crew jerarquica (manager fuera de `agents`) y ejecucion de un caso."""
import contextvars, json, threading, unicodedata
from crewai import Crew, Process
from . import casos as K
from . import trazador as T
from .agentes import crear_agentes, crear_llm
from .tareas import crear_tareas
from . import herramientas as H
from .config import CREW_TIMEOUT_S
from .contract_net import subtareas_por_defecto
from .herramientas import case_id_de


def _traza_tarea(out):
    """task_callback: TASK_END por cada tarea terminada (agente = rol del agente que la ejecuto)."""
    tr = T.de()
    if tr: tr.evento("TASK_END", str(getattr(out, "agent", "")) or None, None, {"resumen": str(getattr(out, "summary", ""))[:200], "salida_hash": T._hash(str(getattr(out, "raw", "")))})


def _traza_paso(paso):
    """step_callback: cada accion del agente; las delegaciones del manager se registran como DELEGATION."""
    tr = T.de()
    if not tr: return
    herr = getattr(paso, "tool", None)
    if herr:
        tipo = "DELEGATION" if "elegate" in str(herr) else "STEP"
        tr.evento(tipo, None, None, {"herramienta": str(herr), "entrada": str(getattr(paso, "tool_input", ""))[:300]})


def construir_crew(llm=None, verbose=True):
    manager, validador, perceptor, licitador, espectral, prosodico, razonador, explicador = crear_agentes(llm)
    tareas = crear_tareas(validador, perceptor, licitador, espectral, prosodico, razonador, explicador)
    return Crew(agents=[validador, perceptor, licitador, espectral, prosodico, razonador, explicador], tasks=tareas,
                process=Process.hierarchical, manager_agent=manager, verbose=verbose,
                task_callback=_traza_tarea, step_callback=_traza_paso)


def _faltantes(cid):
    """Etapas T1-T7 cuyo resultado no quedo en el almacen del caso (el LLM no llamo la herramienta o la llamo mal)."""
    c = K.caso(cid); tr = T.de(cid)
    if c.get("rechazado"): return [] if c.get("explicacion") else ["T7"]
    f = []
    if "validacion" not in c: f.append("T1")
    if "facts" not in c: f.append("T2")
    if not getattr(tr, "licitacion", None): f.append("T3")
    if "hallazgos_espectral" not in c: f.append("T4")
    if "hallazgos_prosodico" not in c: f.append("T5")
    if "ontologia" not in c: f.append("T6")
    if "veredicto" not in c: f.append("T6v")
    if "explicacion" not in c: f.append("T7")
    elif (c.get("veredicto") or {}).get("veredicto") == "ESCALAR" and not c.get("escalado"): f.append("T7e")
    return f


def _suplir(cid, audio_path, motivo):
    """Completa de forma determinista las etapas que la Crew no ejecuto, dejando SUPLENCIA en la traza (nunca se oculta)."""
    from .pipeline import _tarea
    tok = T.suplencia_activa.set(True); hechas = []
    try:
        for et in _faltantes(cid):
            T.emitir(cid, "SUPLENCIA", None, et[:2], etapa=et, motivo=motivo); hechas.append(et)
            c = K.caso(cid)
            if et == "T1": _tarea(cid, "T1", "agente_validador", lambda: H.fn_validar_consistencia_semantica(cid))
            elif et == "T2": _tarea(cid, "T2", "agente_perceptor", lambda: H.fn_extraer_y_validar_rasgos(audio_path))
            elif et == "T3": _tarea(cid, "T3", "agente_licitador", lambda: H.fn_contract_net_allocate(json.dumps(subtareas_por_defecto())))
            elif et == "T4": _tarea(cid, "T4", "agente_espectral", lambda: H.fn_buscar_patrones_espectrales(cid))
            elif et == "T5": _tarea(cid, "T5", "agente_prosodico", lambda: H.fn_buscar_patrones_prosodicos(cid))
            elif et == "T6": _tarea(cid, "T6", "agente_razonador", lambda: H.fn_razonar_con_ontologia(cid))
            elif et == "T6v": H.fn_decidir_veredicto(cid)
            elif et == "T7": _tarea(cid, "T7", "agente_explicador", lambda: H.fn_construir_explicacion(cid))
            elif et == "T7e": H.fn_escalar_a_analista(cid, (c.get("veredicto") or {}).get("motivo", "sin motivo"))
            if K.caso(cid).get("rechazado"): break          # el Validador freno la operacion: no se completa nada mas
        c = K.caso(cid)
        if (c.get("veredicto") or {}).get("veredicto") == "ESCALAR" and c.get("explicacion") and not c.get("escalado"):
            H.fn_escalar_a_analista(cid, c["veredicto"]["motivo"]); hechas.append("T7e")
    finally:
        T.suplencia_activa.reset(tok)
    return hechas


def analizar_con_crew(audio_path, llm=None, verbose=True, timeout_s=None):
    """Ejecuta la crew sobre un audio con un tope de tiempo. El veredicto autoritativo es el que dejo la herramienta decidir_veredicto en el
    almacen de casos (no el texto libre del LLM). Si la Crew falla, se agota el tiempo o un agente no llama su herramienta (p. ej. un modelo
    local pequeno que responde sin delegar), el sistema completa las etapas faltantes con las mismas herramientas y lo declara
    (`crew_estado`, `suplencias` y eventos SUPLENCIA en la traza)."""
    cid = case_id_de(audio_path); timeout_s = CREW_TIMEOUT_S if timeout_s is None else timeout_s
    crew = construir_crew(llm or crear_llm(), verbose)
    caja = {}
    def correr(): 
        try: caja["salida"] = crew.kickoff(inputs={"audio_path": audio_path, "case_id": cid})
        except Exception as e: caja["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])     # noqa: BLE001
    hilo = threading.Thread(target=contextvars.copy_context().run, args=(correr,), daemon=True); hilo.start(); hilo.join(timeout_s)
    tr = T.de(cid)
    if hilo.is_alive():
        if tr: tr.cancelado = True; tr.evento("ERROR", None, None, {"tipo": "CrewTimeout", "mensaje": "La Crew excedio %s s" % timeout_s})
        estado = "tiempo_agotado"
    elif "error" in caja:
        if tr: tr.evento("ERROR", None, None, {"tipo": "CrewError", "mensaje": caja["error"]})
        estado = "error"
    else: estado = "completa"
    suplencias = _suplir(cid, audio_path, estado) if _faltantes(cid) else []
    if suplencias and estado == "completa": estado = "con_suplencia"
    elif suplencias: estado += "+suplencia"
    c = K.caso(cid); v = c.get("veredicto"); salida = caja.get("salida", "")
    claves = {"SINTETICO": ("SINTETIC",), "AUTENTICO": ("AUTENTIC",), "ESCALAR": ("ESCALAR", "ESCALAD"), "RECHAZADO": ("RECHAZ",)}
    texto = unicodedata.normalize("NFD", str(salida).upper()); texto = "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")
    veredicto = v["veredicto"] if v else ("RECHAZADO" if c.get("rechazado") else None)
    return dict(case_id=cid, veredicto=veredicto, detalle=v, explicacion=c.get("explicacion"), informe_llm=str(salida),
                crew_estado=estado, suplencias=suplencias,
                coincide_con_herramienta=bool(veredicto) and any(k in texto for k in claves.get(veredicto, ())))
