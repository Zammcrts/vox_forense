# -*- coding: utf-8 -*-
"""Modo directo (sin LLM): ejecuta el mismo flujo T1-T5 llamando a las funciones de las herramientas.
Sirve para evaluar en lote, reproducir resultados y probar sin clave de API."""
import glob, os
import pandas as pd
from . import casos as K
from . import herramientas as H
from .config import cargar_umbrales
from .contract_net import subtareas_por_defecto
from . import ontologia as O
import json, time
from . import trazador as T

EXT = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".mp4")


def _tarea(cid, tid, agente, fn):
    """Envuelve una tarea T1-T5 con TASK_START / TASK_END (como hara el task_callback de CrewAI)."""
    T.emitir(cid, "TASK_START", agente, tid, task_id=tid); t0 = time.time()
    r = fn(); T.emitir(cid, "TASK_END", agente, tid, task_id=tid, estado="ok", t_fin=round(time.time() - t0, 3)); return r


def _flujo(audio_path, rasgos=None, resultado_lote=None, declaracion=None):
    """Flujo T1-T7. Si no hay un trazador activo (uso por CLI), crea uno local y lo libera al terminar."""
    cid = H.case_id_de(audio_path); propio = T.de(cid) is None
    if propio: tr = T.Trazador("local-" + cid); tr.activar(cid)
    if declaracion: K.registrar(cid, "declaracion", declaracion=declaracion)
    try: return _flujo_interno(audio_path, cid, rasgos, resultado_lote)
    finally:
        if propio: tr.desactivar(cid)


def _flujo_interno(audio_path, cid, rasgos=None, resultado_lote=None):
    v0 = _tarea(cid, "T1", "agente_validador", lambda: H.fn_validar_consistencia_semantica(cid))
    if not v0.get("ok", True): return dict(case_id=cid, veredicto="ERROR", error=v0.get("error"))
    if not v0.get("consistente", True):          # el Validador frena la operacion: no se analiza el audio
        ex = _tarea(cid, "T7", "agente_explicador", lambda: H.fn_construir_explicacion(cid))
        return dict(case_id=cid, licitacion=None, veredicto="RECHAZADO", regla="V1", n_indicios=None, n_naturalidad=None,
                    contradictoria=False, explicacion=ex["explicacion"], justificacion=v0["justificacion"])
    t1 = _tarea(cid, "T2", "agente_perceptor", lambda: H.fn_extraer_y_validar_rasgos(audio_path, rasgos))
    if not t1["ok"]: return dict(case_id=cid, veredicto="ERROR", error=t1["error"])
    lic = _tarea(cid, "T3", "agente_licitador", lambda: H.fn_contract_net_allocate(json.dumps(subtareas_por_defecto())))
    ejecutores = {w["agent_id"] for w in lic["awards"].values()}
    if "agente_espectral" in ejecutores: _tarea(cid, "T4", "agente_espectral", lambda: H.fn_buscar_patrones_espectrales(cid))
    if "agente_prosodico" in ejecutores: _tarea(cid, "T5", "agente_prosodico", lambda: H.fn_buscar_patrones_prosodicos(cid))
    t4 = _tarea(cid, "T6", "agente_razonador", lambda: H.fn_razonar_con_ontologia(cid, resultado_lote))
    if not t4["ok"]: return dict(case_id=cid, veredicto="ERROR", error=t4["error"])
    v = H.fn_decidir_veredicto(cid)
    def t5():
        ex = H.fn_construir_explicacion(cid)
        if v["veredicto"] == "ESCALAR": H.fn_escalar_a_analista(cid, v["motivo"])
        return ex
    ex = _tarea(cid, "T7", "agente_explicador", t5)
    return dict(case_id=cid, licitacion=lic["status"], veredicto=v["veredicto"], regla=v["regla"], n_indicios=v["n_indicios"],
                n_naturalidad=v["n_naturalidad"], contradictoria=v["contradictoria"], explicacion=ex["explicacion"])


def analizar_directo(audio_path, declaracion=None):
    K.limpiar(); return _flujo(audio_path, declaracion=declaracion)


def _etiqueta_por_nombre(ruta):
    """Etiqueta real inferida del nombre del archivo o de su carpeta (falso/fake/sintetico -> fake; real/autentico -> real); None si no hay pista."""
    t = ruta.lower().replace("\\", "/")
    if any(k in t for k in ("falso", "fake", "sint", "clon")): return "fake"
    if any(k in t for k in ("real", "autent", "genuin")): return "real"
    return None


def analizar_lote(carpeta=None, csv=None, salida=None):
    """Procesa una carpeta de audios o un CSV de rasgos ya extraidos. HermiT se ejecuta una sola vez para todo el lote."""
    cfg = cargar_umbrales(); K.limpiar(); usados = list(cfg["ontologia"])
    if csv:
        df = pd.read_csv(csv); filas = {r["file"]: r.to_dict() for _, r in df.iterrows()}
        origen = {k: k for k in filas}
    else:
        from .features import extraer
        archivos = sorted(f for f in glob.glob(os.path.join(carpeta, "**", "*"), recursive=True) if f.lower().endswith(EXT))
        filas = {}
        for f in archivos:
            try: filas[os.path.basename(f)] = extraer(f, label=_etiqueta_por_nombre(f))
            except Exception as e: print("ERROR", f, e)           # noqa: E722
        origen = {os.path.basename(f): f for f in archivos}
    casos = {k: {c: r.get(c) for c in usados if c in r} for k, r in filas.items()}
    lote = O.clasificar_lote(casos)
    res = []
    for k, r in filas.items():
        out = _flujo(origen.get(k, k), rasgos=r, resultado_lote=lote[k]); out.pop("explicacion", None)
        if r.get("label") in ("fake", "real"): out["etiqueta_real"] = r["label"]
        res.append(out)
    df = pd.DataFrame(res)
    if salida: df.to_csv(salida, index=False)
    return df
