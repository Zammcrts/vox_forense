# -*- coding: utf-8 -*-
"""Capa de servicio: ejecuta un caso (modo directo o crew), lo traza y lo persiste en Postgres; actualiza la competencia del Contract Net
cuando un analista resuelve un escalamiento (fase 6 del protocolo)."""
import os, shutil, tempfile, uuid
from datetime import datetime, timezone
from sqlalchemy import select
from . import casos as K, trazador as T, ontologia as O, contract_net as CN
from .db import Caso, Competencia, Ejecucion, Escalamiento, EventoTraza, sesion
from .herramientas import case_id_de

EXT_PERMITIDAS = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".mp4")


def nuevo_run_id(): return "r-" + uuid.uuid4().hex[:10]


def registrar_ejecucion(run_id, case_id, modo, nombre):
    with sesion() as s, s.begin(): s.add(Ejecucion(run_id=run_id, case_id=case_id, modo=modo, nombre_original=nombre))


def _persistir(run_id, case_id, tr, salida, nombre):
    c = K.caso(case_id); lic = getattr(tr, "licitacion", None) or {}
    v = c.get("veredicto") or {}
    onto = dict(c.get("ontologia") or {})
    if c.get("declaracion"): onto["declaracion"] = c["declaracion"]
    if c.get("validacion"): onto["validacion"] = c["validacion"]
    with sesion() as s, s.begin():
        s.add(Caso(case_id=case_id, run_id=run_id, nombre_original=nombre, rasgos=c.get("facts") or {}, faltantes=c.get("faltantes") or [],
                   hallazgos={"espectral": c.get("hallazgos_espectral"), "prosodico": c.get("hallazgos_prosodico")},
                   ontologia=onto, licitacion=lic, veredicto=v.get("veredicto"), regla=v.get("regla"), motivo=v.get("motivo"),
                   n_indicios=v.get("n_indicios"), n_naturalidad=v.get("n_naturalidad"), contradictoria=bool(v.get("contradictoria")),
                   explicacion=c.get("explicacion"), version_ontologia=O.version_ontologia()))
        s.flush()
        if v.get("veredicto") == "ESCALAR":
            s.add(Escalamiento(case_id=case_id, motivo=c.get("motivo_escalamiento") or v.get("motivo") or "sin motivo"))
        s.add_all(EventoTraza(**e) for e in tr.eventos)
        ej = s.get(Ejecucion, run_id); ej.estado = "completado"; ej.terminado = datetime.now(timezone.utc)


def ejecutar_audio(ruta_audio, run_id, nombre_original, modo="directo", llm=None, declaracion=None):
    """Procesa un archivo de audio YA guardado como <run_id><ext> en un directorio temporal. Borra el audio al terminar."""
    from .pipeline import _flujo
    case_id = case_id_de(ruta_audio); tr = T.Trazador(run_id); tr.persistente = True; tr.activar(case_id)
    if declaracion: K.registrar(case_id, "declaracion", declaracion=declaracion)
    try:
        if modo == "crew":
            from .crew import analizar_con_crew
            salida = analizar_con_crew(ruta_audio, llm)
        else:
            salida = _flujo(ruta_audio)          # la declaracion ya esta en el almacen del caso
        if salida.get("veredicto") in (None, "ERROR"):
            raise RuntimeError(salida.get("error") or "la ejecucion no produjo veredicto")
        _persistir(run_id, case_id, tr, salida, nombre_original)
        return dict(run_id=run_id, case_id=case_id, estado="completado", veredicto=salida["veredicto"])
    except Exception as e:                                            # noqa: BLE001
        tr.evento("ERROR", None, None, {"tipo": type(e).__name__, "mensaje": str(e)[:500]})
        with sesion() as s, s.begin():
            ej = s.get(Ejecucion, run_id)
            if ej:
                ej.estado = "error"; ej.error = str(e)[:1000]; ej.terminado = datetime.now(timezone.utc)
                if not s.scalar(select(EventoTraza.id).where(EventoTraza.run_id == run_id).limit(1)):
                    s.add_all(EventoTraza(**ev) for ev in tr.eventos)
        return dict(run_id=run_id, case_id=case_id, estado="error", error=str(e))
    finally:
        tr.desactivar(case_id); K.eliminar(case_id)
        try: os.remove(ruta_audio)                                   # el audio es efimero: no se conserva
        except OSError: pass


def ejecutar_rasgos(rasgos, nombre, run_id, modo="directo", declaracion=None):
    """Igual que ejecutar_audio pero con el vector de rasgos ya calculado (no requiere ffmpeg ni el archivo)."""
    from .pipeline import _flujo
    case_id = "%s__%s" % (run_id, os.path.basename(nombre)); tr = T.Trazador(run_id); tr.persistente = True; tr.activar(case_id)
    registrar_ejecucion(run_id, case_id, modo, nombre)
    try:
        salida = _flujo(case_id, rasgos=rasgos, declaracion=declaracion)
        if salida.get("veredicto") in (None, "ERROR"): raise RuntimeError(salida.get("error") or "sin veredicto")
        _persistir(run_id, case_id, tr, salida, nombre)
        return dict(run_id=run_id, case_id=case_id, estado="completado", veredicto=salida["veredicto"])
    except Exception as e:                                            # noqa: BLE001
        with sesion() as s, s.begin():
            ej = s.get(Ejecucion, run_id); ej.estado = "error"; ej.error = str(e)[:1000]; ej.terminado = datetime.now(timezone.utc)
        return dict(run_id=run_id, case_id=case_id, estado="error", error=str(e))
    finally:
        tr.desactivar(case_id); K.eliminar(case_id)


# ------------------------------------------------------------------ competencia persistente y retroalimentacion
def cargar_competencias():
    with sesion() as s: return {(c.agent_id, c.task_type): c.competencia for c in s.scalars(select(Competencia))}


def conectar_competencias():
    CN.PROVEEDOR_COMPETENCIA = cargar_competencias


def resolver_escalamiento(case_id, etiqueta, comentario=None):
    """El analista fija la etiqueta real. Con ella se actualiza (EMA) la competencia de cada agente segun si sus hallazgos acertaron
    (indicio -> fake, naturalidad -> real); fase 6 del protocolo Contract Net."""
    assert etiqueta in ("real", "fake")
    with sesion() as s, s.begin():
        caso = s.get(Caso, case_id); esc = s.scalar(select(Escalamiento).where(Escalamiento.case_id == case_id))
        if not caso: raise KeyError("caso no encontrado")
        caso.etiqueta_real = etiqueta
        if esc: esc.estado = "resuelto"; esc.etiqueta_analista = etiqueta; esc.comentario = comentario; esc.resuelto = datetime.now(timezone.utc)
        cambios = {}
        for clave, agent_id in (("espectral", "agente_espectral"), ("prosodico", "agente_prosodico")):
            hs = [h for h in (caso.hallazgos.get(clave) or []) if h["estado"] in ("indicio", "naturalidad")]
            if not hs: continue
            for h in hs:
                tarea = CN.TAREA_DE_RASGO[h["rasgo"]]
                acierto = (h["estado"] == "indicio") == (etiqueta == "fake")
                fila = s.get(Competencia, (agent_id, tarea))
                if fila:
                    fila.competencia = (1 - CN.ALPHA) * fila.competencia + CN.ALPHA * (1.0 if acierto else 0.0)
                    fila.n_actualizaciones += 1; cambios[f"{agent_id}/{tarea}"] = round(fila.competencia, 4)
    return dict(case_id=case_id, etiqueta=etiqueta, competencias_actualizadas=cambios)
