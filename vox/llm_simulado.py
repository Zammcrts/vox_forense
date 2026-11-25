# -*- coding: utf-8 -*-
"""LLM SIMULADO y determinista para ejecutar la crew de CrewAI sin clave de API.
NO es un modelo de lenguaje: es un guion que emite el formato ReAct (Thought/Action/Action Input/Final Answer) que CrewAI espera, de modo que
manager, delegacion, herramientas, callbacks, trazas y persistencia se ejerciten de verdad. Sirve para pruebas y demostraciones;
la calidad de la redaccion y las decisiones de delegacion de un LLM real no se evaluan con el."""
import json, re
from crewai import BaseLLM

ROL = {"agente_espectral": "Especialista en Artefactos Espectrales", "agente_prosodico": "Especialista en Prosodia y Voz"}


def _texto(m):
    c = m.get("content", "")
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


def _ultima_observacion(msgs):
    tx = "\n".join(_texto(m) for m in msgs if m.get("role") != "system")
    partes = re.split(r"Observation:\s*", tx)
    if len(partes) < 2: return ""
    return re.split(r"\n\s*Analyze the tool result", partes[-1])[0].strip()


def _n_obs(msgs):
    return len(re.findall(r"Observation:", "\n".join(_texto(m) for m in msgs if m.get("role") != "system")))


def _accion(pensamiento, herramienta, entrada):
    return "Thought: %s\nAction: %s\nAction Input: %s" % (pensamiento, herramienta, json.dumps(entrada, ensure_ascii=False))


def _final(texto): return "Thought: I now know the final answer\nFinal Answer: " + (texto or "listo")


def guion(msgs):
    sistema = _texto(msgs[0]); rol = (re.match(r"You are ([^.\n]+)\.", sistema) or [None, ""])[1].strip()
    tarea = next((_texto(m) for m in msgs if m.get("role") != "system" and "Current Task:" in _texto(m)), _texto(msgs[-1]))
    n = _n_obs(msgs); obs = _ultima_observacion(msgs)
    m = re.search(r"Current Task:\s*(.*?)\n\s*\nThis is the expected", tarea, re.S) or re.search(r"Current Task:\s*(.*)", tarea, re.S)
    tarea_txt = m.group(1).strip() if m else tarea
    cid = (re.search(r"case_id '([^']+)'", tarea_txt) or re.search(r"case_id[:=]\s*'?([^\s',]+)", tarea_txt) or [None, None])[1]
    audio = (re.search(r"audio_path='([^']+)'", tarea_txt) or [None, None])[1]

    if rol == "Coordinador Forense de Audio":
        if n >= 1: return _final(obs)
        paso = (re.match(r"(T\d)", tarea_txt) or [None, "T?"])[1]
        deleg = lambda cow, tk: _accion("Delego la tarea al especialista indicado.", "delegate_work_to_coworker", {"task": tk, "context": "case_id '%s'. %s" % (cid, tarea_txt), "coworker": cow})
        if paso == "T1": return deleg("Validador Semantico", "Ejecuta validar_consistencia_semantica para el case_id '%s'." % cid)
        if paso == "T2": return deleg("Analista de Percepcion de Audio", "Usa extraer_y_validar_rasgos con audio_path='%s'. Reporta el case_id '%s'." % (audio, cid))
        if paso == "T3": return deleg("Licitador Contract Net", tarea_txt)
        ctx = tarea  # el contexto con el resultado de la licitacion viaja en el mensaje
        if paso in ("T4", "T5"):
            tt = "SPECTRAL" if paso == "T4" else "PROSODIC"
            g = re.search(r'"%s":\s*\{\s*"agent_id":\s*"(\w+)"' % tt, ctx)
            cow = ROL.get(g.group(1) if g else ("agente_espectral" if paso == "T4" else "agente_prosodico"))
            return deleg(cow, "Busca los patrones %s del case_id '%s'." % ("espectrales" if paso == "T4" else "prosodicos", cid))
        if paso == "T6": return deleg("Razonador Ontologico", "Ejecuta razonar_con_ontologia y luego decidir_veredicto para el case_id '%s'." % cid)
        if paso == "T7": return deleg("Explicador y Auditor de Decisiones", "Ejecuta construir_explicacion para el case_id '%s'; si el veredicto es ESCALAR, ejecuta escalar_a_analista con el motivo." % cid)
        return _final("Sin accion para la tarea.")

    if rol == "Validador Semantico":
        return _final(obs) if n else _accion("Valido la solicitud contra la ontologia con HermiT.", "validar_consistencia_semantica", {"case_id": cid})
    if rol == "Analista de Percepcion de Audio":
        return _final(obs) if n else _accion("Extraigo y valido los rasgos.", "extraer_y_validar_rasgos", {"audio_path": audio})
    if rol == "Licitador Contract Net":
        if n: return _final(obs)
        sub = re.search(r"subtasks_json=(\{.*\})\s+y deadline_ms", tarea_txt, re.S)
        return _accion("Ejecuto la subasta Contract Net.", "contract_net_allocate", {"subtasks_json": sub.group(1) if sub else "{}", "deadline_ms": 2000})
    if rol == "Especialista en Artefactos Espectrales":
        return _final(obs) if n else _accion("Evaluo los rasgos espectrales.", "buscar_patrones_espectrales", {"case_id": cid})
    if rol == "Especialista en Prosodia y Voz":
        return _final(obs) if n else _accion("Evaluo los rasgos prosodicos.", "buscar_patrones_prosodicos", {"case_id": cid})
    if rol == "Razonador Ontologico":
        if n == 0: return _accion("Razono con la ontologia (HermiT en RAM).", "razonar_con_ontologia", {"case_id": cid})
        if n == 1: return _accion("Aplico la politica de decision.", "decidir_veredicto", {"case_id": cid})
        return _final(obs)
    if rol == "Explicador y Auditor de Decisiones":
        if n == 0: return _accion("Construyo la explicacion con respaldo en la traza.", "construir_explicacion", {"case_id": cid})
        if n == 1 and "ESCALADO" in obs: return _accion("El caso se escala.", "escalar_a_analista", {"case_id": cid, "motivo": "Evidencia insuficiente o contradictoria (ver explicacion)."})
        return _final(obs if n == 1 else "Caso escalado a un analista humano.")
    return _final("Sin guion para el rol: " + rol)


class LLMSimulado(BaseLLM):
    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None, from_agent=None, response_model=None):
        msgs = messages if isinstance(messages, list) else [{"role": "user", "content": messages}]
        return guion(msgs)

    def supports_function_calling(self): return False
    def supports_stop_words(self): return True
    def get_context_window_size(self): return 100000
