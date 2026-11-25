# -*- coding: utf-8 -*-
"""Capturador de trazas (Seccion 3.3 del diseno): eventos inmutables con run_id, seq, parent_seq, ts, agente y tarea.
Se acumulan en memoria durante la ejecucion y se persisten de una sola vez en Postgres (tabla eventos_traza)."""
import contextvars, hashlib, json, threading, time
from functools import wraps

AGENTE_DE_HERRAMIENTA = {"validar_consistencia_semantica": ("agente_validador", "T1"), "extraer_y_validar_rasgos": ("agente_perceptor", "T2"),
                         "contract_net_allocate": ("agente_licitador", "T3"), "buscar_patrones_espectrales": ("agente_espectral", "T4"),
                         "buscar_patrones_prosodicos": ("agente_prosodico", "T5"), "razonar_con_ontologia": ("agente_razonador", "T6"),
                         "decidir_veredicto": ("agente_razonador", "T6"), "construir_explicacion": ("agente_explicador", "T7"),
                         "escalar_a_analista": ("agente_explicador", "T7")}
_actual = contextvars.ContextVar("vox_trazador", default=None)
suplencia_activa = contextvars.ContextVar("vox_suplencia", default=False)   # True mientras el sistema completa tareas por su cuenta
_por_caso = {}; _lock = threading.Lock()


class Trazador:
    def __init__(self, run_id):
        self.run_id, self.eventos, self._seq, self._ultimo = run_id, [], 0, None
        self.licitacion = None; self.rechazado = False; self.cancelado = False
        self._l = threading.Lock()

    def evento(self, tipo, agente=None, tarea=None, payload=None, parent=None):
        with self._l:
            self._seq += 1
            e = dict(run_id=self.run_id, seq=self._seq, parent_seq=parent if parent is not None else self._ultimo,
                     ts=round(time.time(), 3), evento=tipo, agente=agente, tarea=tarea, payload=payload or {})
            self.eventos.append(e); self._ultimo = self._seq
            return self._seq

    # --- registro por caso / contexto
    def activar(self, case_id=None):
        _actual.set(self)
        if case_id:
            with _lock: _por_caso[case_id] = self

    def desactivar(self, case_id=None):
        _actual.set(None)
        if case_id:
            with _lock: _por_caso.pop(case_id, None)


def de(case_id=None):
    with _lock:
        t = _por_caso.get(case_id) if case_id else None
        unico = next(iter(_por_caso.values())) if len(_por_caso) == 1 else None
    return t or _actual.get() or unico       # ultimo recurso: hay una sola ejecucion activa


def emitir(case_id, tipo, agente=None, tarea=None, **payload):
    t = de(case_id)
    return t.evento(tipo, agente, tarea, payload) if t else None


def _hash(x): return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:12]


def trazada(nombre, caso_arg="case_id"):
    """Decorador: registra TOOL_CALL / TOOL_RESULT (latencia, error) alrededor de una funcion de herramienta."""
    def deco(fn):
        @wraps(fn)
        def interno(*a, **k):
            import os
            c0 = a[0] if a else k.get(caso_arg) or k.get("audio_path")
            cid = os.path.basename(c0) if nombre == "extraer_y_validar_rasgos" and isinstance(c0, str) else (c0 if isinstance(c0, str) and nombre != "contract_net_allocate" else None)
            ag, tarea = AGENTE_DE_HERRAMIENTA[nombre]
            tr = de(cid); t0 = time.time()
            s = tr.evento("TOOL_CALL", ag, tarea, {"tool": nombre, "args_hash": _hash([a, k])}) if tr else None
            try:
                r = fn(*a, **k)
            except Exception as e:                      # noqa: BLE001
                if tr: tr.evento("ERROR", ag, tarea, {"tipo": type(e).__name__, "mensaje": str(e)[:300], "tool": nombre}, parent=s)
                raise
            if tr:
                err = r.get("error") if isinstance(r, dict) else None
                tr.evento("TOOL_RESULT", ag, tarea, {"tool": nombre, "latencia_ms": round((time.time() - t0) * 1000, 1), "error": err,
                                                     "transporte": r.get("transporte") if isinstance(r, dict) else None, "resultado_hash": _hash(r)}, parent=s)
            return r
        return interno
    return deco
