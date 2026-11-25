# -*- coding: utf-8 -*-
import json, math, sys
from pathlib import Path
import pandas as pd, pytest
RAIZ = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(RAIZ))
from vox import ontologia as O, contract_net as C, politica as P, herramientas as H, casos as K
from vox.config import cargar_umbrales

CFG = cargar_umbrales(); COLS = [r["feature"] for ag in CFG["reglas"] for r in CFG["reglas"][ag]]
DF = pd.read_csv(RAIZ / "datos" / "ejemplo_rasgos.csv")


def test_umbrales_vienen_de_la_ontologia():
    u = O.umbrales()
    assert u["JitterAnomalo"][1]["maxExclusive"] == pytest.approx(CFG["reglas"]["Pr"][1]["t_ind"])
    assert O.evaluar_rasgo("spectral_centroid", 2000.0) == "indicio" and O.evaluar_rasgo("spectral_centroid", 900.0) == "naturalidad"
    assert O.evaluar_rasgo("jitter", float("nan")) == "sin_valor"


def test_politica():
    assert P.decidir(2, 1)["veredicto"] == "SINTETICO" and P.decidir(0, 2)["veredicto"] == "AUTENTICO"
    assert P.decidir(1, 1)["veredicto"] == "ESCALAR" and P.decidir(0, 0)["regla"] == "R0" and P.decidir(2, 2)["veredicto"] == "ESCALAR"


def test_contract_net_estados():
    reg = C.build_registry(); sub = C.subtareas_por_defecto()["subtasks"]
    r = C.contract_net([{**s, "deadline_ms": 2000} for s in sub], reg); assert r["status"] == "AWARDED"
    assert r["awards"]["PROSODIC"]["agent_id"] == "agente_prosodico" and r["awards"]["SPECTRAL"]["agent_id"] == "agente_espectral"
    assert C.contract_net([{"task_type": "SPECTRAL", "required_features": ["mfcc_delta"], "deadline_ms": 2000}], reg)["status"] == "NO_BIDS"
    assert C.contract_net([{"task_type": "SPECTRAL", "required_features": ["hf_energy"], "deadline_ms": 10}], reg)["status"] == "NO_BIDS"   # no llega al plazo
    out = json.loads(json.dumps(H.fn_contract_net_allocate("no es json")))
    assert out["status"] == "ERROR"


def test_hermit_coincide_con_umbrales_de_la_ontologia():
    casos = {r.file: {c: float(r[c]) for c in COLS} for _, r in DF.iterrows()}
    res = O.clasificar_lote(casos)
    for _, r in DF.iterrows():
        est = [O.evaluar_rasgo(c, float(r[c])) for c in COLS]
        assert res[r.file]["consistente"]
        assert (res[r.file]["n_indicios"], res[r.file]["n_naturalidad"]) == (est.count("indicio"), est.count("naturalidad"))


def test_flujo_directo_por_lote_precision():
    from vox.pipeline import analizar_lote
    df = analizar_lote(csv=str(RAIZ / "datos" / "ejemplo_rasgos.csv"))
    dec = df[df.veredicto.isin(["SINTETICO", "AUTENTICO"])]
    assert len(dec) >= 1 and ((dec.veredicto == "SINTETICO") == (dec.etiqueta_real == "fake")).mean() >= 0.9
    assert set(df.veredicto) <= {"SINTETICO", "AUTENTICO", "ESCALAR"}


def test_audio_sin_rasgos_se_escala():
    K.limpiar(); H.fn_extraer_y_validar_rasgos("x.mp3", rasgos={"duration_s": 10.0})
    H.fn_buscar_patrones_espectrales("x.mp3"); H.fn_buscar_patrones_prosodicos("x.mp3")
    assert H.fn_razonar_con_ontologia("x.mp3")["n_indicios"] == 0
    assert H.fn_decidir_veredicto("x.mp3")["veredicto"] == "ESCALAR"


def test_shapley_eficiencia():
    from vox.auditoria import auditar
    r = auditar(str(RAIZ / "datos" / "ejemplo_rasgos.csv"), solo_prueba=False)
    assert abs(r["suma_phi"] - r["v_N"]) < 1e-9


def test_crew_se_construye_jerarquica():
    pytest.importorskip("crewai")
    from vox.crew import construir_crew
    from vox.agentes import crear_llm
    crew = construir_crew(crear_llm("openai/gpt-4o-mini"), verbose=False)
    assert len(crew.agents) == 7 and len(crew.tasks) == 7 and crew.manager_agent.allow_delegation and not crew.manager_agent.tools
    assert str(crew.process).lower().endswith("hierarchical") and crew.manager_agent not in crew.agents


def test_crew_completa_con_llm_simulado():
    """Ejecuta la crew JERARQUICA real de CrewAI (manager + 6 agentes + herramientas) con el guion determinista (sin API)."""
    pytest.importorskip("crewai")
    from vox import trazador as T
    from vox.crew import analizar_con_crew
    from vox.agentes import crear_llm
    import vox.features as F
    fila = DF.iloc[0].to_dict(); original = F.extraer; F.extraer = lambda path, label=None: dict(fila)
    K.limpiar(); tr = T.Trazador("r-test-crew"); tr.activar("crew.wav")
    try:
        r = analizar_con_crew("crew.wav", crear_llm("simulado"), verbose=False)
    finally:
        F.extraer = original; tr.desactivar("crew.wav")
    assert r["veredicto"] in ("SINTETICO", "AUTENTICO", "ESCALAR") and r["coincide_con_herramienta"] and r["explicacion"]
    tipos = {e["evento"] for e in tr.eventos}
    assert {"TASK_END", "DELEGATION", "CFP", "BID", "AWARD", "RULE_FIRED", "VERDICT"} <= tipos
    herramientas = {e["payload"].get("tool") for e in tr.eventos if e["evento"] == "TOOL_CALL"}
    assert {"validar_consistencia_semantica", "extraer_y_validar_rasgos", "contract_net_allocate", "buscar_patrones_espectrales", "buscar_patrones_prosodicos",
            "razonar_con_ontologia", "decidir_veredicto", "construir_explicacion"} <= herramientas


# ------------------------------------------------------------------ Validador semantico (HermiT detecta inconsistencias)
def test_validador_acepta_declaracion_consistente():
    r = O.validar_declaracion({"etiquetas": ["AUTENTICO"], "hablante": "Ana"})
    assert r["consistente"] and r["conflicto_minimo"] == []


@pytest.mark.parametrize("decl,aserciones,formal", [
    ({"etiquetas": ["AUTENTICO"], "hablante": "Ana", "generado_por": ["VocoderGAN"]}, 2, "fueGeneradoPor"),
    ({"etiquetas": ["AUTENTICO", "SINTETICO"]}, 2, "DisjointClasses(AudioAutentico, AudioSintetico)"),
    ({"etiquetas": ["AUTENTICO"], "hablante": "Ana", "hablantes_adicionales": ["Luis"]}, 3, "producidoPor"),
    ({"rasgos_declarados": [{"rasgo": "jitter", "valor": 0.001, "clase": "natural"}]}, 1, "JitterAnomalo"),
])
def test_validador_rechaza_con_conflicto_minimo(decl, aserciones, formal):
    r = O.validar_declaracion(decl)
    assert not r["consistente"] and len(r["conflicto_minimo"]) == aserciones
    assert any(formal in a["formal"] for a in r["axiomas_violados"]) and "INCONSISTENTE" in r["justificacion"]


def test_flujo_directo_rechaza_y_bloquea_herramientas():
    from vox.pipeline import _flujo
    K.limpiar()
    r = _flujo("rechazo.wav", rasgos=dict(DF.iloc[0]), declaracion={"etiquetas": ["AUTENTICO"], "generado_por": ["VocoderGAN"]})
    assert r["veredicto"] == "RECHAZADO" and r["regla"] == "V1" and "fueGeneradoPor" in r["explicacion"]
    assert K.caso("rechazo.wav").get("facts") is None                        # el audio ni siquiera se proceso
    assert H.fn_extraer_y_validar_rasgos("rechazo.wav", dict(DF.iloc[0]))["bloqueado"]
    assert H.fn_buscar_patrones_espectrales("rechazo.wav")["bloqueado"] and H.fn_razonar_con_ontologia("rechazo.wav")["bloqueado"]


def test_crew_rechaza_solicitud_inconsistente_con_llm_simulado():
    pytest.importorskip("crewai")
    from vox import trazador as T
    from vox.crew import analizar_con_crew
    from vox.agentes import crear_llm
    import vox.features as F
    fila = DF.iloc[0].to_dict(); original = F.extraer; F.extraer = lambda path, label=None: dict(fila)
    K.limpiar(); K.registrar("crew_rechazo.wav", "declaracion", declaracion={"etiquetas": ["AUTENTICO"], "generado_por": ["VocoderGAN"]})
    tr = T.Trazador("r-test-rechazo"); tr.activar("crew_rechazo.wav")
    try: r = analizar_con_crew("crew_rechazo.wav", crear_llm("simulado"), verbose=False)
    finally: F.extraer = original; tr.desactivar("crew_rechazo.wav")
    assert r["veredicto"] == "RECHAZADO" and r["coincide_con_herramienta"] and "fueGeneradoPor" in r["explicacion"]
    tipos = [e["evento"] for e in tr.eventos]; assert "REJECTION" in tipos and "CONFLICT" in tipos
    assert K.caso("crew_rechazo.wav").get("hallazgos_espectral") is None          # el flujo se detuvo: nadie analizo el audio


class _LLMQueNoLlamaHerramientas:
    """Doble de un modelo local pequeno que responde texto sin delegar ni llamar herramientas (el fallo observado con llama3.1:8b)."""
    @staticmethod
    def hacer():
        from vox.llm_simulado import LLMSimulado
        class Mudo(LLMSimulado):
            def call(self, messages, *a, **k): return "Final Answer: Lo siento, no puedo delegar; invento hallazgos sin usar herramientas."
        return Mudo(model="simulado")


def test_crew_con_llm_que_no_llama_herramientas_se_completa_por_suplencia():
    pytest.importorskip("crewai")
    from vox import trazador as T
    from vox.crew import analizar_con_crew
    import vox.features as F
    fila = DF.iloc[0].to_dict(); original = F.extraer; F.extraer = lambda path, label=None: dict(fila)
    K.limpiar(); tr = T.Trazador("r-test-suplencia"); tr.activar("suplencia.wav")
    try: r = analizar_con_crew("suplencia.wav", _LLMQueNoLlamaHerramientas.hacer(), verbose=False, timeout_s=120)
    finally: F.extraer = original; tr.desactivar("suplencia.wav")
    assert r["veredicto"] in ("SINTETICO", "AUTENTICO", "ESCALAR")          # el veredicto sale de la herramienta, no del texto del LLM
    assert r["crew_estado"] == "con_suplencia" and r["suplencias"]
    assert "SUPLENCIA" in [e["evento"] for e in tr.eventos] and "VERDICT" in [e["evento"] for e in tr.eventos]
    assert r["explicacion"]


def test_crew_con_tiempo_agotado_se_completa_y_se_declara():
    pytest.importorskip("crewai")
    import time
    from vox import trazador as T
    from vox.crew import analizar_con_crew
    from vox.llm_simulado import LLMSimulado
    import vox.features as F
    class Lento(LLMSimulado):
        def call(self, messages, *a, **k): time.sleep(5); return "Final Answer: tarde"
    fila = DF.iloc[0].to_dict(); original = F.extraer; F.extraer = lambda path, label=None: dict(fila)
    K.limpiar(); tr = T.Trazador("r-test-timeout"); tr.activar("timeout.wav")
    try: r = analizar_con_crew("timeout.wav", Lento(model="simulado"), verbose=False, timeout_s=1)
    finally: F.extraer = original; tr.desactivar("timeout.wav")
    assert r["crew_estado"] == "tiempo_agotado+suplencia" and r["veredicto"]
    assert any(e["evento"] == "ERROR" and e["payload"].get("tipo") == "CrewTimeout" for e in tr.eventos)
