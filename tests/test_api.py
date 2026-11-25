# -*- coding: utf-8 -*-
"""Pruebas de la API contra un Postgres real (DATABASE_URL). Se omiten si no hay base disponible."""
import io, math, struct, sys, wave
from pathlib import Path
import pandas as pd, pytest
RAIZ = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(RAIZ))
from vox.db import db_disponible
pytestmark = pytest.mark.skipif(not db_disponible(), reason="Postgres no disponible (define DATABASE_URL)")
from fastapi.testclient import TestClient
from vox.api import app

DF = pd.read_csv(RAIZ / "datos" / "ejemplo_rasgos.csv")
COLS = ["jitter", "shimmer", "f0_var", "spectral_flatness", "spectral_centroid", "hf_energy", "plateau_ratio", "peak_count", "mel_kurt"]


@pytest.fixture(scope="module")
def cli():
    with TestClient(app) as c: yield c


def _vec(i):
    r = DF.iloc[i]; return {"nombre": r.file, **{k: float(r[k]) for k in COLS if pd.notna(r[k])}}


def test_salud_y_umbrales(cli):
    s = cli.get("/salud").json(); assert s["postgres"] and s["java"] and s["ontologia"]
    u = cli.get("/ontologia/umbrales").json(); assert len(u["reglas"]) == 5
    assert {r["clase_indicio"] for r in u["reglas"]} >= {"JitterAnomalo", "CentroideEspectralAnomalo"}


def test_razonamiento_efimero_en_ram_no_persiste(cli):
    from vox.db import Caso, sesion
    from sqlalchemy import func, select
    with sesion() as s: antes = s.scalar(select(func.count()).select_from(Caso))
    r = cli.post("/ontologia/razonar", json=_vec(0)); assert r.status_code == 200
    m = r.json()["meta"]; assert m["almacen"] == "memoria" and r.json()["resultado"]["consistente"]
    with sesion() as s: assert s.scalar(select(func.count()).select_from(Caso)) == antes


def test_analisis_por_rasgos_persiste_caso_traza_y_escalamiento(cli):
    ids = []
    for i in range(len(DF)):
        r = cli.post("/analisis/rasgos", json=_vec(i)); assert r.status_code == 200, r.text
        ids.append(r.json())
        if any(x["resultado"]["veredicto"] == "ESCALAR" for x in ids) and any(x["resultado"]["veredicto"] != "ESCALAR" for x in ids) and len(ids) >= 6: break
    ok = [x for x in ids if x["estado"] == "completado"]; assert ok and all(x["resultado"]["veredicto"] in ("SINTETICO", "AUTENTICO", "ESCALAR") for x in ok)
    x = ok[0]; t = cli.get("/analisis/%s/traza" % x["run_id"]).json(); tipos = {e["evento"] for e in t["eventos"]}
    assert {"TASK_START", "TASK_END", "TOOL_CALL", "TOOL_RESULT", "CFP", "BID", "AWARD", "DELEGATION", "RULE_FIRED", "VERDICT"} <= tipos
    assert [e["seq"] for e in t["eventos"]] == sorted(e["seq"] for e in t["eventos"])
    esc = [x for x in ok if x["resultado"]["veredicto"] == "ESCALAR"]
    if esc:
        pend = {e["case_id"] for e in cli.get("/escalamientos?limite=500").json()}; assert esc[0]["case_id"] in pend or len(pend) >= 500
    c = cli.get("/casos/" + x["case_id"]).json(); assert c["rasgos"] and c["ontologia"]["consistente"] and c["version_ontologia"]


def test_resolver_actualiza_competencia_y_shapley(cli):
    antes = {(c["agent_id"], c["task_type"]): c["n_actualizaciones"] for c in cli.get("/competencias").json()}
    etiquetas = []
    for i, r in DF.iterrows():
        x = cli.post("/analisis/rasgos", json=_vec(i)).json(); assert x["estado"] == "completado"
        etiquetas.append((x["case_id"], r.label))
    for case_id, lab in etiquetas:
        rr = cli.post("/escalamientos/%s/resolver" % case_id, json={"etiqueta": lab}); assert rr.status_code == 200
    filas = cli.get("/competencias").json()          # la EMA converge, asi que se verifica el conteo de actualizaciones, no el valor
    assert all(c["n_actualizaciones"] > antes[(c["agent_id"], c["task_type"])] for c in filas) and all(0 <= c["competencia"] <= 1 for c in filas)
    sh = cli.get("/auditoria/shapley?minimo=20"); assert sh.status_code == 200, sh.text
    j = sh.json(); assert abs(j["suma_phi"] - j["v_N"]) < 1e-9 and len(j["agentes"]) == 5


def _wav(seg=2.0, sr=16000):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 180 * t / sr) * (0.5 + 0.5 * math.sin(2 * math.pi * 3 * t / sr)))) for t in range(int(seg * sr))))
    return b.getvalue()


def test_subida_de_audio_directo_y_borrado(cli, tmp_path):
    import glob, tempfile
    r = cli.post("/analisis", files={"audio": ("tono.wav", _wav(), "audio/wav")}, data={"modo": "directo", "sincrono": "true"})
    assert r.status_code in (200, 202), r.text; j = r.json()
    assert j["estado"] == "completado" and j["resultado"]["veredicto"] in ("SINTETICO", "AUTENTICO", "ESCALAR")
    assert not glob.glob(tempfile.gettempdir() + "/vox_*/*.wav")            # el audio no se conserva
    assert cli.post("/analisis", files={"audio": ("x.txt", b"hola", "text/plain")}).status_code == 415


def test_autenticacion_opcional(cli, monkeypatch):
    import vox.api as A
    monkeypatch.setattr(A, "API_KEY", "secreto")
    assert cli.get("/casos").status_code == 401
    assert cli.get("/casos", headers={"X-API-Key": "secreto"}).status_code == 200


def test_subida_de_audio_modo_crew_con_llm_simulado(cli, monkeypatch):
    pytest.importorskip("crewai")
    import vox.agentes as AG
    monkeypatch.setattr(AG, "LLM_MODELO", "simulado")
    r = cli.post("/analisis", files={"audio": ("tono.wav", _wav(), "audio/wav")}, data={"modo": "crew", "sincrono": "true"})
    j = r.json(); assert j["estado"] == "completado", j
    assert j["resultado"]["veredicto"] in ("SINTETICO", "AUTENTICO", "ESCALAR")
    tipos = {e["evento"] for e in cli.get("/analisis/%s/traza" % j["run_id"]).json()["eventos"]}
    assert {"DELEGATION", "TOOL_CALL", "VERDICT", "TASK_END"} <= tipos


# ------------------------------------------------------------------ validacion semantica y puente httpx
def test_endpoint_validar_rechaza_con_justificacion(cli):
    ok = cli.post("/ontologia/validar", json={"declaracion": {"etiquetas": ["AUTENTICO"], "hablante": "Ana"}}).json(); assert ok["consistente"]
    r = cli.post("/ontologia/validar", json={"declaracion": {"etiquetas": ["AUTENTICO"], "generado_por": ["VocoderGAN"]}}).json()
    assert not r["consistente"] and len(r["conflicto_minimo"]) == 2 and "fueGeneradoPor" in r["axiomas_violados"][0]["formal"]
    assert r["meta"]["almacen"] == "memoria" and r["meta"]["ejecuciones_hermit"] >= 2
    assert cli.post("/ontologia/validar", json={"declaracion": {"generado_por": ["Inventada"]}}).status_code == 422


def test_analisis_rechazado_se_persiste_con_traza(cli):
    d = {"etiquetas": ["AUTENTICO"], "generado_por": ["VocoderGAN"]}
    j = cli.post("/analisis/rasgos", json={**_vec(0), "nombre": "rechazo.mp3", "declaracion": d}).json()
    assert j["estado"] == "completado" and j["resultado"]["veredicto"] == "RECHAZADO" and not j["resultado"]["validacion"]["consistente"]
    tipos = {e["evento"] for e in cli.get("/analisis/%s/traza" % j["run_id"]).json()["eventos"]}
    assert {"REJECTION", "CONFLICT", "TOOL_CALL"} <= tipos
    from vox.db import Escalamiento, sesion
    from sqlalchemy import select
    with sesion() as s: assert s.scalar(select(Escalamiento).where(Escalamiento.case_id == j["case_id"])) is None        # un rechazo no es un escalamiento


def test_herramientas_de_la_crew_llaman_a_la_api_con_httpx(cli):
    """Con el cliente httpx inyectado (TestClient) las herramientas de razonamiento y validacion viajan por HTTP; la traza lo registra."""
    from vox import cliente_api
    cliente_api.FACTORY = lambda: cli
    try:
        j = cli.post("/analisis/rasgos", json={**_vec(2), "nombre": "http.mp3", "declaracion": {"etiquetas": ["AUTENTICO"], "hablante": "Ana"}}).json()
    finally: cliente_api.FACTORY = None
    assert j["estado"] == "completado", j
    ev = cli.get("/analisis/%s/traza?evento=TOOL_RESULT" % j["run_id"]).json()["eventos"]
    transporte = {e["payload"]["tool"]: e["payload"]["transporte"] for e in ev}
    assert transporte["razonar_con_ontologia"] == "http" and transporte["validar_consistencia_semantica"] == "http"


def test_subida_de_audio_con_declaracion_inconsistente_modo_crew(cli, monkeypatch):
    pytest.importorskip("crewai")
    import vox.agentes as AG, json as js
    monkeypatch.setattr(AG, "LLM_MODELO", "simulado")
    d = js.dumps({"etiquetas": ["AUTENTICO", "SINTETICO"]})
    j = cli.post("/analisis", files={"audio": ("tono.wav", _wav(), "audio/wav")}, data={"modo": "crew", "sincrono": "true", "declaracion": d}).json()
    assert j["estado"] == "completado" and j["resultado"]["veredicto"] == "RECHAZADO", j
    assert "DisjointClasses" in j["resultado"]["explicacion"]
