# -*- coding: utf-8 -*-
"""Demostracion en vivo (5 min) contra la API en marcha.  Uso:
    python demo/demo.py --audio fragmentos/audio_001.mp3 [--modo crew] [--url http://127.0.0.1:8000] [--api-key ...]
Minuto 1: entorno | 2: caso exitoso (crew, trazas, Postgres) | 3: rechazo semantico (HermiT) | 4-5: trazas y auditoria."""
import argparse, json, sys, time
import httpx

ap = argparse.ArgumentParser(); ap.add_argument("--audio", required=True); ap.add_argument("--modo", default="crew", choices=["crew", "directo"])
ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--api-key", default=""); a = ap.parse_args()
H = {"X-API-Key": a.api_key} if a.api_key else {}
c = httpx.Client(base_url=a.url, headers=H, timeout=600)
titulo = lambda s: print("\n" + "=" * 8, s, "=" * 8)
pausa = lambda: input("\n[Enter para continuar] ") if sys.stdin.isatty() else time.sleep(1)


def mostrar_traza(run_id, tipos=("TASK_START", "DELEGATION", "TOOL_CALL", "CFP", "BID", "AWARD", "RULE_FIRED", "CONFLICT", "REJECTION", "VERDICT", "ESCALATION", "SUPLENCIA", "ERROR")):
    for e in c.get("/analisis/%s/traza" % run_id).json()["eventos"]:
        if e["evento"] == "TOOL_RESULT" and (e["payload"].get("transporte") or e["payload"].get("error")):
            p = e["payload"]; print("  #%-3s %-11s %-18s %-4s %s%s" % (e["seq"], "TOOL_RESULT", e["agente"] or "-", e["tarea"] or "-", p["tool"], (" [via %s]" % p["transporte"]) if p.get("transporte") else ""),
                                   ("<- BLOQUEADA: " + p["error"][:70]) if p.get("error") else "")
        if e["evento"] in tipos:
            p = e["payload"]; det = p.get("tool") or p.get("task_id") or p.get("regla_id") or p.get("veredicto") or p.get("motivo") or p.get("agente_destino") or p.get("task_type") or ""
            tr = (" [%s]" % p["transporte"]) if p.get("transporte") else ""
            print("  #%-3s %-11s %-18s %-4s %s%s" % (e["seq"], e["evento"], e["agente"] or "-", e["tarea"] or "-", str(det)[:60], tr))


titulo("MINUTO 1 · Entorno y servicios")
s = c.get("/salud").json(); print(json.dumps({k: s[k] for k in ("estado", "postgres", "java", "ffmpeg", "crewai", "ontologia", "tmp_jvm")}, ensure_ascii=False, indent=1))
u = c.get("/ontologia/umbrales").json(); print("Ontologia cargada (version %s). Reglas leidas del .owx:" % u["version_ontologia"])
for r in u["reglas"]: print("  %-18s indicio: %-28s clase: %s" % (r["feature"], r["umbral_indicio"], r["clase_indicio"]))
pausa()

titulo("MINUTO 2 · Caso 1: solicitud valida (modo %s)" % a.modo)
with open(a.audio, "rb") as f:
    r = c.post("/analisis", files={"audio": (a.audio.split("/")[-1].split("\\")[-1], f)}, data={"modo": a.modo, "sincrono": "true"}).json()
res = r["resultado"]; print("Estado:", r["estado"], "| Veredicto:", res["veredicto"], "| regla", res["regla"]); print(res["explicacion"])
print("\nTraza (log de la Crew y de las @tool):"); mostrar_traza(r["run_id"])
print("\nFila persistida en Postgres:", json.dumps({k: c.get("/casos/" + r["case_id"]).json()[k] for k in ("case_id", "veredicto", "regla", "n_indicios", "n_naturalidad", "version_ontologia")}, ensure_ascii=False))
pausa()

titulo("MINUTO 3 · Caso 2: solicitud que viola un axioma (rechazo semantico con HermiT)")
decl = {"etiquetas": ["AUTENTICO"], "hablante": "Ana", "generado_por": ["VocoderGAN"]}
print("Declaracion:", json.dumps(decl, ensure_ascii=False))
with open(a.audio, "rb") as f:
    r2 = c.post("/analisis", files={"audio": ("audio_declarado.mp3", f)}, data={"modo": a.modo, "sincrono": "true", "declaracion": json.dumps(decl)}).json()
v = r2["resultado"]["validacion"]; print("Veredicto:", r2["resultado"]["veredicto"], "| consistente:", v["consistente"], "| HermiT: %d ejecuciones, %d ms" % (v["meta"]["ejecuciones_hermit"], v["meta"]["ms"]))
print("Conflicto minimo:", v["conflicto_minimo"]); print("Axioma violado:", [x["formal"] for x in v["axiomas_violados"]]); print("\n" + r2["resultado"]["explicacion"])
print("\nTraza del rechazo:"); mostrar_traza(r2["run_id"])
pausa()

titulo("MINUTO 4-5 · Auditoria: trazas, escalamientos, competencias y Shapley")
print("Escalamientos pendientes:", len(c.get("/escalamientos?limite=500").json()))
print("Competencias del Contract Net:", [(x["agent_id"], x["task_type"], x["competencia"], x["n_actualizaciones"]) for x in c.get("/competencias").json()])
sh = c.get("/auditoria/shapley")
if sh.status_code == 200:
    j = sh.json(); print("Shapley sobre %d casos etiquetados (suma phi = %.3f = v(N) = %.3f):" % (j["n"], j["suma_phi"], j["v_N"]))
    for x in j["agentes"]: print("  %-11s phi = %+.3f" % (x["agente"], x["shapley"]))
else:
    print(sh.json()["detail"], "\n-> Para la grafica con el corpus de prueba:  python main.py auditar rasgos.csv --grafica salidas/shapley.png")
