# -*- coding: utf-8 -*-
"""API REST (FastAPI) de VOX-Forense.  Arranque:  uvicorn vox.api:app --host 0.0.0.0 --port 8000
 - El audio se recibe, se procesa y se BORRA; en Postgres solo quedan rasgos, resultados y trazas.
 - La ontologia se evalua de forma efimera en RAM (World owlready2 en memoria + HermiT) en cada solicitud."""
import os, shutil, tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from . import __version__, ontologia as O, servicio as S
from .config import API_KEY, MAX_BYTES_AUDIO, cargar_umbrales
from .db import Caso, Competencia, Ejecucion, Escalamiento, EventoTraza, db_disponible, iniciar_db, sesion


@asynccontextmanager
async def lifespan(app):
    app.state.tmp = O.configurar_tmp_ram()                 # JVM de HermiT -> tmpfs (RAM) si existe
    iniciar_db(); S.conectar_competencias(); O.tbox()      # tablas, competencias persistentes y TBox en RAM
    yield


app = FastAPI(title="VOX-Forense", version=__version__, lifespan=lifespan,
              description="Deteccion explicable de voz sintetica: CrewAI + Contract Net + ontologia OWL (HermiT) + Shapley.")


def autenticar(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY: raise HTTPException(401, "X-API-Key invalida o ausente")


class RasgoDeclarado(BaseModel):
    rasgo: str = Field(description="p. ej. jitter, spectral_centroid")
    valor: float
    clase: Literal["natural", "anomalo"] | None = Field(default=None, description="clase que la solicitud AFIRMA para ese rasgo")


class Declaracion(BaseModel):
    """Lo que la solicitud AFIRMA sobre el audio. El Validador Semantico lo contrasta con los axiomas de la ontologia (HermiT, en RAM)."""
    etiquetas: list[Literal["AUTENTICO", "SINTETICO"]] = []
    hablante: str | None = None
    hablantes_adicionales: list[str] = []
    generado_por: list[Literal["VocoderGAN", "SintesisTextoAVoz", "ClonacionDeVoz", "ConversionDeVoz"]] = []
    imita_a: str | None = None
    rasgos_declarados: list[RasgoDeclarado] = []


class SolicitudValidacion(BaseModel):
    declaracion: Declaracion


class Rasgos(BaseModel):
    """Vector de caracteristicas ya extraido (mismos nombres que rasgos.csv). Los no informados se tratan como ausentes."""
    nombre: str = Field(default="manual", max_length=120)
    jitter: float | None = None; shimmer: float | None = None; f0_var: float | None = None
    spectral_flatness: float | None = None; spectral_centroid: float | None = None; hf_energy: float | None = None
    plateau_ratio: float | None = None; peak_count: float | None = None; mel_kurt: float | None = None
    mel_mean: float | None = None; mel_std: float | None = None; mel_skew: float | None = None; duration_s: float | None = None
    declaracion: Declaracion | None = None


class Resolucion(BaseModel):
    etiqueta: Literal["real", "fake"]
    comentario: str | None = None


# ------------------------------------------------------------------ salud y ontologia
@app.get("/salud")
def salud():
    import shutil as sh, subprocess
    try: java = next(l for l in subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10).stderr.splitlines() if "version" in l)
    except Exception: java = None                         # noqa: BLE001
    try: import crewai; cw = crewai.__version__
    except Exception: cw = None                           # noqa: BLE001
    return dict(estado="ok" if db_disponible() and java else "degradado", version=__version__, postgres=db_disponible(), java=java,
                ffmpeg=bool(sh.which("ffmpeg")), crewai=cw, ontologia=O.version_ontologia(), tmp_jvm=tempfile.gettempdir())


@app.get("/ontologia/umbrales", dependencies=[Depends(autenticar)])
def umbrales():
    """Reglas vigentes, leidas del propio .owx (no de una copia en codigo)."""
    cfg = cargar_umbrales()
    return dict(version_ontologia=O.version_ontologia(), meta=cfg["meta"],
                reglas=[O.regla_de(r["feature"], cfg) | {"auc": r["auc"]} for ag in cfg["reglas"] for r in cfg["reglas"][ag]])


@app.post("/ontologia/razonar", dependencies=[Depends(autenticar)])
async def razonar(r: Rasgos):
    """Razonamiento efimero: inserta el vector como individuos en un World en RAM, ejecuta HermiT y descarta todo. No persiste nada."""
    f = {k: v for k, v in r.model_dump().items() if k not in ("nombre", "duration_s", "declaracion") and v is not None}
    if not f: raise HTTPException(422, "Indica al menos un rasgo")
    res, meta = await run_in_threadpool(O.razonar_efimero, {r.nombre: f})
    return dict(resultado=res[r.nombre], meta=meta)


@app.post("/ontologia/validar", dependencies=[Depends(autenticar)])
async def validar(s: SolicitudValidacion):
    """Validador semantico: HermiT (RAM, efimero) comprueba que lo declarado sea coherente con los axiomas. Si no, devuelve el conflicto
    minimo de aserciones y el axioma violado como justificacion logica. No persiste nada."""
    try: return await run_in_threadpool(O.validar_declaracion, s.declaracion.model_dump())
    except ValueError as e: raise HTTPException(422, str(e))


# ------------------------------------------------------------------ analisis
def _estado(run_id):
    with sesion() as s:
        ej = s.get(Ejecucion, run_id)
        if not ej: raise HTTPException(404, "run_id no encontrado")
        out = dict(run_id=ej.run_id, case_id=ej.case_id, modo=ej.modo, estado=ej.estado, error=ej.error, nombre_original=ej.nombre_original)
        caso = s.get(Caso, ej.case_id)
        if caso:
            out["resultado"] = dict(veredicto=caso.veredicto, regla=caso.regla, motivo=caso.motivo, n_indicios=caso.n_indicios,
                                    n_naturalidad=caso.n_naturalidad, contradictoria=caso.contradictoria, explicacion=caso.explicacion,
                                    clases_audio=(caso.ontologia or {}).get("clases_audio"), tecnicas_sospechadas=(caso.ontologia or {}).get("tecnicas"),
                                    validacion=(caso.ontologia or {}).get("validacion"))
        return out


@app.post("/analisis", status_code=202, dependencies=[Depends(autenticar)])
async def analizar(bg: BackgroundTasks, audio: UploadFile = File(...), modo: Literal["directo", "crew"] = Form("directo"),
                   sincrono: bool = Form(False), declaracion: str | None = Form(None, description="JSON opcional con la Declaracion (ver /docs)")):
    """Sube un audio. Modo `directo` (sin LLM) o `crew` (CrewAI con LLM; requiere clave del proveedor). Devuelve un run_id;
    consulta el resultado en GET /analisis/{run_id} (o usa sincrono=true)."""
    try: decl = Declaracion.model_validate_json(declaracion).model_dump() if declaracion else None
    except Exception as e: raise HTTPException(422, "declaracion invalida: %s" % e)                 # noqa: BLE001
    ext = Path(audio.filename or "").suffix.lower()
    if ext not in S.EXT_PERMITIDAS: raise HTTPException(415, "Formato no admitido: " + ", ".join(S.EXT_PERMITIDAS))
    run_id = S.nuevo_run_id(); d = tempfile.mkdtemp(prefix="vox_"); ruta = os.path.join(d, run_id + ext); n = 0
    with open(ruta, "wb") as f:
        while chunk := await audio.read(1 << 20):
            n += len(chunk)
            if n > MAX_BYTES_AUDIO:
                f.close(); shutil.rmtree(d, ignore_errors=True); raise HTTPException(413, "Audio demasiado grande")
            f.write(chunk)
    S.registrar_ejecucion(run_id, os.path.basename(ruta), modo, audio.filename)

    def trabajo():
        try: S.ejecutar_audio(ruta, run_id, audio.filename, modo, declaracion=decl)
        finally: shutil.rmtree(d, ignore_errors=True)
    if sincrono: await run_in_threadpool(trabajo); return _estado(run_id)
    bg.add_task(trabajo)
    return dict(run_id=run_id, estado="en_proceso", consulta="/analisis/" + run_id)


@app.post("/analisis/rasgos", dependencies=[Depends(autenticar)])
async def analizar_rasgos(r: Rasgos):
    """Analiza un vector de rasgos ya extraido (sin audio, sin ffmpeg). Sincrono."""
    run_id = S.nuevo_run_id()
    d = r.model_dump(); decl = d.pop("declaracion", None)
    rasgos = {k: v for k, v in d.items() if k != "nombre" and v is not None}
    await run_in_threadpool(S.ejecutar_rasgos, rasgos, r.nombre, run_id, "directo", decl)
    return _estado(run_id)


@app.get("/analisis/{run_id}", dependencies=[Depends(autenticar)])
def consultar(run_id: str): return _estado(run_id)


@app.get("/analisis/{run_id}/traza", dependencies=[Depends(autenticar)])
def traza(run_id: str, evento: str | None = None):
    with sesion() as s:
        q = select(EventoTraza).where(EventoTraza.run_id == run_id).order_by(EventoTraza.seq)
        if evento: q = q.where(EventoTraza.evento == evento.upper())
        ev = [dict(seq=e.seq, parent_seq=e.parent_seq, ts=e.ts, evento=e.evento, agente=e.agente, tarea=e.tarea, payload=e.payload) for e in s.scalars(q)]
    if not ev: raise HTTPException(404, "sin eventos para ese run_id")
    return dict(run_id=run_id, n=len(ev), eventos=ev)


# ------------------------------------------------------------------ casos, escalamientos, competencias, Shapley
@app.get("/casos", dependencies=[Depends(autenticar)])
def casos(veredicto: str | None = None, limite: int = Query(50, le=500)):
    with sesion() as s:
        q = select(Caso).order_by(Caso.creado.desc()).limit(limite)
        if veredicto: q = q.where(Caso.veredicto == veredicto.upper())
        return [dict(case_id=c.case_id, nombre=c.nombre_original, veredicto=c.veredicto, regla=c.regla, etiqueta_real=c.etiqueta_real, creado=c.creado) for c in s.scalars(q)]


@app.get("/casos/{case_id}", dependencies=[Depends(autenticar)])
def caso(case_id: str):
    with sesion() as s:
        c = s.get(Caso, case_id)
        if not c: raise HTTPException(404, "caso no encontrado")
        return {k: getattr(c, k) for k in ("case_id", "run_id", "nombre_original", "rasgos", "faltantes", "hallazgos", "ontologia", "licitacion", "veredicto",
                                          "regla", "motivo", "n_indicios", "n_naturalidad", "contradictoria", "explicacion", "version_ontologia", "etiqueta_real", "creado")}


@app.get("/escalamientos", dependencies=[Depends(autenticar)])
def escalamientos(estado: Literal["pendiente", "resuelto"] = "pendiente", limite: int = Query(50, le=500)):
    with sesion() as s:
        return [dict(case_id=e.case_id, motivo=e.motivo, estado=e.estado, etiqueta_analista=e.etiqueta_analista, creado=e.creado)
                for e in s.scalars(select(Escalamiento).where(Escalamiento.estado == estado).order_by(Escalamiento.creado.desc()).limit(limite))]


@app.post("/escalamientos/{case_id}/resolver", dependencies=[Depends(autenticar)])
def resolver(case_id: str, r: Resolucion):
    """El analista fija la etiqueta real; se actualiza la competencia de los agentes (EMA) y el caso queda disponible para la auditoria Shapley."""
    try: return S.resolver_escalamiento(case_id, r.etiqueta, r.comentario)
    except KeyError: raise HTTPException(404, "caso no encontrado")


@app.get("/competencias", dependencies=[Depends(autenticar)])
def competencias():
    with sesion() as s:
        return [dict(agent_id=c.agent_id, task_type=c.task_type, competencia=round(c.competencia, 4), n_actualizaciones=c.n_actualizaciones)
                for c in s.scalars(select(Competencia).order_by(Competencia.agent_id, Competencia.task_type))]


@app.get("/auditoria/shapley", dependencies=[Depends(autenticar)])
def shapley(minimo: int = Query(20, ge=2)):
    """Valores de Shapley de los 5 agentes sobre los casos de la base que ya tienen etiqueta real (fijada por un analista)."""
    import pandas as pd
    from .auditoria import auditar_df
    with sesion() as s:
        filas = [dict(c.rasgos, label=c.etiqueta_real, file=c.case_id) for c in s.scalars(select(Caso).where(Caso.etiqueta_real.is_not(None)))]
    if len(filas) < minimo or len({f["label"] for f in filas}) < 2:
        raise HTTPException(409, "Se necesitan al menos %d casos etiquetados y de ambas clases (hay %d)." % (minimo, len(filas)))
    r = auditar_df(pd.DataFrame(filas))
    return dict(n=r["n"], suma_phi=r["suma_phi"], v_N=r["v_N"], exactitud_global=r["exactitud_global"], escalados=r["escalados"],
                errores=r["errores"], exactitud_resueltos=r["exactitud_resueltos"], agentes=r["tabla"].round(4).to_dict(orient="records"))
