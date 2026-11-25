# VOX-Forense · CrewAI + FastAPI + Postgres + ontología OWL (HermiT efímero en RAM)

Detección **explicable** de voz sintética. Un sistema multiagente (CrewAI) valida y razona sobre una ontología OWL 2 con HermiT, reparte el trabajo con Contract Net, guarda casos y trazas en PostgreSQL, expone todo por una API REST asíncrona y audita la contribución de cada agente con valores de Shapley.

```
solicitud ─► FastAPI (async) ─► Crew jerárquica (manager)
                                  T1 Validador Semántico ──┐ (HermiT: ¿lo declarado es coherente? si no → RECHAZO y se frena todo)
                                  T2 Perceptor · T3 Licitador (Contract Net) · T4 Espectral · T5 Prosódico
                                  T6 Razonador (HermiT en RAM) · T7 Explicador/Auditor
        @tool ──httpx──► POST /ontologia/validar · POST /ontologia/razonar   (Reasoning Offloading)
        Postgres: ejecuciones · casos · eventos_traza · escalamientos · competencias
```

## Qué pide la guía y dónde está
| Requisito | Dónde |
|---|---|
| **CrewAI** (roles, metas, backstories, `Process.hierarchical`) | `vox/agentes.py`, `vox/tareas.py`, `vox/crew.py` (manager + 7 agentes, T1–T7) |
| **@tool con httpx** que conecta la Crew con la API asíncrona | `vox/herramientas.py` + `vox/cliente_api.py` (con `VOX_API_URL`; sin ella, en proceso) |
| **FastAPI** asíncrona | `vox/api.py` (`async def` + `run_in_threadpool`) |
| **Base de datos relacional** (PostgreSQL; la guía admite "MongoDB Atlas **o** relacional") | `vox/db.py`, `sql/esquema.sql` |
| **Ontología OWL 2 (TBox)** | `ontologia/Ontologia_Artefactos_Audio_calibrada.owx` |
| **Evaluación efímera en RAM con HermiT** | `vox/ontologia.py` (`razonar_efimero`, `validar_declaracion`) |
| **Detección y rechazo semántico** (inconsistencia → Validador frena con justificación lógica) | `validar_declaracion` + tool `validar_consistencia_semantica` + `POST /ontologia/validar` |
| **Contract Net** | `vox/contract_net.py` (competencia persistida en Postgres, EMA) |
| **Shapley** (φᵢ, gráfica) | `vox/auditoria.py`, `python main.py auditar … --grafica`, `GET /auditoria/shapley` |
| **Trazas** auditables | `vox/trazador.py` → tabla `eventos_traza` |
| **Pruebas unitarias** | `tests/` (27 pruebas) |

## Evaluación efímera de la ontología en RAM
1. El TBox se lee **una vez** del `.owx` y queda como bytes en memoria; el archivo nunca se modifica.
2. Por solicitud se crea un `World()` de owlready2 sin archivo (SQLite `:memory:`), aislado.
3. Se generan al vuelo los individuos del caso (ABox) y se cargan con el TBox desde memoria.
4. HermiT (proceso Java aparte) clasifica o detecta inconsistencias; se leen las clases inferidas.
5. El `World` se cierra y se descarta: nada de la instancia queda persistido.

Detalle honesto: owlready2 le entrega la ontología a HermiT por un archivo temporal que borra al terminar. En Linux/Docker va a `/dev/shm` o a un `tmpfs` (RAM; el compose monta `/tmp` como tmpfs). En Windows/macOS usa el temporal del SO y se elimina enseguida.

## Rechazo semántico (Validador Semántico)
La solicitud puede **declarar** hechos sobre el audio (`etiquetas`, `hablante`, `generado_por`, `imita_a`, `rasgos_declarados`). La tarea **T1** los contrasta con los axiomas de la ontología usando HermiT:
* Consistente → el flujo sigue.
* **Inconsistente** → HermiT lo detecta; se calcula el **conflicto mínimo** de aserciones (re-ejecutando HermiT sin cada una) y se nombra el axioma violado, p. ej. *«audio auténtico» + «generado con VocoderGAN»* ⇒ `AudioAutentico ⊑ ≤0 fueGeneradoPor.TecnicaDeSintesis`. La operación **se frena**: las demás herramientas quedan bloqueadas por código (no depende de que el LLM obedezca), el caso se guarda con veredicto `RECHAZADO` (regla V1) y la traza registra `CONFLICT` y `REJECTION`.

Axiomas que se pueden violar en la demostración: autenticidad+síntesis (disjuntas), auténtico+técnica de síntesis, auténtico+imitación, auténtico con dos hablantes distintos, un mismo hablante como productor e imitado, y un rasgo cuyo valor cae en la zona anómala declarado como natural.

## Integración con CrewAI (notas)
* CrewAI (≥ 1.x) **prohíbe herramientas en el manager**; por eso la subasta la ejecuta el agente **Licitador** y la validación el **Validador Semántico**. T4 y T5 no llevan agente fijo: decide la licitación.
* El veredicto lo emite siempre la herramienta determinista `decidir_veredicto`; el texto del LLM nunca lo sustituye. Los agentes intercambian un `case_id`, no cifras.
* `VOX_LLM=ollama/llama3.1:8b` usa un modelo local con Ollama (sin clave; `ollama pull llama3.1:8b`, `OLLAMA_BASE_URL` por defecto `http://localhost:11434`). Los modelos muy chicos (3B) suelen fallar en el formato de herramientas de la crew jerárquica.
* `VOX_LLM=simulado` ejecuta la crew con un guion determinista (no es un LLM): ejerce delegación, herramientas, callbacks, trazas y persistencia sin clave de API.

## Instalación y arranque
Python 3.10–3.13 (CrewAI no soporta 3.14), Java y ffmpeg en el PATH.
```bash
pip install -r requirements.txt
docker compose up -d db                      # o cualquier PostgreSQL 16 (DATABASE_URL en .env)
export VOX_API_URL=http://127.0.0.1:8000     # (PowerShell: $env:VOX_API_URL="http://127.0.0.1:8000")
python main.py servir --puerto 8000          # http://localhost:8000/docs
```
Todo en contenedores: `docker compose up --build` (API :8000, Postgres :5432; `VOX_LLM=simulado` por defecto). Si ya habías creado las tablas con una versión anterior, borra el volumen (`docker compose down -v`).

## Uso de la API
```bash
curl localhost:8000/salud
curl -F audio=@audio.mp3 -F modo=crew -F sincrono=true localhost:8000/analisis
curl -F audio=@audio.mp3 -F modo=crew -F sincrono=true \
     -F 'declaracion={"etiquetas":["AUTENTICO"],"generado_por":["VocoderGAN"]}' localhost:8000/analisis      # → RECHAZADO
curl -X POST localhost:8000/ontologia/validar -H 'content-type: application/json' \
     -d '{"declaracion":{"etiquetas":["AUTENTICO","SINTETICO"]}}'
curl localhost:8000/analisis/r-xxxxxxxxxx/traza?evento=REJECTION
curl localhost:8000/escalamientos ; curl localhost:8000/competencias ; curl localhost:8000/auditoria/shapley
```
Sin `sincrono=true` el análisis responde 202 con un `run_id` y se consulta con `GET /analisis/{run_id}`. Con `VOX_API_KEY` definida, añade `-H 'X-API-Key: …'`.

## Demostración en vivo (5 min)
```bash
python demo/demo.py --audio fragmentos/audio_001.mp3 --modo crew      # con VOX_LLM real, o simulado
python main.py auditar rasgos.csv --grafica salidas/shapley.png        # gráfica de φᵢ (minutos 4–5)
```
| Min | Qué muestra `demo.py` |
|---|---|
| 1 | `/salud` (Postgres, Java, ffmpeg, CrewAI, versión de la ontología) y las reglas leídas del `.owx` |
| 2 | Caso válido por la Crew: log de trazas con cada `@tool` (marcada `[http]`), CFP/BID/AWARD, RULE_FIRED, VERDICT y la fila en Postgres |
| 3 | Declaración que viola un axioma → HermiT, conflicto mínimo, axioma, `RECHAZADO` y herramientas bloqueadas |
| 4–5 | Escalamientos, competencias del Contract Net y Shapley |

Inspección en Postgres (`docker compose exec db psql -U vox -d vox`):
```sql
select case_id, veredicto, regla, n_indicios, n_naturalidad from casos order by creado desc limit 5;
select seq, evento, agente, tarea, payload->>'tool' as tool from eventos_traza where run_id = 'r-…' order by seq;
select * from competencias;  select case_id, estado from escalamientos order by creado desc limit 5;
```

## Verificación hecha
* `pytest tests` → **27 pruebas**: umbrales leídos de la ontología; HermiT = lectura de umbrales (40/40); Contract Net; política; **crew jerárquica real de CrewAI con el LLM simulado** (flujo normal y flujo con rechazo); Validador Semántico (4 inconsistencias con conflicto mínimo y axioma); bloqueo de herramientas tras el rechazo; API contra **PostgreSQL 16 real** (casos, trazas, escalamientos, competencias, Shapley, subida de audio con borrado, 415, API key); **puente httpx** (la traza registra `transporte: http`).
* Servidor uvicorn real con `VOX_API_URL` apuntando a sí mismo: el log del servidor muestra a la Crew llamando `POST /ontologia/validar` y `POST /ontologia/razonar`.
* Con los 273 audios de prueba el flujo reproduce el reporte (118/120 aciertos, 153 escalados, Σφ = 0.532).
* Tiempos medidos (entorno de desarrollo): CRUD Postgres 1.5–3.2 ms; HermiT efímero ≈ 0.96 s por audio (≈ 81 ms/audio en lote de 40); validación con inconsistencia ≈ 2–4 s (3–4 ejecuciones de HermiT).

## Límites
* **No se ejecutó la crew con un LLM real** (sin clave): el guion determinista prueba la integración, no la calidad ni las decisiones de un modelo. Con un LLM real el manager podría delegar de otra forma; el bloqueo tras el rechazo y el veredicto están garantizados por código.
* **Docker no se construyó aquí** (sin daemon): `Dockerfile` y `docker-compose.yml` no están probados; la API y Postgres sí, fuera de contenedor.
* La API serializa el razonamiento HermiT con un candado; para más carga, más réplicas.
* Umbrales calibrados con un corpus propio (21 voces, 6 de prueba); revísalos con otros generadores o grabaciones (`python main.py calibrar`). El aporte del agente Prosódico en Shapley es nulo o ligeramente negativo.
* La declaración la aporta el solicitante: el sistema verifica su **coherencia lógica**, no su veracidad.

## Instalación y ejecución local con Ollama (Windows / macOS / Linux)
Requisitos: Python 3.10–3.13 (recomendado 3.12), Java 17+ (HermiT), ffmpeg, PostgreSQL 16 y Ollama.
```powershell
# 1. Entorno
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
winget install EclipseAdoptium.Temurin.17.JRE ; winget install Gyan.FFmpeg      # abre una terminal nueva despues
# 2. PostgreSQL (con Docker; o instala PostgreSQL y crea usuario/base vox)
docker compose up -d db
# 3. Ollama (https://ollama.com/download) y un modelo de 7-8B
ollama pull llama3.1:8b
# 4. Configuracion
copy .env.example .env          # ya apunta a Postgres local y a ollama/llama3.1:8b
# 5. Terminal 1: API       |   Terminal 2: demo
python main.py servir          #   python demo/demo.py --audio fragmentos\audio_001.mp3 --modo crew
python main.py auditar rasgos.csv --grafica salidas\shapley.png
```
`.env` se carga solo (no pisa variables ya definidas en el sistema). Para ensayar sin modelo: `VOX_LLM=simulado`.
Los modelos de 3B (`llama3.2`) suelen fallar con herramientas en una crew jerárquica; usa 7–8B o más. La integración con un modelo Ollama real no se ha probado: lo probado es la Crew con el LLM simulado.

## Robustez con LLM locales: suplencia
Un modelo local pequeño puede responder sin delegar ni llamar herramientas, o tardar demasiado. Tras la Crew, el sistema revisa qué etapas T1–T7 quedaron sin resultado en el almacén de casos y las completa con las mismas herramientas deterministas. Cada una queda registrada como evento `SUPLENCIA` en la traza, y el resultado trae `crew_estado` (`completa`, `con_suplencia`, `tiempo_agotado+suplencia`, `error+suplencia`). `VOX_CREW_TIMEOUT_S` (300 s por defecto) limita el tiempo de la Crew. El veredicto siempre lo emite `decidir_veredicto`, nunca el texto del LLM. Al presentar, declara cuando una ejecución necesitó suplencia.
