<div align="center">

# VOX-Forense

### Sistema multiagente híbrido para la detección explicable de voz sintética

Arantza García Vázquez · matrícula [198447] · [Ingeniería en Sistemas Computacionales]

Edna Samantha Cortés Carisoza · matrícula [198343] · [Ingeniería en Sistemas Computacionales]

Agentes Inteligentes · Otoño 2026 · Universidad Iberoamericana Puebla

<br>

<img src="docs/badges/python.svg" alt="Python"> <img src="docs/badges/java.svg" alt="Java"> <img src="docs/badges/fastapi.svg" alt="FastAPI"> <img src="docs/badges/postgresql.svg" alt="PostgreSQL"> <img src="docs/badges/ollama.svg" alt="Ollama"> <img src="docs/badges/owl2.svg" alt="OWL 2"> <img src="docs/badges/scikitlearn.svg" alt="scikit-learn"> <img src="docs/badges/numpy.svg" alt="NumPy"> <img src="docs/badges/pandas.svg" alt="pandas"> <img src="docs/badges/ffmpeg.svg" alt="FFmpeg"> <img src="docs/badges/docker.svg" alt="Docker"> <img src="docs/badges/pytest.svg" alt="pytest"> <img src="docs/badges/git.svg" alt="Git">

</div>

---

## Índice

1. [Resumen](#1-resumen)
2. [Problema e hipótesis](#2-problema-e-hipótesis)
3. [Arquitectura del sistema](#3-arquitectura-del-sistema)
4. [Base de conocimientos: persistir y razonar](#4-base-de-conocimientos-persistir-y-razonar)
5. [Cómo funciona una solicitud (T1–T7)](#5-cómo-funciona-una-solicitud-t1t7)
6. [Coordinación y herramientas](#6-coordinación-y-herramientas)
7. [Evaluación empírica](#7-evaluación-empírica)
8. [Auditoría con valores de Shapley](#8-auditoría-con-valores-de-shapley)
9. [Estructura del repositorio](#9-estructura-del-repositorio)
10. [Instalación y ejecución](#10-instalación-y-ejecución)
11. [Uso: API, CLI y demostración](#11-uso-api-cli-y-demostración)
12. [Pruebas](#12-pruebas)
13. [Limitaciones y consideraciones](#13-limitaciones-y-consideraciones)
14. [Correspondencia con la guía de la actividad](#14-correspondencia-con-la-guía-de-la-actividad)
15. [Referencias](#15-referencias)

---

## 1. Resumen

VOX-Forense determina si una grabación de voz es **auténtica** o **sintética**, y lo hace justificando cada decisión. En lugar de un clasificador que entrega una probabilidad, el sistema organiza a siete agentes especializados que miden rasgos acústicos, subastan las tareas entre sí, razonan sobre una **ontología OWL 2** con el razonador **HermiT** y emiten un veredicto que puede rastrearse hasta la regla que lo produjo.

El hilo conductor del proyecto es una idea única: **separar *qué se sabe* de *cómo se ejecuta***. El conocimiento del dominio (umbrales, clases, restricciones) vive en la ontología; el código solo coordina. Esto permite tres capacidades que un enfoque monolítico no ofrece:

| Capacidad | Mecanismo |
|---|---|
| **Decidir con justificación** | Cada veredicto cita las reglas ontológicas y los rasgos que lo sustentan |
| **Abstenerse** | Con evidencia insuficiente o contradictoria, escala a un analista humano (supuesto de mundo abierto) |
| **Rechazar** | Si la solicitud contradice un axioma, HermiT lo detecta y el Validador frena la operación con una justificación lógica |

---

## 2. Problema e hipótesis

**Dominio: audio forense.** Los vocoders neuronales (p. ej., HiFi-GAN) y la clonación de voz producen habla casi indistinguible de la humana. El costo del error es asimétrico: acusar de falsa una voz auténtica y aceptar una sintética tienen consecuencias legales distintas. Un dictamen forense debe, por tanto, **decidir, justificar y reconocer cuándo la evidencia no alcanza**.

**Limitaciones de los enfoques monolíticos:**

- *Clasificador único*: devuelve una probabilidad, sin razones ni abstención.
- *LLM aislado*: genera explicaciones plausibles sin respaldo (alucinación).

**Hipótesis.** Descomponer la tarea en agentes especializados y anclar sus conclusiones a una ontología formal reduce las afirmaciones sin respaldo y hace auditable cada decisión.

---

## 3. Arquitectura del sistema

El sistema es una **Crew jerárquica de CrewAI** (`Process.hierarchical`) con un coordinador y siete agentes. El veredicto lo emite **una herramienta determinista**; el LLM coordina y redacta, pero no decide.

```mermaid
flowchart TD
    U[Solicitud: audio + declaración opcional] --> API[FastAPI asíncrona]
    API --> M{{ Coordinador Forense<br/>manager jerárquico}}
    M --> T1[T1 Validador Semántico]
    T1 -- inconsistente --> R[RECHAZO V1<br/>se bloquean las demás herramientas]
    T1 -- consistente --> T2[T2 Perceptor]
    T2 --> T3[T3 Licitador<br/>Contract Net]
    T3 --> T4[T4 Espectral]
    T3 --> T5[T5 Prosódico]
    T4 --> T6[T6 Razonador<br/>HermiT en RAM]
    T5 --> T6
    T6 --> T7[T7 Explicador y Auditor]
    T7 --> V[SINTÉTICO · AUTÉNTICO · ESCALAR]
    T1 -. "@tool + httpx" .-> OA[/"POST /ontologia/validar"/]
    T6 -. "@tool + httpx" .-> OR[/"POST /ontologia/razonar"/]
    OA --- API
    OR --- API
    V --> PG[( PostgreSQL)]
    R --> PG
```

### Agentes (roles, metas y backstories)

| Agente | Meta | Backstory (idea fuerza) |
|---|---|---|
| **Coordinador Forense** (manager) | Delegar, validar coherencia, resolver o escalar | Prefiere escalar antes que cerrar un caso sin respaldo |
| **Validador Semántico** | Frenar solicitudes que contradicen la ontología | Verifica la coherencia antes de gastar un cálculo |
| **Perceptor** | Extraer y validar el vector de rasgos | Desconfía de todo número sin unidad ni rango |
| **Licitador Contract Net** | Ejecutar la subasta de tareas | Reporta exactamente lo que devuelve la herramienta |
| **Espectral** | Centroide, energía de alta frecuencia, curtosis mel | Reporta qué evidencia falta |
| **Prosódico** | Jitter y variabilidad de F0 | La ausencia de ruido natural también es indicio |
| **Razonador Ontológico** | Razonar con HermiT y arbitrar conflictos | No elige al azar entre reglas en conflicto |
| **Explicador y Auditor** | Cadena percepción → hechos → reglas → conclusión | No acepta una conclusión sin su regla |

> **Decisión de diseño.** CrewAI prohíbe herramientas en el agente manager. Por ello la subasta y la validación semántica se asignaron a agentes trabajadores dedicados (Licitador y Validador). Las tareas T4 y T5 **no tienen agente fijo**: las asigna la subasta de T3.

---

## 4. Base de conocimientos: persistir y razonar

Persistir y razonar son problemas distintos, y el sistema los resuelve con tecnologías distintas.

| Capa | Tecnología | Función |
|---|---|---|
| Persistencia transaccional | **PostgreSQL** (SQLAlchemy 2, psycopg3, JSONB) | Ejecuciones, casos, trazas inmutables, escalamientos, competencias |
| Conocimiento y reglas | **Ontología OWL 2 (TBox)** | Axiomas del dominio y coherencia lógica |
| Inferencia | **HermiT** (vía owlready2) | Clasificación y detección de inconsistencias, efímeras en RAM |
| Servicio | **FastAPI** asíncrona | Análisis, validación, razonamiento, trazas y Shapley |

### La ontología (TBox)

- **49 clases base** (55 en la versión calibrada), **21 propiedades de objeto** (10 pares de inversas) y **14 propiedades de datos**.
- Los **umbrales calibrados son restricciones de datos** dentro de clases equivalentes, p. ej. `JitterAnomalo ≡ Jitter ⊓ ∃valor.(double < 0.0226)`. Recalibrar modifica la ontología, no el código ni los agentes.
- **Restricciones que producen rechazo:** `AudioAutentico ⊥ AudioSintetico` y `AudioAutentico ⊑ ≤0 fueGeneradoPor.TecnicaDeSintesis`.
- **Supuesto de mundo abierto (OWA):** lo no afirmado no se asume falso. La falta de evidencia produce abstención, no un veredicto.
- Archivo: [`ontologia/Ontologia_Artefactos_Audio_calibrada.owx`](ontologia/Ontologia_Artefactos_Audio_calibrada.owx) (se regenera con `ontologia/gen_owx.py`).

### Reasoning Offloading efímero

El razonamiento se descarga a un proceso aparte y **no deja rastro persistente**:

1.  El TBox se lee **una vez** del `.owx` y queda como bytes en memoria; el archivo nunca se modifica.
2.  Por solicitud se crea un `World()` de owlready2 aislado (SQLite `:memory:`).
3.  Se generan al vuelo los individuos del caso (ABox, con `xsd:double`) y se cargan junto con el TBox.
4.  HermiT, en un proceso Java independiente, clasifica o detecta inconsistencias.
5.  Se leen las clases inferidas y el `World` se cierra y se descarta.

> **Límite declarado.** owlready2 entrega la ontología a HermiT mediante un archivo temporal. En Linux y Docker se dirige a `/dev/shm` (RAM); en Windows y macOS usa el directorio temporal del sistema operativo y se elimina al terminar. El endpoint `/salud` informa la ruta efectiva (`tmp_jvm`).

---

## 5. Cómo funciona una solicitud (T1–T7)

| Tarea | Agente | Qué ocurre |
|---|---|---|
| **T1**  | Validador Semántico | Descompone la declaración en aserciones atómicas y verifica su consistencia con HermiT. Si hay contradicción, calcula el **conjunto mínimo de conflicto** (eliminación de aserciones, 3–4 ejecuciones) y cita el axioma violado. Veredicto `RECHAZADO` (regla V1) y **todas las herramientas posteriores quedan bloqueadas por código**. |
| **T2**  | Perceptor | Extrae los rasgos del audio (Praat y librosa), los valida contra rangos y los convierte en hechos. |
| **T3**  | Licitador | Ejecuta Contract Net: anuncia (CFP), recibe propuestas (BID) y adjudica (AWARD). |
| **T4**  | Espectral (según subasta) | Evalúa centroide espectral, energía de alta frecuencia y curtosis mel contra las clases ontológicas. |
| **T5**  | Prosódico (según subasta) | Evalúa jitter y variabilidad de F0. |
| **T6**  | Razonador | HermiT clasifica el audio; luego la herramienta `decidir_veredicto` aplica la política. |
| **T7**  | Explicador | Construye la cadena percepción → hechos → reglas → conclusión y, si corresponde, escala a un analista. |

### Política de decisión

| Condición | Veredicto | Regla |
|---|---|---|
| ≥ 2 indicios y ≤ 1 rasgo de naturalidad | **SINTÉTICO** | R1 |
| ≥ 2 rasgos de naturalidad y 0 indicios | **AUTÉNTICO** | R2 |
| Sin evidencia | **ESCALAR** | R0 |
| Evidencia insuficiente o contradictoria | **ESCALAR** | R3 |
| Declaración inconsistente con la ontología | **RECHAZADO** | V1 |

---

## 6. Coordinación y herramientas

### Custom Tool (`@tool` con `httpx`)

Las herramientas `validar_consistencia_semantica` y `razonar_con_ontologia` conectan la Crew con la API asíncrona mediante `httpx` (`vox/cliente_api.py`). Con `VOX_API_URL` definida llaman a `POST /ontologia/validar` y `POST /ontologia/razonar`; sin ella se ejecutan en proceso. Cada llamada queda en la traza con el campo `transporte: http | local`.

Controles contra la alucinación:

- Los agentes intercambian **solo un `case_id`**. Las cifras permanecen en un almacén de casos (`vox/casos.py`), por lo que el LLM no puede alterarlas.
- El veredicto lo emite únicamente la herramienta determinista `decidir_veredicto`.
- Tras un rechazo, el código bloquea las herramientas posteriores (`_bloqueado`).

### Contract Net Protocol

- Fases: **CFP → BID → AWARD**.
- Utilidad: `U = 0.5·competencia + 0.3·cobertura + 0.2·(1 − carga)`, con umbral `τmin = 0.35`.
- Estados: `AWARDED`, `PARTIAL`, `NO_BIDS`, `ERROR`.
- La **competencia** de cada agente se persiste en PostgreSQL y se actualiza con media móvil exponencial (α = 0.1) cuando un analista resuelve un escalamiento.
- La herramienta corrige los nombres de rasgos inventados por un LLM y registra el ajuste en `ajustes_de_entrada`.

### Suplencia: robustez ante LLM locales

Los modelos locales pequeños pueden responder sin delegar, inventar hallazgos o tardar demasiado. Tras ejecutar la Crew, el sistema revisa qué etapas T1–T7 no dejaron resultado y **las completa con las mismas herramientas deterministas**. Nada se oculta:

- Cada etapa suplida queda como evento `SUPLENCIA` en la traza.
- El resultado incluye `crew_estado`: `completa`, `con_suplencia`, `tiempo_agotado+suplencia` o `error+suplencia`.
- `VOX_CREW_TIMEOUT_S` (300 s por defecto) limita el tiempo de la Crew.

### Trazabilidad

Cada acción genera un evento inmutable con `run_id`, `seq`, `parent_seq`, `ts`, agente y tarea: `TASK_START/END`, `TOOL_CALL/RESULT`, `DELEGATION`, `CFP/BID/AWARD`, `RULE_FIRED`, `CONFLICT/CONFLICT_RESOLVED`, `ESCALATION`, `REJECTION`, `VERDICT`, `SUPLENCIA`, `ERROR`. Se persisten en la tabla `eventos_traza`.

---

## 7. Evaluación empírica

### Protocolo

- **Corpus:** 963 audios de 10 s (502 auténticos y 461 sintéticos), 21 voces.
- **División por voz, no por fragmento:** 15 voces de entrenamiento (690 audios) y 6 de prueba (273 audios), para evitar fuga entre fragmentos del mismo hablante. Semilla 42.
- **Calibración solo con entrenamiento:** 3 rasgos por agente seleccionados por AUC; zonas de indicio y naturalidad por cuantil 0.20.
- **Controles:** 100 divisiones aleatorias por voz y regresión logística como referencia sin abstención.

### Resultados (273 audios de prueba)

| Métrica | Valor |
|---|---|
| Veredictos emitidos | 120 |
| **Aciertos entre los emitidos** | **118 (98.3 %)** |
| Errores | 2 (0.7 %) |
| Escalados a un analista | 153 (56.0 %) |
| Exactitud global (escalar no cuenta como acierto) | 0.432 |
| 100 divisiones por voz: exactitud en resueltos | 0.950 ± 0.034 |
| Umbrales iniciales sin calibrar | 89 % escalados |
| Regresión logística (sin abstención) | 0.923 |
| Rechazo semántico | 5 de 5 inconsistencias de prueba detectadas |

**Lectura.** El sistema cede cobertura por fiabilidad: casi nunca se equivoca cuando decide, y escala cuando la evidencia no alcanza. La regresión logística lo supera en exactitud global, pero no explica, no se abstiene y no rechaza solicitudes incoherentes.

### Tiempos (CRUD frente a inferencia semántica)

| Operación | Tiempo |
|---|---|
| CRUD en PostgreSQL: insertar / leer / actualizar | ≈ 3.2 / 1.5 / 2.3 ms |
| HermiT efímero, un audio | ≈ 0.96 s |
| HermiT en lote | ≈ 81 ms por audio |
| Rechazo con justificación (4 ejecuciones de HermiT) | ≈ 2 a 5 s |

La inferencia cuesta de 300 a 600 veces más que una operación CRUD: es el precio de la justificación lógica, y se amortiza por lotes. Los tiempos dependen del equipo.

---

## 8. Auditoría con valores de Shapley

Se modela un juego cooperativo con **5 jugadores y 32 coaliciones**:

`u(S) = 0.40·Exactitud + 0.20·Cobertura + 0.15·Consistencia + 0.20·Explicabilidad − 0.05·Latencia`, con `v(S) = u(S) − u(∅)`.

| Agente | φᵢ |
|---|---|
| Perceptor | 0.249 |
| Razonador | 0.158 |
| Explicador | 0.090 |
| Espectral | 0.047 |
| Prosódico | **−0.011** |
| **Σφ = v(N)** | **0.532** |

- El orden fue idéntico en 100 de 100 divisiones y en 200 perturbaciones de ±25 % en los pesos.
- **Hallazgo:** el aporte del Prosódico es nulo o ligeramente negativo (−0.006 ± 0.009 en 100 divisiones). Al retirarlo, la exactitud global sube de 0.432 a 0.729. La variabilidad de F0 resultó **mayor** en lo sintético, contra la hipótesis inicial.
- La latencia usa costos nominales de diseño.

Generación: `python main.py auditar datos/rasgos.csv --grafica salidas/shapley.png`.

---

## 9. Estructura del repositorio

```text
vox_forense/
├── main.py                  # CLI: analizar · lote · auditar · calibrar · servir
├── vox/
│   ├── api.py               # FastAPI asíncrona (endpoints, validación, límites)
│   ├── agentes.py           # Roles, metas y backstories (CrewAI)
│   ├── tareas.py            # Tareas T1–T7
│   ├── crew.py              # Ensamblado jerárquico, tope de tiempo y suplencia
│   ├── herramientas.py      # Funciones y @tool (con puente httpx)
│   ├── cliente_api.py       # Cliente httpx hacia la API
│   ├── ontologia.py         # Razonamiento efímero (HermiT) y validación semántica
│   ├── contract_net.py      # Protocolo Contract Net
│   ├── politica.py          # Reglas R0–R3
│   ├── casos.py             # Almacén de casos (el LLM solo ve el case_id)
│   ├── trazador.py          # Eventos de traza inmutables
│   ├── db.py · servicio.py  # Modelo relacional y persistencia
│   ├── auditoria.py         # Valores de Shapley
│   ├── calibracion.py · features.py   # Calibración y extracción de rasgos
│   └── llm_simulado.py      # LLM determinista para ensayo y pruebas
├── ontologia/               # .owx calibrada y generador
├── config/                  # Umbrales calibrados
├── sql/esquema.sql          # Esquema de PostgreSQL
├── demo/demo.py             # Demostración en vivo (5 min)
├── evaluacion/              # Análisis de resultados
├── datos/                   # Rasgos de ejemplo
├── tests/                   # Pruebas unitarias y de integración
├── Dockerfile · docker-compose.yml · requirements.txt · .env.example
```

---

## 10. Instalación y ejecución

### Requisitos

| Componente | Versión |
|---|---|
| Python | 3.10 a 3.13 (recomendado **3.12**; CrewAI no es compatible con 3.14) |
| Java (JRE) | 8 o superior (HermiT); se probó con 8 y 17 |
| ffmpeg | En el PATH |
| PostgreSQL | 16 (con Docker) |
| Ollama | Para el modo `crew` con un LLM local (opcional) |

### Windows, macOS y Linux

```powershell
# 1. Entorno virtual
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # en Linux/macOS: python3.12 -m venv .venv && source .venv/bin/activate
pip install --default-timeout=300 -r requirements.txt

# 2. Java y ffmpeg (Windows)
winget install EclipseAdoptium.Temurin.17.JRE
winget install Gyan.FFmpeg            # abrir una terminal nueva después

# 3. PostgreSQL
docker compose up -d db

# 4. Configuración
copy .env.example .env                # cp .env.example .env en Linux/macOS

# 5. API
python main.py servir                 # http://127.0.0.1:8000
```

### Variables de entorno (`.env`)

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `DATABASE_URL` | Conexión a PostgreSQL | `postgresql+psycopg://vox:vox@localhost:5432/vox` |
| `VOX_API_URL` | Si se define, las `@tool` llaman a la API por HTTP (`httpx`) | vacío (en proceso) |
| `VOX_LLM` | Modelo de la Crew: `ollama/llama3.1:8b`, `simulado`, u otro de CrewAI | `ollama/llama3.1:8b` |
| `OLLAMA_BASE_URL` | Servidor de Ollama | `http://localhost:11434` |
| `VOX_CREW_TIMEOUT_S` | Tope de tiempo de la Crew antes de la suplencia | `300` |
| `VOX_API_KEY` | Si se define, la API exige la cabecera `X-API-Key` | vacío |
| `VOX_MAX_MB` | Tamaño máximo de audio (MB) | `25` |

### LLM local con Ollama (sin clave de API)

```powershell
ollama pull llama3.1:8b
$env:OLLAMA_CONTEXT_LENGTH = "16384"     # recomendado: el contexto por defecto (4096) trunca los prompts de la Crew
ollama serve
```

Para ensayar la orquestación sin modelo, use `VOX_LLM=simulado`: ejecuta la Crew real de CrewAI con un guion determinista. No es un modelo de lenguaje.

### Docker

```bash
docker compose up --build             # API en :8000 y PostgreSQL en :5432
```

El `docker-compose.yml` monta `/tmp` como `tmpfs` (RAM) para el archivo temporal de HermiT y apunta a Ollama en `host.docker.internal:11434`.

---

## 11. Uso: API, CLI y demostración

### Endpoints

| Método y ruta | Descripción |
|---|---|
| `GET /salud` | Estado de PostgreSQL, Java, ffmpeg, CrewAI y versión de la ontología |
| `GET /ontologia/umbrales` | Reglas leídas del `.owx` |
| `POST /ontologia/razonar` | HermiT efímero sobre un conjunto de rasgos |
| `POST /ontologia/validar` | Validación semántica con conflicto mínimo y axioma violado |
| `POST /analisis` | Sube un audio (`modo=directo\|crew`, `sincrono=true`, `declaracion` opcional) |
| `POST /analisis/rasgos` | Igual, a partir de rasgos ya extraídos |
| `GET /analisis/{run_id}` y `/traza` | Resultado y log de eventos |
| `GET /casos` · `GET /casos/{id}` | Casos guardados |
| `GET /escalamientos` · `POST /escalamientos/{id}/resolver` | Cola de analistas; al resolver se actualiza la competencia |
| `GET /competencias` | Competencia de cada agente |
| `GET /auditoria/shapley` | Valores de Shapley |

La documentación interactiva se genera en **`http://127.0.0.1:8000/docs`**.

```bash
curl.exe http://127.0.0.1:8000/salud
curl.exe -F audio=@audio.wav -F modo=crew -F sincrono=true http://127.0.0.1:8000/analisis
# Solicitud inconsistente: auténtico + generado con un vocoder → RECHAZADO
curl.exe -X POST http://127.0.0.1:8000/ontologia/validar -H "content-type: application/json" \
     -d "{\"declaracion\": {\"etiquetas\": [\"AUTENTICO\", \"SINTETICO\"]}}"
```

### CLI

```bash
python main.py analizar audio.wav --modo directo            # sin LLM
python main.py analizar audio.wav --modo crew               # con la Crew
python main.py lote --carpeta fragmentos                    # varios audios
python main.py calibrar datos/rasgos.csv                    # recalcula los umbrales
python main.py auditar datos/rasgos.csv --grafica salidas/shapley.png
python main.py servir --host 127.0.0.1 --puerto 8000
```

### Demostración (5 minutos)

```powershell
python demo/demo.py --audio "fragmentos\falso1.wav" --modo crew
```

| Minuto | Contenido |
|---|---|
| 1 | Entorno: PostgreSQL, Java, ffmpeg, CrewAI y reglas leídas del `.owx` |
| 2 | Caso exitoso: log de la Crew, `@tool` por HTTP, subasta, veredicto y fila en PostgreSQL |
| 3 | Rechazo semántico: HermiT detecta la inconsistencia y el Validador frena la operación |
| 4–5 | Trazas, escalamientos, competencias y gráfica de Shapley |

### Inspección en PostgreSQL

```powershell
docker compose exec db psql -U vox -d vox
```

```sql
select case_id, veredicto, regla, n_indicios, n_naturalidad from casos order by creado desc limit 5;
select evento, count(*) from eventos_traza group by evento order by 2 desc;
select * from competencias;
select case_id, estado from escalamientos order by creado desc limit 5;
```

---

## 12. Pruebas

```bash
pytest tests
```

Las pruebas cubren:

- umbrales leídos de la ontología y equivalencia entre HermiT y la lectura directa de umbrales;
- política de decisión, Contract Net y eficiencia de Shapley (Σφ = v(N));
- Crew jerárquica real de CrewAI con el LLM simulado (flujo normal, rechazo, suplencia y tiempo agotado);
- Validador Semántico con cuatro inconsistencias, conflicto mínimo y bloqueo de herramientas;
- API sobre **PostgreSQL real**: persistencia de casos, trazas, escalamientos y competencias, subida y borrado de audio, autenticación y puente `httpx`.

Las pruebas de la API requieren PostgreSQL disponible; sin él se omiten.

---

## 13. Limitaciones y consideraciones

- **Generalización.** La evaluación usa 6 voces de prueba. Los rasgos espectrales podrían reflejar diferencias de grabación o códec, no solo de síntesis. Los umbrales provienen de un corpus propio.
- **Alcance del validador.** Comprueba la **coherencia lógica** de lo declarado, no su veracidad.
- **LLM local.** Con `llama3.1:8b` se observó que, con contexto limitado, el coordinador puede no delegar o inventar hallazgos. La suplencia lo contiene sin ocultarlo. La corrida con el LLM simulado verifica la orquestación, no la calidad de un modelo.
- **Concurrencia.** La API serializa HermiT con un candado. Con varios usuarios simultáneos se forma fila.
- **RAM efímera.** El archivo temporal de HermiT reside en RAM solo en Linux y Docker.
- **Latencia de Shapley.** Usa costos nominales de diseño.
- **Privacidad.** El audio no se conserva: se borra tras procesarlo y solo se guardan rasgos, resultados y trazas.

### Solución de problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `pip` no encuentra CrewAI | Python 3.14: use 3.12 |
| `password authentication failed for user "vox"` | Otro PostgreSQL ocupa el puerto 5432. Mapee Docker a otro puerto (`127.0.0.1:5544:5432`) y ajuste `DATABASE_URL` |
| `/salud` sin `java` | Instale un JRE y abra una terminal nueva |
| La Crew no termina | Aumente `OLLAMA_CONTEXT_LENGTH`, use un modelo de 7–8B o más, o `VOX_LLM=simulado` |

---

## 14. Correspondencia con la guía de la actividad

| Requisito | Dónde |
|---|---|
| CrewAI (roles, metas, backstories, proceso jerárquico) | `vox/agentes.py`, `vox/tareas.py`, `vox/crew.py` |
| `@tool` con `httpx` hacia la API asíncrona | `vox/herramientas.py`, `vox/cliente_api.py` |
| FastAPI asíncrona | `vox/api.py` |
| Base de datos relacional | `vox/db.py`, `sql/esquema.sql` |
| Ontología OWL 2 (TBox) | `ontologia/Ontologia_Artefactos_Audio_calibrada.owx` |
| Evaluación efímera en RAM con HermiT | `vox/ontologia.py` |
| Detección y rechazo semántico | `validar_declaracion`, `POST /ontologia/validar` |
| Contract Net | `vox/contract_net.py` |
| Shapley | `vox/auditoria.py`, `GET /auditoria/shapley` |
| Trazas auditables | `vox/trazador.py`, tabla `eventos_traza` |
| Pruebas unitarias | `tests/` |

---

## 15. Referencias

1. J. Kong, J. Kim y J. Bae, "HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis," *NeurIPS*, vol. 33, 2020.
2. Z. Ji *et al.*, "Survey of hallucination in natural language generation," *ACM Comput. Surv.*, vol. 55, no. 12, 2023.
3. M. Wooldridge, *An Introduction to MultiAgent Systems*, 2.ª ed. Wiley, 2009.
4. R. G. Smith, "The Contract Net Protocol," *IEEE Trans. Comput.*, vol. C-29, no. 12, pp. 1104–1113, 1980.
5. W3C, "OWL 2 Web Ontology Language Document Overview (Second Edition)," 2012.
6. T. R. Gruber, "A translation approach to portable ontology specifications," *Knowledge Acquisition*, vol. 5, no. 2, 1993.
7. B. Glimm *et al.*, "HermiT: An OWL 2 reasoner," *J. Autom. Reasoning*, vol. 53, no. 3, 2014.
8. J.-B. Lamy, "Owlready: Ontology-oriented programming in Python…," *Artif. Intell. Med.*, vol. 80, 2017.
9. L. S. Shapley, "A value for n-person games," en *Contributions to the Theory of Games II*, Princeton Univ. Press, 1953.
10. P. Lewis *et al.*, "Retrieval-augmented generation for knowledge-intensive NLP tasks," *NeurIPS*, vol. 33, 2020.
11. Documentación de [CrewAI](https://docs.crewai.com), [FastAPI](https://fastapi.tiangolo.com), [PostgreSQL 16](https://www.postgresql.org/docs/16/), [HTTPX](https://www.python-httpx.org) y [Ollama](https://docs.ollama.com).

---

<div align="center">

**VOX-Forense** · Agentes Inteligentes · Universidad Iberoamericana Puebla · 2026
*De la confianza ciega a la decisión auditable.*

</div>
