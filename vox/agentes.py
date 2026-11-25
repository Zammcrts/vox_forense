# -*- coding: utf-8 -*-
"""Agentes de CrewAI: manager + 7 trabajadores.
NOTA DE INTEGRACION: CrewAI (>= 1.x) prohibe que el agente manager de Process.hierarchical tenga herramientas
("Manager agent should not have tools"). Por eso `contract_net_allocate` NO va en el manager (como se esbozo en el diseno inicial):
la ejecuta un trabajador dedicado, el Licitador, al que el manager delega la subasta antes de repartir el analisis."""
from crewai import Agent, LLM
from .config import LLM_MODELO, OLLAMA_BASE_URL
from . import herramientas as H


def crear_llm(modelo=None):
    if (modelo or LLM_MODELO) == "simulado":                     # guion determinista, sin API (ver llm_simulado.py)
        from .llm_simulado import LLMSimulado
        return LLMSimulado(model="simulado")
    m = modelo or LLM_MODELO
    if m.startswith("ollama/"):                                  # modelo local (como en el cuaderno de clase), sin clave de API
        return LLM(model=m, base_url=OLLAMA_BASE_URL, api_key="ollama", temperature=0)
    return LLM(model=m, temperature=0)       # temperature=0: reproducibilidad para las replicas de Shapley


def crear_agentes(llm=None):
    llm = llm or crear_llm()
    comun = dict(llm=llm, verbose=True)
    manager = Agent(role="Coordinador Forense de Audio",
        goal="Descomponer cada caso de audio en tareas, delegarlas al especialista adecuado (la asignacion de analisis la decide el Licitador con Contract Net), validar la coherencia de los resultados y decidir si el caso se resuelve o se escala a un analista humano.",
        backstory="Dirigio durante anos un laboratorio de analisis forense de audio. Sabe que ningun especialista domina todas las familias de rasgos y que una mala asignacion cuesta tiempo y errores. Delega con evidencia (propuestas y competencia historica), exige resultados trazables y prefiere escalar a un humano antes que cerrar un caso sin respaldo.",
        tools=[], allow_delegation=True, max_iter=12, **comun)
    validador = Agent(role="Validador Semantico",
        goal="Comprobar con el razonador HermiT (en RAM, de forma efimera) que lo declarado sobre el audio sea logicamente coherente con la ontologia y, si contradice un axioma, frenar la operacion con una justificacion logica.",
        backstory="Ingeniero de conocimiento intransigente con la consistencia. Antes de gastar un solo calculo en un audio, verifica que la solicitud no viole los axiomas del dominio; si los viola, cita las aserciones en conflicto y el axioma exacto, y la operacion se detiene.",
        tools=[H.validar_consistencia_semantica], allow_delegation=False, **comun)
    perceptor = Agent(role="Analista de Percepcion de Audio",
        goal="Validar y normalizar el vector de caracteristicas del audio y convertirlo en hechos estructurados listos para la ontologia, senalando valores faltantes o fuera de rango.",
        backstory="Ingeniera de senales que paso de procesar espectrogramas a limpiar datos para sistemas de razonamiento. Desconfia de cualquier numero que no tenga unidad, rango esperado y procedencia.",
        tools=[H.extraer_y_validar_rasgos], allow_delegation=False, **comun)
    licitador = Agent(role="Licitador Contract Net",
        goal="Ejecutar el protocolo Contract Net (anuncio, propuestas, adjudicacion) para decidir que especialista analiza cada familia de rasgos, y reportar el estado de la licitacion (AWARDED, PARTIAL, NO_BIDS o ERROR).",
        backstory="Especialista en sistemas multiagente. No asigna por intuicion: llama a la herramienta contract_net_allocate y reporta exactamente lo que ella devuelve, incluidas las brechas de conocimiento.",
        tools=[H.contract_net_allocate], allow_delegation=False, **comun)
    espectral = Agent(role="Especialista en Artefactos Espectrales",
        goal="Identificar patrones espectrales asociados a sintesis (centroide, energia de alta frecuencia, curtosis del espectrograma mel) y vincularlos con clases de la ontologia con un grado de evidencia.",
        backstory="Fonetista computacional que estudio como los vocoders dejan huellas en el espectro que el oido no percibe. Reporta siempre que evidencia observo y cual le falta.",
        tools=[H.buscar_patrones_espectrales], allow_delegation=False, **comun)
    prosodico = Agent(role="Especialista en Prosodia y Voz",
        goal="Evaluar jitter, shimmer y variabilidad de F0 para detectar una prosodia antinatural y asociar los hallazgos con clases de la ontologia con un grado de evidencia.",
        backstory="Logopeda forense reconvertida al analisis automatico. Sabe que una voz real es imperfecta de forma caracteristica y que la ausencia de ese ruido natural tambien es un indicio.",
        tools=[H.buscar_patrones_prosodicos], allow_delegation=False, **comun)
    razonador = Agent(role="Razonador Ontologico",
        goal="Insertar la instancia en la ontologia (en RAM, de forma efimera, via la API), ejecutar el razonador OWL HermiT, arbitrar conflictos entre reglas y determinar si el conocimiento es suficiente para emitir un veredicto.",
        backstory="Ingeniero de conocimiento obsesionado con la consistencia logica. Ante dos reglas que se contradicen no elige al azar: aplica prioridades y evidencia, y deja constancia de la decision. El veredicto lo emite siempre la herramienta decidir_veredicto.",
        tools=[H.razonar_con_ontologia, H.decidir_veredicto], allow_delegation=False, **comun)
    explicador = Agent(role="Explicador y Auditor de Decisiones",
        goal="Producir una explicacion en lenguaje natural con la cadena percepcion -> hechos -> reglas -> conclusion, verificar que cada afirmacion tenga respaldo en la traza y escalar al analista humano cuando el conocimiento sea insuficiente.",
        backstory="Auditora que traduce razonamientos formales a informes entendibles para analistas. No acepta una conclusion sin poder senalar la regla y el hecho que la sostienen.",
        tools=[H.construir_explicacion, H.escalar_a_analista], allow_delegation=False, **comun)
    return manager, validador, perceptor, licitador, espectral, prosodico, razonador, explicador
