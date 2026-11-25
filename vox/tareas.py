# -*- coding: utf-8 -*-
"""Tareas T1-T7. Los marcadores {audio_path} y {case_id} se rellenan con kickoff(inputs=...).
En Process.hierarchical el manager decide a quien delegar; cada tarea nombra al especialista esperado.
T4 y T5 no llevan agente fijo: la adjudicacion del Contract Net (T3) decide quien las ejecuta."""
import json
from crewai import Task
from .contract_net import subtareas_por_defecto


def crear_tareas(validador, perceptor, licitador, espectral, prosodico, razonador, explicador):
    plan = json.dumps(subtareas_por_defecto(), ensure_ascii=False)
    t1 = Task(description="T1 Validacion semantica. Delega al Validador Semantico ejecutar validar_consistencia_semantica para el case_id '{case_id}'. Si la herramienta responde que la solicitud es INCONSISTENTE (rechazado=true), la operacion se detiene: reporta la justificacion logica tal cual, y las herramientas posteriores quedaran bloqueadas.",
              expected_output="JSON con consistente (true/false), conflicto minimo, axiomas violados y justificacion logica.", agent=validador)
    t2 = Task(description="T2 Percepcion. Delega al Analista de Percepcion de Audio el uso de extraer_y_validar_rasgos con audio_path='{audio_path}'. Reporta el case_id, los rasgos faltantes y los avisos de rango. Si T1 rechazo la solicitud, no hay nada que percibir: reportalo. Usa SIEMPRE el case_id '{case_id}' en las tareas siguientes.",
              expected_output="JSON breve con case_id, numero de rasgos validos, faltantes y avisos (o el bloqueo por rechazo).", agent=perceptor, context=[t1])
    t3 = Task(description="T3 Licitacion. Delega al Licitador Contract Net la ejecucion de contract_net_allocate con subtasks_json=" + plan + " y deadline_ms=2000. Reporta el estado (AWARDED, PARTIAL, NO_BIDS o ERROR), los ganadores y las utilidades tal como los devolvio la herramienta.",
              expected_output="Resultado de la licitacion: status, ganadores por tipo de tarea, utilidades y brechas.", agent=licitador, context=[t1, t2])
    t4 = Task(description="T4 Analisis espectral. Delega al agente que gano SPECTRAL/PLATEAU_STATS en T3 (normalmente el Especialista en Artefactos Espectrales) la busqueda de patrones espectrales del case_id '{case_id}'.",
              expected_output="Hallazgos espectrales con clase ontologica y peso de evidencia, o la declaracion explicita de brecha.", context=[t2, t3])
    t5 = Task(description="T5 Analisis prosodico. Delega al agente que gano PROSODIC en T3 (normalmente el Especialista en Prosodia y Voz) la busqueda de patrones prosodicos del case_id '{case_id}'. Si la licitacion termino en PARTIAL o NO_BIDS, indicalo como brecha de conocimiento.",
              expected_output="Hallazgos prosodicos con clase ontologica y peso de evidencia, o la declaracion explicita de brecha.", context=[t2, t3])
    t6 = Task(description="T6 Razonamiento ontologico. Delega al Razonador Ontologico ejecutar razonar_con_ontologia y luego decidir_veredicto para el case_id '{case_id}'. Debe reportar tal cual el veredicto, la regla, los conteos y cualquier discrepancia con los especialistas, sin cambiar el veredicto de la herramienta.",
              expected_output="JSON con clases inferidas por la ontologia, conteos, discrepancias y veredicto (SINTETICO, AUTENTICO, ESCALAR o RECHAZADO) con su regla.", agent=razonador, context=[t4, t5])
    t7 = Task(description="T7 Explicacion y auditoria. Delega al Explicador y Auditor ejecutar construir_explicacion para el case_id '{case_id}'; si el veredicto es ESCALAR, tambien escalar_a_analista con el motivo. Redacta un informe breve en espanol que use SOLO hechos presentes en la traza.",
              expected_output="Informe en espanol: veredicto, cadena percepcion -> hechos -> reglas -> conclusion y limites. Si se escalo, el registro del escalamiento; si se rechazo, la justificacion logica.", agent=explicador, context=[t1, t2, t3, t4, t5, t6])
    return [t1, t2, t3, t4, t5, t6, t7]
