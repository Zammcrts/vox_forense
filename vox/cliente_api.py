# -*- coding: utf-8 -*-
"""Puente httpx entre las @tool de la Crew y la API asincrona de FastAPI (Reasoning Offloading).
Con VOX_API_URL definida, el razonamiento ontologico y la validacion semantica se piden a la API por HTTP; sin ella, las herramientas
usan las mismas funciones en proceso (modo CLI/offline). `FACTORY` permite inyectar un cliente httpx (las pruebas usan el TestClient)."""
import httpx
from . import config

FACTORY = None


def disponible():
    return FACTORY is not None or bool(config.API_URL)


def post(ruta, json, timeout=180):
    """POST sincrono con httpx; devuelve el JSON o lanza httpx.HTTPError. Incluye X-API-Key si esta configurada."""
    cab = {"X-API-Key": config.API_KEY} if config.API_KEY else {}
    if FACTORY is not None:                                # cliente inyectado (no se cierra)
        r = FACTORY().post(ruta, json=json, headers=cab)
    else:
        with httpx.Client(base_url=config.API_URL, headers=cab, timeout=timeout) as c: r = c.post(ruta, json=json)
    r.raise_for_status()
    return r.json()
