# -*- coding: utf-8 -*-
"""Rutas y configuracion compartida. Todo el conocimiento (umbrales) sale de la ontologia .owx."""
import json, os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
def _cargar_env():
    """Lee .env de la raiz del proyecto (sin dependencias) sin pisar variables ya definidas en el sistema."""
    f = RAIZ / ".env"
    if f.exists():
        for linea in f.read_text(encoding="utf-8").splitlines():
            linea = linea.split("#", 1)[0].strip()
            if "=" in linea:
                k, v = linea.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
_cargar_env()
RUTA_ONTOLOGIA = Path(os.getenv("VOX_ONTOLOGIA", RAIZ / "ontologia" / "Ontologia_Artefactos_Audio_calibrada.owx"))
RUTA_UMBRALES = RAIZ / "config" / "umbrales_calibrados.json"
RUTA_ESCALAMIENTOS = RAIZ / "salidas" / "escalamientos.jsonl"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CREW_TIMEOUT_S = float(os.getenv("VOX_CREW_TIMEOUT_S", "300"))   # tope de tiempo de la Crew; despues, el sistema completa por suplencia
LLM_MODELO = os.getenv("VOX_LLM", "ollama/llama3.1:8b")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://vox:vox@localhost:5432/vox")
API_URL = os.getenv("VOX_API_URL", "")                 # si se define, las @tool de la Crew llaman a la API por HTTP (httpx)
API_KEY = os.getenv("VOX_API_KEY", "")                 # si se define, la API exige la cabecera X-API-Key
MAX_BYTES_AUDIO = int(os.getenv("VOX_MAX_MB", "25")) * 1024 * 1024
MAX_DURACION_S = float(os.getenv("VOX_MAX_DURACION_S", "120"))

# Costos nominales de diseno (segundos). NO son mediciones; solo alimentan el termino de latencia de la utilidad.
LATENCIA_NOMINAL = {"P": 0.4, "E": 1.2, "Pr": 1.1, "R": 1.5, "X": 1.8}
NOMBRE_AGENTE = {"P": "Perceptor", "E": "Espectral", "Pr": "Prosódico", "R": "Razonador", "X": "Explicador"}

# Rasgos que cada agente puede evaluar (los que tengan regla calibrada se usan)
RASGOS_ESPECTRALES = ["spectral_flatness", "spectral_centroid", "hf_energy", "plateau_ratio", "peak_count",
                      "mel_mean", "mel_std", "mel_skew", "mel_kurt"]
RASGOS_PROSODICOS = ["jitter", "shimmer", "f0_var"]

# Rangos fisicamente plausibles para validar el vector de caracteristicas (agente Perceptor)
RANGOS = {"jitter": (0, 1), "shimmer": (0, 1), "f0_var": (0, 5), "hf_energy": (0, 1), "spectral_flatness": (0, 1),
          "spectral_centroid": (0, 8000), "plateau_ratio": (0, 1), "peak_count": (0, 64), "duration_s": (0.5, 3600)}


def cargar_umbrales(ruta=None):
    with open(ruta or RUTA_UMBRALES, encoding="utf-8") as f:
        return json.load(f)
