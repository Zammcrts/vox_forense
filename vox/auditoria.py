# -*- coding: utf-8 -*-
"""Auditoria de contribuciones con valores de Shapley (5 jugadores, 32 coaliciones) sobre un CSV de rasgos con etiqueta.
Usa los umbrales de la ontologia calibrada. La latencia usa costos nominales de diseno (no medidos)."""
import sys
import numpy as np, pandas as pd
from .config import RAIZ, NOMBRE_AGENTE, cargar_umbrales
sys.path.insert(0, str(RAIZ / "evaluacion"))
import analizar_resultados as A      # noqa: E402


def auditar(csv, solo_prueba=True):
    cfg = cargar_umbrales(); df = pd.read_csv(csv); df["speaker"] = df.apply(A.speaker, axis=1)
    if solo_prueba and cfg["meta"].get("hablantes_prueba"): df = df[df.speaker.isin(cfg["meta"]["hablantes_prueba"])]
    return auditar_df(df)


def auditar_df(df):
    """Ablacion de las 32 coaliciones y Shapley sobre un DataFrame con rasgos y `label` (real|fake)."""
    cfg = cargar_umbrales(); df = df.copy()
    for c in set(A.FEAT["E"]) | set(A.FEAT["Pr"]):
        if c not in df: df[c] = float("nan")
    df = df.reset_index(drop=True)
    C = A.counts(df, cfg["reglas"]); M, V, phi = A.run(df, C)
    full = M[frozenset(A.AGENTS)]
    tabla = pd.DataFrame([dict(agente=NOMBRE_AGENTE[a], shapley=phi[a],
                               v_sin_agente=V[frozenset(x for x in A.AGENTS if x != a)],
                               exactitud_sin_agente=M[frozenset(x for x in A.AGENTS if x != a)]["acc"]) for a in A.AGENTS]).sort_values("shapley", ascending=False)
    return dict(n=len(df), tabla=tabla, suma_phi=float(sum(phi.values())), v_N=float(V[frozenset(A.AGENTS)]),
                exactitud_global=float(full["acc"]), escalados=float(full["esc_rate"]), errores=float(full["wrong"] / len(df)),
                exactitud_resueltos=float(full["correct"] / max(1, full["correct"] + full["wrong"])))


def grafica(tabla, ruta, titulo="Valores de Shapley por agente"):
    """Barras horizontales de phi_i (el negativo se resalta). Devuelve la ruta del PNG."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = tabla.sort_values("shapley"); col = ["#c0392b" if v < 0 else "#2c6e91" for v in d.shapley]
    fig, ax = plt.subplots(figsize=(7, 3.6)); ax.barh(d.agente, d.shapley, color=col)
    for i, v in enumerate(d.shapley): ax.text(max(v, 0) + 0.005, i, "%+.3f" % v, va="center", ha="left", fontsize=9, color="#c0392b" if v < 0 else "#222")
    ax.axvline(0, color="#444", lw=.8); ax.set_xlabel("contribucion marginal φ"); ax.set_title(titulo + "  (Σφ = %.3f)" % d.shapley.sum())
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(ruta, dpi=160); plt.close(fig); return ruta
