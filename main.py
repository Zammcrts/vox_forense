# -*- coding: utf-8 -*-
"""VOX-Forense - CLI.
  python main.py analizar AUDIO.mp3 [--modo directo|crew]   analiza un audio (crew requiere clave de API del LLM)
  python main.py lote --carpeta DIR | --csv rasgos.csv [--salida out.csv]   analisis en lote sin LLM
  python main.py auditar rasgos.csv [--todo]                valores de Shapley y ablacion de agentes
  python main.py calibrar rasgos.csv [--todo]               recalibra umbrales y regenera la ontologia .owx
  python main.py servir [--host 0.0.0.0 --puerto 8000]      API REST (FastAPI + Postgres; requiere DATABASE_URL)
  --modelo simulado                                         ejecuta la crew con un guion determinista (sin API de LLM)"""
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("analizar"); a.add_argument("audio"); a.add_argument("--modo", choices=["directo", "crew"], default="directo")
    a.add_argument("--modelo", default=None, help="modelo LLM (por defecto VOX_LLM)")
    a.add_argument("--declaracion", default=None, help='JSON con lo que se afirma del audio, p. ej. \'{"etiquetas":["AUTENTICO"],"generado_por":["VocoderGAN"]}\'')
    l = sp.add_parser("lote"); l.add_argument("--carpeta"); l.add_argument("--csv"); l.add_argument("--salida", default="salidas/resultados_lote.csv")
    u = sp.add_parser("auditar"); u.add_argument("csv"); u.add_argument("--todo", action="store_true", help="usar todos los audios y no solo los de prueba"); u.add_argument("--grafica", default=None, help="ruta del PNG con la grafica de phi")
    v = sp.add_parser("servir"); v.add_argument("--host", default="127.0.0.1"); v.add_argument("--puerto", type=int, default=8000)
    c = sp.add_parser("calibrar"); c.add_argument("csv"); c.add_argument("--todo", action="store_true")
    args = ap.parse_args()

    if args.cmd == "analizar":
        if args.modo == "crew":
            from vox.crew import analizar_con_crew
            from vox.agentes import crear_llm
            from vox import casos as K
            if args.declaracion: K.registrar(__import__('os').path.basename(args.audio), 'declaracion', declaracion=json.loads(args.declaracion))
            r = analizar_con_crew(args.audio, crear_llm(args.modelo))
            print("\n=== RESULTADO (herramienta decidir_veredicto) ===\nVeredicto:", r["veredicto"]); print(r["explicacion"] or "")
            print("\n=== INFORME DEL LLM ===\n" + r["informe_llm"])
        else:
            from vox.pipeline import analizar_directo
            r = analizar_directo(args.audio, json.loads(args.declaracion) if args.declaracion else None); print("Licitacion:", r.get("licitacion"), "| Veredicto:", r["veredicto"]); print(r.get("explicacion", r.get("error")))
    elif args.cmd == "lote":
        if not (args.carpeta or args.csv): sys.exit("Indica --carpeta o --csv")
        import os; os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
        from vox.pipeline import analizar_lote
        df = analizar_lote(args.carpeta, args.csv, args.salida)
        print(df.veredicto.value_counts().to_string())
        cols = [c for c in ("case_id", "file", "veredicto", "regla", "etiqueta_real") if c in df]
        print(df[cols].to_string(index=False))
        if "etiqueta_real" in df:
            etq = df[df.etiqueta_real.isin(["fake", "real"])]
            dec = etq[etq.veredicto.isin(["SINTETICO", "AUTENTICO"])]
            ok = ((dec.veredicto == "SINTETICO") == (dec.etiqueta_real == "fake")).sum()
            print(f"Con etiqueta: {len(etq)} | resueltos: {len(dec)} | aciertos entre resueltos: {ok}/{len(dec)} | escalados: {len(etq)-len(dec)}")
        print("Guardado en", args.salida)
    elif args.cmd == "auditar":
        from vox.auditoria import auditar
        r = auditar(args.csv, solo_prueba=not args.todo)
        print(f"Audios: {r['n']} | exactitud global {r['exactitud_global']:.3f} | escalados {r['escalados']:.3f} | errores {r['errores']:.3f} | exactitud en resueltos {r['exactitud_resueltos']:.3f}")
        print(r["tabla"].round(3).to_string(index=False)); print(f"Suma de phi = {r['suma_phi']:.3f}  v(N) = {r['v_N']:.3f}")
        if args.grafica:
            import os; from vox.auditoria import grafica; os.makedirs(os.path.dirname(args.grafica) or ".", exist_ok=True); print("Grafica:", grafica(r["tabla"], args.grafica))
    elif args.cmd == "servir":
        import uvicorn; uvicorn.run("vox.api:app", host=args.host, port=args.puerto)
    elif args.cmd == "calibrar":
        from vox.calibracion import calibrar
        cfg = calibrar(args.csv, todo=args.todo); print(json.dumps(cfg["meta"], ensure_ascii=False, indent=1))
        for ag in cfg["reglas"]:
            for r in cfg["reglas"][ag]: print(ag, r["feature"], "AUC", r["auc"], "t_indicio", round(r["t_ind"], 6), "t_natural", round(r["t_nat"], 6))


if __name__ == "__main__":
    main()
