# -*- coding: utf-8 -*-
"""Acceso a la ontologia OWL (.owx): umbrales leidos del propio archivo y razonamiento EFIMERO en RAM con HermiT.

 - umbrales(): lee del XML los axiomas  <Clase> == <ClaseBase> and valor(facet)  => el conocimiento vive en la ontologia.
 - clasificar_lote(): inserta cada audio y sus rasgos como individuos, ejecuta HermiT y devuelve las clases inferidas.
   Se usa mundo abierto (OWA): la ontologia infiere pertenencia a clases, el conteo de evidencia lo hace la politica."""
import hashlib, io, math, os, re, tempfile, threading, time
import xml.etree.ElementTree as ET
from functools import lru_cache
from .config import RUTA_ONTOLOGIA, cargar_umbrales

OWL = "{http://www.w3.org/2002/07/owl#}"
CLASES_AUDIO = ("AudioSospechoso", "AudioSospechosoDeVocoderGAN", "AudioConRasgosDeNaturalidad",
                "AudioConEvidenciaContradictoria", "AudioQueRequiereRevisionHumana")
_lock = threading.Lock()


def _seguro(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


@lru_cache(maxsize=4)
def umbrales(ruta=None):
    """Devuelve {clase_equivalente: (clase_base, facet, valor)} leyendo los axiomas EquivalentClasses del .owx."""
    ruta = ruta or str(RUTA_ONTOLOGIA)
    out = {}
    for eq in ET.parse(ruta).getroot().iter(OWL + "EquivalentClasses"):
        cl = [c for c in eq if c.tag == OWL + "Class"]
        inter = eq.find(OWL + "ObjectIntersectionOf")
        if len(cl) != 1 or inter is None: continue
        base = [c.get("IRI").lstrip("#") for c in inter if c.tag == OWL + "Class"]
        fr = inter.find(".//" + OWL + "FacetRestriction")
        if not base or fr is None: continue
        facets = {f.get("facet").split("#")[-1]: float(f.find(OWL + "Literal").text)
                  for f in inter.iter(OWL + "FacetRestriction")}
        out[cl[0].get("IRI").lstrip("#")] = (base[0], facets)
    return out


def regla_de(feature, cfg=None):
    """Regla de un rasgo: clases de indicio/naturalidad y sus umbrales, tomados de la ontologia."""
    cfg = cfg or cargar_umbrales()
    m = cfg["ontologia"][feature]; u = umbrales()
    return dict(feature=feature, base=m["base"], clase_indicio=m["indicio"], clase_naturalidad=m["naturalidad"],
                tecnica=m["tecnica"], umbral_indicio=u[m["indicio"]][1], umbral_naturalidad=u[m["naturalidad"]][1])


def _cumple(valor, facets):
    ok = True
    for k, v in facets.items():
        ok &= {"minExclusive": valor > v, "maxExclusive": valor < v, "minInclusive": valor >= v, "maxInclusive": valor <= v}[k]
    return ok


def evaluar_rasgo(feature, valor, cfg=None):
    """Evaluacion rapida (sin JVM) usando los umbrales de la ontologia. Devuelve 'indicio' | 'naturalidad' | 'sin_evidencia'."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)): return "sin_valor"
    r = regla_de(feature, cfg)
    if _cumple(valor, r["umbral_indicio"]): return "indicio"
    if _cumple(valor, r["umbral_naturalidad"]): return "naturalidad"
    return "sin_evidencia"


def _ancestros(ind):
    nombres = set()
    for c in ind.is_a:
        if hasattr(c, "ancestors"): nombres |= {a.name for a in c.ancestors() if hasattr(a, "name")}
    return nombres


def _aserciones(casos, cfg):
    """Axiomas de aserciones (individuos) en OWL/XML. Se usa xsd:double para que coincida con el rango de `valor`."""
    from xml.sax.saxutils import escape
    ind = lambda n: '<NamedIndividual IRI="#%s"/>' % n
    cls = lambda c: '<Class IRI="#%s"/>' % c
    xs, nombres = [], {}
    for cid, feats in casos.items():
        a = _seguro("audio_" + cid); nombres[cid] = (a, {}); grupo = [a]
        xs += ['<Declaration>%s</Declaration>' % ind(a), '<ClassAssertion>%s%s</ClassAssertion>' % (cls("Audio"), ind(a)),
               '<DataPropertyAssertion><DataProperty IRI="#identificadorDeAudio"/>%s<Literal datatypeIRI="http://www.w3.org/2001/XMLSchema#string">%s</Literal></DataPropertyAssertion>' % (ind(a), escape(cid))]
        for f, v in feats.items():
            if f not in cfg["ontologia"] or v is None or (isinstance(v, float) and math.isnan(v)): continue
            fi = _seguro("%s__%s" % (cid, f)); nombres[cid][1][f] = fi; grupo.append(fi)
            xs += ['<Declaration>%s</Declaration>' % ind(fi),
                   '<ClassAssertion>%s%s</ClassAssertion>' % (cls(cfg["ontologia"][f]["base"]), ind(fi)),
                   '<DataPropertyAssertion><DataProperty IRI="#valor"/>%s<Literal datatypeIRI="http://www.w3.org/2001/XMLSchema#double">%r</Literal></DataPropertyAssertion>' % (ind(fi), float(v)),
                   '<ObjectPropertyAssertion><ObjectProperty IRI="#tieneCaracteristica"/>%s%s</ObjectPropertyAssertion>' % (ind(a), ind(fi))]
        if len(grupo) > 1:       # distintos solo dentro de cada audio (evita el costo cuadratico en lotes grandes)
            xs.append('<DifferentIndividuals>%s</DifferentIndividuals>' % "".join(ind(n) for n in grupo))
    return "\n".join(xs), nombres


# ------------------------------------------------------------------ TBox en memoria (se lee una sola vez)
_tbox = {"ruta": None, "mtime": None, "bytes": None, "sha": None}


def tbox(ruta=None):
    """Axiomas de la ontologia como bytes en RAM. El archivo .owx nunca se modifica: cada razonamiento trabaja sobre una copia."""
    ruta = str(ruta or RUTA_ONTOLOGIA); mt = os.path.getmtime(ruta)
    if _tbox["ruta"] != ruta or _tbox["mtime"] != mt:
        b = open(ruta, "rb").read()
        _tbox.update(ruta=ruta, mtime=mt, bytes=b, sha=hashlib.sha256(b).hexdigest())
    return _tbox["bytes"]


def version_ontologia(ruta=None):
    tbox(ruta); return _tbox["sha"][:16]


def configurar_tmp_ram():
    """HermiT corre en una JVM aparte y owlready2 le pasa la ontologia por un archivo temporal que borra al terminar.
    En Linux/Docker lo dirigimos a /dev/shm (tmpfs = RAM) para que nada toque el disco. En Windows/macOS usa el temporal del SO."""
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        tempfile.tempdir = "/dev/shm"
    return tempfile.gettempdir()


def _aserciones(casos, cfg):
    """Axiomas de aserciones (individuos) en OWL/XML. Se usa xsd:double para que coincida con el rango de `valor`."""
    from xml.sax.saxutils import escape
    ind = lambda n: '<NamedIndividual IRI="#%s"/>' % n
    cls = lambda c: '<Class IRI="#%s"/>' % c
    xs, nombres = [], {}
    for cid, feats in casos.items():
        a = _seguro("audio_" + cid); nombres[cid] = (a, {}); grupo = [a]
        xs += ['<Declaration>%s</Declaration>' % ind(a), '<ClassAssertion>%s%s</ClassAssertion>' % (cls("Audio"), ind(a)),
               '<DataPropertyAssertion><DataProperty IRI="#identificadorDeAudio"/>%s<Literal datatypeIRI="http://www.w3.org/2001/XMLSchema#string">%s</Literal></DataPropertyAssertion>' % (ind(a), escape(cid))]
        for f, v in feats.items():
            if f not in cfg["ontologia"] or v is None or (isinstance(v, float) and math.isnan(v)): continue
            fi = _seguro("%s__%s" % (cid, f)); nombres[cid][1][f] = fi; grupo.append(fi)
            xs += ['<Declaration>%s</Declaration>' % ind(fi),
                   '<ClassAssertion>%s%s</ClassAssertion>' % (cls(cfg["ontologia"][f]["base"]), ind(fi)),
                   '<DataPropertyAssertion><DataProperty IRI="#valor"/>%s<Literal datatypeIRI="http://www.w3.org/2001/XMLSchema#double">%r</Literal></DataPropertyAssertion>' % (ind(fi), float(v)),
                   '<ObjectPropertyAssertion><ObjectProperty IRI="#tieneCaracteristica"/>%s%s</ObjectPropertyAssertion>' % (ind(a), ind(fi))]
        if len(grupo) > 1:       # distintos solo dentro de cada audio (evita el costo cuadratico en lotes grandes)
            xs.append('<DifferentIndividuals>%s</DifferentIndividuals>' % "".join(ind(n) for n in grupo))
    return "\n".join(xs), nombres


def razonar_efimero(casos, ruta=None):
    """Evaluacion EFIMERA de la ontologia en RAM.
       1) World() de owlready2 sin archivo -> almacen SQLite ':memory:' aislado (no hay estado compartido entre solicitudes).
       2) TBox (bytes cacheados) + ABox del caso (individuos generados al vuelo) se cargan desde memoria, sin escribir la ontologia.
       3) HermiT clasifica; se extraen las clases inferidas.  4) El World se cierra y se descarta: nada queda persistido.
    casos: {id: {rasgo: valor}}.  Devuelve (resultados, meta)."""
    from owlready2 import World, sync_reasoner_hermit, OwlReadyInconsistentOntologyError
    cfg = cargar_umbrales(); t0 = time.time()
    xml, nombres = _aserciones(casos, cfg)
    datos = tbox(ruta).decode("utf-8").replace("</Ontology>", xml + "\n</Ontology>").encode("utf-8")
    meta = dict(almacen="memoria", n_audios=len(casos), n_individuos=sum(1 + len(f) for _, f in nombres.values()),
                tmp_jvm=tempfile.gettempdir(), version_ontologia=version_ontologia(ruta))
    with _lock:
        w = World()                                              # RAM: SQLite ':memory:'
        try:
            o = w.get_ontology("https://example.org/ontologias/artefactos-audio").load(fileobj=io.BytesIO(datos))
            try:
                with o: sync_reasoner_hermit(w, infer_property_values=False, debug=0)
            except OwlReadyInconsistentOntologyError:
                meta["ms"] = round((time.time() - t0) * 1000)
                return {cid: dict(consistente=False) for cid in casos}, meta
            res = {}
            for cid, (an, fis) in nombres.items():
                a = o[an]; anc = _ancestros(a); rasgos = {}
                for f, fn in fis.items():
                    x = _ancestros(o[fn])
                    rasgos[f] = dict(clase_inferida=[n for n in (cfg["ontologia"][f]["indicio"], cfg["ontologia"][f]["naturalidad"]) if n in x],
                                     indicio="IndicioDeDeepfake" in x, naturalidad="RasgoDeNaturalidad" in x)
                tecnicas = sorted({cfg["ontologia"][f]["tecnica"] for f, r in rasgos.items() if r["indicio"]})   # esIndicioDe de las clases de indicio
                res[cid] = dict(consistente=True, clases_audio=sorted(c for c in CLASES_AUDIO if c in anc), rasgos=rasgos,
                                n_indicios=sum(r["indicio"] for r in rasgos.values()),
                                n_naturalidad=sum(r["naturalidad"] for r in rasgos.values()), tecnicas=tecnicas)
        finally:
            try: w.close()                                       # descarta el almacen en RAM
            except Exception: pass                               # noqa: BLE001
            del w
    meta["ms"] = round((time.time() - t0) * 1000)
    return res, meta


def clasificar_lote(casos, ruta=None, inferir_tecnicas=True):
    """Compatibilidad: devuelve solo los resultados de razonar_efimero."""
    return razonar_efimero(casos, ruta)[0]


# ================================================================== validacion semantica de una declaracion (rechazo con justificacion)
ETIQUETAS = {"AUTENTICO": "AudioAutentico", "SINTETICO": "AudioSintetico"}
TECNICAS = ("VocoderGAN", "SintesisTextoAVoz", "ClonacionDeVoz", "ConversionDeVoz")

# Catalogo de restricciones de la ontologia (texto para la justificacion). La DETECCION la hace HermiT; esto solo nombra el axioma.
AXIOMAS = [
    ({"etiqueta:AUTENTICO", "etiqueta:SINTETICO"}, "Las clases AudioAutentico y AudioSintetico son disjuntas.", "DisjointClasses(AudioAutentico, AudioSintetico)"),
    ({"etiqueta:AUTENTICO", "generado_por"}, "Un audio auténtico no puede haber sido generado por una técnica de síntesis.", "AudioAutentico ⊑ ≤0 fueGeneradoPor.TecnicaDeSintesis"),
    ({"etiqueta:AUTENTICO", "imita_a"}, "Un audio auténtico no imita la voz de ningún hablante.", "AudioAutentico ⊑ ≤0 imitaA.Hablante"),
    ({"etiqueta:AUTENTICO", "hablante", "hablante_extra"}, "Un audio auténtico fue producido por exactamente un hablante.", "AudioAutentico ⊑ =1 producidoPor.Hablante"),
    ({"hablante", "imita_a_mismo"}, "Un mismo hablante no puede ser a la vez quien produjo el audio y a quien se imita.", "DisjointObjectProperties(producidoPor, imitaA)"),
]


def _unidades(decl, cfg):
    """Descompone la declaracion en aserciones atomicas: (clave, texto, [xml]). Cada una puede quitarse para buscar el conflicto minimo."""
    from xml.sax.saxutils import escape
    ind = lambda n: '<NamedIndividual IRI="#%s"/>' % n; cls = lambda c: '<Class IRI="#%s"/>' % c
    decl_ = lambda n: '<Declaration>%s</Declaration>' % ind(n); ca = lambda c, n: '<ClassAssertion>%s%s</ClassAssertion>' % (cls(c), ind(n))
    opa = lambda p, a, b: '<ObjectPropertyAssertion><ObjectProperty IRI="#%s"/>%s%s</ObjectPropertyAssertion>' % (p, ind(a), ind(b))
    A = "audio_declarado"; us = []
    for e in decl.get("etiquetas") or []:
        if e not in ETIQUETAS: raise ValueError("etiqueta no valida: %s" % e)
        us.append(("etiqueta:" + e, "El audio es %s." % ("auténtico" if e == "AUTENTICO" else "sintético"), [ca(ETIQUETAS[e], A)]))
    h = decl.get("hablante")
    if h: us.append(("hablante", "El audio fue producido por el hablante «%s»." % h, [decl_("h_" + _seguro(h)), ca("Hablante", "h_" + _seguro(h)), opa("producidoPor", A, "h_" + _seguro(h))]))
    for k, h2 in enumerate(decl.get("hablantes_adicionales") or []):
        n2 = "hx_%d_%s" % (k, _seguro(h2)); n1 = "h_" + _seguro(h or "desconocido")
        xs = [decl_(n2), ca("Hablante", n2), opa("producidoPor", A, n2)]
        if h: xs.append('<DifferentIndividuals>%s%s</DifferentIndividuals>' % (ind(n1), ind(n2)))
        us.append(("hablante_extra", "El audio también fue producido por el hablante distinto «%s»." % h2, xs))
    for k, t in enumerate(decl.get("generado_por") or []):
        if t not in TECNICAS: raise ValueError("tecnica no valida: %s" % t)
        us.append(("generado_por", "El audio fue generado con %s." % t, [decl_("tec_%d" % k), ca(t, "tec_%d" % k), opa("fueGeneradoPor", A, "tec_%d" % k)]))
    m = decl.get("imita_a")
    if m:
        nm = "h_" + _seguro(m)
        clave = "imita_a_mismo" if m == h else "imita_a"
        xs = ([] if m == h else [decl_(nm), ca("Hablante", nm)]) + [opa("imitaA", A, nm)]
        us.append((clave, "El audio imita la voz de «%s»." % m, xs))
    for k, r in enumerate(decl.get("rasgos_declarados") or []):
        f = r["rasgo"]
        if f not in cfg["ontologia"]: raise ValueError("rasgo sin clase en la ontologia: %s" % f)
        o = cfg["ontologia"][f]; n = "rd_%d_%s" % (k, f)
        xs = [decl_(n), ca(o["base"], n), '<DataPropertyAssertion><DataProperty IRI="#valor"/>%s<Literal datatypeIRI="http://www.w3.org/2001/XMLSchema#double">%r</Literal></DataPropertyAssertion>' % (ind(n), float(r["valor"])),
              opa("tieneCaracteristica", A, n)]
        txt = "El rasgo %s vale %s" % (f, r["valor"])
        if r.get("clase") in ("natural", "anomalo"):
            xs.append(ca(o["naturalidad"] if r["clase"] == "natural" else o["indicio"], n))
            txt += " y se declara %s (%s)" % ("natural" if r["clase"] == "natural" else "anómalo", o["naturalidad"] if r["clase"] == "natural" else o["indicio"])
        us.append(("rasgo:%s:%s" % (f, r.get("clase") or "valor"), txt + ".", xs))
    return us, A


def _es_consistente(unidades, A, ruta=None):
    """True/False con HermiT sobre TBox + aserciones (en RAM). Devuelve tambien el tiempo."""
    from owlready2 import World, sync_reasoner_hermit, OwlReadyInconsistentOntologyError
    xml = '<Declaration><NamedIndividual IRI="#%s"/></Declaration><ClassAssertion><Class IRI="#Audio"/><NamedIndividual IRI="#%s"/></ClassAssertion>' % (A, A)
    xml += "\n".join(x for _, _, xs in unidades for x in xs)
    datos = tbox(ruta).decode("utf-8").replace("</Ontology>", xml + "\n</Ontology>").encode("utf-8")
    with _lock:
        w = World()
        try:
            o = w.get_ontology("https://example.org/ontologias/artefactos-audio").load(fileobj=io.BytesIO(datos))
            try:
                with o: sync_reasoner_hermit(w, infer_property_values=False, debug=0)
                return True
            except OwlReadyInconsistentOntologyError:
                return False
        finally:
            try: w.close()
            except Exception: pass                        # noqa: BLE001


def validar_declaracion(decl, ruta=None):
    """Validador semantico: comprueba con HermiT (RAM, efimero) que lo declarado sobre un audio sea coherente con la ontologia.
    Si no lo es, calcula un subconjunto MINIMO de aserciones en conflicto (re-ejecutando HermiT sin cada una) y nombra el axioma violado."""
    cfg = cargar_umbrales(); t0 = time.time(); decl = decl or {}
    unidades, A = _unidades(decl, cfg)
    meta = dict(almacen="memoria", version_ontologia=version_ontologia(ruta), ejecuciones_hermit=0)
    if not unidades:
        return dict(consistente=True, aserciones=[], conflicto_minimo=[], axiomas_violados=[], justificacion="No hay aserciones que validar.", meta={**meta, "ms": 0})
    meta["ejecuciones_hermit"] += 1
    if _es_consistente(unidades, A, ruta):
        meta["ms"] = round((time.time() - t0) * 1000)
        return dict(consistente=True, aserciones=[u[1] for u in unidades], conflicto_minimo=[], axiomas_violados=[],
                    justificacion="Las %d aserciones son consistentes con la ontologia (HermiT)." % len(unidades), meta=meta)
    nucleo = list(unidades)                              # minimizacion por eliminacion: conserva solo lo imprescindible para la inconsistencia
    for u in list(nucleo):
        prueba = [x for x in nucleo if x is not u]
        meta["ejecuciones_hermit"] += 1
        if prueba and not _es_consistente(prueba, A, ruta): nucleo = prueba
    claves = {u[0] for u in nucleo}
    axiomas = [dict(descripcion=d, formal=f) for req, d, f in AXIOMAS if req <= claves]
    if not axiomas:
        for u in nucleo:
            if u[0].startswith("rasgo:"):
                f = u[0].split(":")[1]; o = cfg["ontologia"][f]
                umb = regla_de(f, cfg)["umbral_indicio"]
                axiomas.append(dict(descripcion="El valor declarado cae en la zona de indicio de la ontologia (%s), que la clasifica automáticamente como %s, clase disjunta de %s." % (
                    " y ".join("valor %s %.6g" % ({"minExclusive": ">", "maxExclusive": "<"}.get(k, k), v) for k, v in umb.items()), o["indicio"], o["naturalidad"]),
                                    formal="%s ≡ %s ⊓ ∃valor.(umbral) ; DisjointClasses(%s, %s)" % (o["indicio"], o["base"], o["indicio"], o["naturalidad"])))
    if not axiomas: axiomas.append(dict(descripcion="La combinación de aserciones es insatisfacible según los axiomas de la ontología.", formal="(ver ontologia .owx)"))
    just = "HermiT declara INCONSISTENTE la solicitud. Aserciones en conflicto: " + " ".join("(%d) %s" % (i + 1, u[1]) for i, u in enumerate(nucleo)) + \
           " Axioma(s) violado(s): " + " ".join("%s [%s]" % (a["descripcion"], a["formal"]) for a in axiomas)
    meta["ms"] = round((time.time() - t0) * 1000)
    return dict(consistente=False, aserciones=[u[1] for u in unidades], conflicto_minimo=[u[1] for u in nucleo], axiomas_violados=axiomas, justificacion=just, meta=meta)
