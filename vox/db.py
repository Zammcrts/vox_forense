# -*- coding: utf-8 -*-
"""Persistencia en Postgres (SQLAlchemy 2.0 + psycopg 3). Se guardan SOLO rasgos y resultados: el audio nunca se almacena."""
from datetime import datetime, timezone
from sqlalchemy import (BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint,
                        create_engine, func, select)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from .config import DATABASE_URL

JSONT = JSON().with_variant(JSONB(), "postgresql")
ahora = lambda: datetime.now(timezone.utc)


class Base(DeclarativeBase): pass


class Ejecucion(Base):
    __tablename__ = "ejecuciones"
    run_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(200), index=True)
    modo: Mapped[str] = mapped_column(String(10))                       # directo | crew
    estado: Mapped[str] = mapped_column(String(15), default="en_proceso")   # en_proceso | completado | error
    nombre_original: Mapped[str | None] = mapped_column(String(300))
    error: Mapped[str | None] = mapped_column(Text)
    creado: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora)
    terminado: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    caso: Mapped["Caso | None"] = relationship(back_populates="ejecucion", uselist=False)


class Caso(Base):
    __tablename__ = "casos"
    case_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("ejecuciones.run_id"), unique=True)
    nombre_original: Mapped[str | None] = mapped_column(String(300))
    rasgos: Mapped[dict] = mapped_column(JSONT, default=dict)           # vector de caracteristicas (hechos)
    faltantes: Mapped[list] = mapped_column(JSONT, default=list)
    hallazgos: Mapped[dict] = mapped_column(JSONT, default=dict)        # especialistas
    ontologia: Mapped[dict] = mapped_column(JSONT, default=dict)        # clases inferidas por HermiT
    licitacion: Mapped[dict] = mapped_column(JSONT, default=dict)
    veredicto: Mapped[str | None] = mapped_column(String(10), index=True)   # SINTETICO | AUTENTICO | ESCALAR
    regla: Mapped[str | None] = mapped_column(String(4))
    motivo: Mapped[str | None] = mapped_column(Text)
    n_indicios: Mapped[int | None] = mapped_column(Integer)
    n_naturalidad: Mapped[int | None] = mapped_column(Integer)
    contradictoria: Mapped[bool] = mapped_column(Boolean, default=False)
    explicacion: Mapped[str | None] = mapped_column(Text)
    version_ontologia: Mapped[str | None] = mapped_column(String(32))
    etiqueta_real: Mapped[str | None] = mapped_column(String(5))        # real | fake (la fija un analista)
    creado: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora)
    ejecucion: Mapped[Ejecucion] = relationship(back_populates="caso")


class EventoTraza(Base):
    """Traza inmutable (Seccion 3.3): solo se inserta, nunca se actualiza."""
    __tablename__ = "eventos_traza"
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("ejecuciones.run_id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    parent_seq: Mapped[int | None] = mapped_column(Integer)
    ts: Mapped[float] = mapped_column(Float)
    evento: Mapped[str] = mapped_column(String(20), index=True)
    agente: Mapped[str | None] = mapped_column(String(40))
    tarea: Mapped[str | None] = mapped_column(String(10))
    payload: Mapped[dict] = mapped_column(JSONT, default=dict)
    __table_args__ = (UniqueConstraint("run_id", "seq"),)


class Escalamiento(Base):
    __tablename__ = "escalamientos"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("casos.case_id"), unique=True)
    motivo: Mapped[str] = mapped_column(Text)
    estado: Mapped[str] = mapped_column(String(10), default="pendiente", index=True)   # pendiente | resuelto
    etiqueta_analista: Mapped[str | None] = mapped_column(String(5))
    comentario: Mapped[str | None] = mapped_column(Text)
    creado: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora)
    resuelto: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Competencia(Base):
    """Registro persistente del Contract Net: competencia historica por agente y tipo de tarea (se actualiza con EMA)."""
    __tablename__ = "competencias"
    agent_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(20), primary_key=True)
    competencia: Mapped[float] = mapped_column(Float)
    n_actualizaciones: Mapped[int] = mapped_column(Integer, default=0)
    actualizado: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora, onupdate=ahora)


_engine = None; _Session = None


def engine():
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
        _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def sesion():
    engine(); return _Session()


def iniciar_db():
    """Crea las tablas (si no existen) y siembra la competencia inicial del Contract Net."""
    Base.metadata.create_all(engine())
    from .contract_net import competencias_iniciales
    with sesion() as s, s.begin():
        existentes = {(c.agent_id, c.task_type) for c in s.scalars(select(Competencia))}
        for (ag, t), v in competencias_iniciales().items():
            if (ag, t) not in existentes: s.add(Competencia(agent_id=ag, task_type=t, competencia=v))


def db_disponible():
    try:
        with engine().connect() as c: c.exec_driver_sql("select 1")
        return True
    except Exception:                  # noqa: BLE001
        return False
