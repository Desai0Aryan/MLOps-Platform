"""Prediction log + ground-truth tables (SQLAlchemy 2.0; SQLite in tests, PostgreSQL in prod).

Design: every prediction is an immutable event row. Labels arrive later (churn is only
known weeks afterwards), so they live in a separate table joined on request_id.
This is the data Part 5 uses for drift detection and live performance monitoring.
"""
from __future__ import annotations
import datetime as dt
from sqlalchemy import JSON, DateTime, text, Float, ForeignKey, Index, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[int] = mapped_column(Integer, index=True)
    role: Mapped[str] = mapped_column(String(20), default="champion")   # champion | challenger | shadow (Part 4)
    features: Mapped[dict] = mapped_column(JSON)
    probability: Mapped[float] = mapped_column(Float)
    prediction: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    __table_args__ = (Index("ix_pred_version_time", "model_version", "created_at"),)


class GroundTruth(Base):
    __tablename__ = "ground_truth"
    request_id: Mapped[str] = mapped_column(String(40), ForeignKey("prediction_log.request_id"), primary_key=True)
    actual: Mapped[int] = mapped_column(Integer)
    labeled_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class ShadowPrediction(Base):
    """What the challenger WOULD have said. Never returned to the caller."""
    __tablename__ = "shadow_prediction"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(40), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)
    model_version: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(String(300), nullable=True)


class DeploymentState(Base):
    """Single-row table: the rollout state shared by every API replica."""
    __tablename__ = "deployment_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(20), default="champion_only")   # champion_only | shadow | ab
    traffic_split: Mapped[float] = mapped_column(Float, default=0.0)
    challenger_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class MonitoringRun(Base):
    __tablename__ = "monitoring_run"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)
    model_version: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(20))          # ok | warning | critical | insufficient_data
    drift_status: Mapped[str] = mapped_column(String(20))
    perf_status: Mapped[str] = mapped_column(String(20))
    n_rows: Mapped[int] = mapped_column(Integer)
    n_labeled: Mapped[int] = mapped_column(Integer)
    max_psi: Mapped[float | None] = mapped_column(Float, nullable=True)
    report: Mapped[dict] = mapped_column(JSON)


class LifecycleEvent(Base):
    """Append-only audit timeline of everything the platform decided on its own."""
    __tablename__ = "lifecycle_event"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)
    kind: Mapped[str] = mapped_column(String(50), index=True)
    model_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


def log_event(SessionLocal, kind: str, detail: dict | None = None, model_version: int | None = None):
    with SessionLocal() as db:
        db.add(LifecycleEvent(kind=kind, detail=detail or {}, model_version=model_version)); db.commit()


def make_engine(url: str):
    if url.startswith("sqlite"):
        kw = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url or url == "sqlite://":
            kw["poolclass"] = StaticPool
        return create_engine(url, **kw)
    return create_engine(url, pool_size=10, max_overflow=20, pool_pre_ping=True)


def init_db(url: str):
    engine = make_engine(url)
    if engine.dialect.name == "postgresql":
        # Several replicas can boot simultaneously: serialise CREATE TABLE with an advisory lock
        # (released automatically at the end of the transaction). Real migrations (Alembic) are the next step up.
        with engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(727274)"))
            Base.metadata.create_all(conn)
    else:
        Base.metadata.create_all(engine)
    return engine, sessionmaker(engine, expire_on_commit=False)
