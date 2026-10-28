"""Prediction log + ground-truth tables (SQLAlchemy 2.0; SQLite in tests, PostgreSQL in prod).

Design: every prediction is an immutable event row. Labels arrive later (churn is only
known weeks afterwards), so they live in a separate table joined on request_id.
This is the data Part 5 uses for drift detection and live performance monitoring.
"""
from __future__ import annotations
import datetime as dt
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, create_engine
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


def make_engine(url: str):
    if url.startswith("sqlite"):
        kw = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url or url == "sqlite://":
            kw["poolclass"] = StaticPool
        return create_engine(url, **kw)
    return create_engine(url, pool_size=10, max_overflow=20, pool_pre_ping=True)


def init_db(url: str):
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    return engine, sessionmaker(engine, expire_on_commit=False)
