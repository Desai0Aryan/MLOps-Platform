"""Retention: prediction logs, monitoring runs and shadow rows grow forever unless pruned.

Ordering matters: labels reference predictions, so labels go first. For very large deployments, swap this for
time-partitioned tables (drop a partition instead of DELETE); the policy stays the same.
"""
from __future__ import annotations
import datetime as dt

from sqlalchemy import delete, select
from sqlalchemy.orm import sessionmaker

from .serving.db import GroundTruth, MonitoringRun, PredictionLog, ShadowPrediction, utcnow


def prune_old_data(engine, cfg: dict, now: dt.datetime | None = None) -> dict:
    r, now = cfg["retention"], now or utcnow()
    assert r["predictions_days"] >= cfg["retraining"]["lookback_days"], \
        "retention.predictions_days must cover retraining.lookback_days, or retraining would lose its data"
    cut = lambda days: now - dt.timedelta(days=days)
    out = {}
    with sessionmaker(engine)() as db:
        out["monitoring_runs"] = db.execute(delete(MonitoringRun).where(MonitoringRun.created_at < cut(r["monitoring_runs_days"]))).rowcount
        out["shadow_predictions"] = db.execute(delete(ShadowPrediction).where(ShadowPrediction.created_at < cut(r["shadow_days"]))).rowcount
        old = select(PredictionLog.request_id).where(PredictionLog.created_at < cut(r["predictions_days"]))
        out["labels"] = db.execute(delete(GroundTruth).where(GroundTruth.request_id.in_(old))).rowcount
        out["predictions"] = db.execute(delete(PredictionLog).where(PredictionLog.created_at < cut(r["predictions_days"]))).rowcount
        db.commit()
    return out
