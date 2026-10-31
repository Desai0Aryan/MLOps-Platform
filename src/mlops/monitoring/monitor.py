"""Monitor: pulls live traffic from the prediction log, compares it with the champion's own
reference profile (stored in MLflow next to the model), and persists the verdict."""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field, asdict

import mlflow
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ..config import feature_columns
from ..serving.db import GroundTruth, MonitoringRun, PredictionLog, utcnow
from .drift import SEV, detect_drift, worst
from .performance import performance_report


@dataclass
class MonitoringReport:
    created_at: str
    model_version: int
    status: str
    trigger_retrain: bool
    reasons: list[str]
    n_rows: int
    n_labeled: int
    drift: dict | None
    performance: dict

    def to_dict(self): return asdict(self)


class Monitor:
    def __init__(self, cfg: dict, engine, registry):
        self.cfg, self.engine, self.registry = cfg, engine, registry
        self.Session = sessionmaker(engine, expire_on_commit=False)
        self._ref_cache: dict[int, tuple[dict, dict]] = {}

    # ----------------------------------------------------------- reference data
    def reference(self, version: int) -> tuple[dict, dict]:
        """(reference profile, baseline metrics) for a model version, cached."""
        if version not in self._ref_cache:
            mv = self.registry.get_version(version)
            run_id = mv.tags["run_id"]
            profile = mlflow.artifacts.load_dict(f"runs:/{run_id}/reference/reference_profile.json")
            m = self.registry.client.get_run(run_id).data.metrics
            baseline = {k: m[f"test_{k}"] for k in ("roc_auc", "log_loss", "brier", "accuracy") if f"test_{k}" in m}
            baseline["positive_rate"] = profile.get("target_positive_rate")
            self._ref_cache[version] = (profile, baseline)
        return self._ref_cache[version]

    # ---------------------------------------------------------------- live data
    def _features_frame(self, rows) -> pd.DataFrame:
        num, cat = feature_columns(self.cfg)
        df = pd.DataFrame([r.features for r in rows])
        for c in num:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def live_window(self, version: int) -> tuple[pd.DataFrame, np.ndarray]:
        n = self.cfg["monitoring"]["window_rows"]
        with self.Session() as db:
            rows = db.execute(select(PredictionLog.features, PredictionLog.probability)
                              .where(PredictionLog.model_version == version, PredictionLog.role == "champion")
                              .order_by(PredictionLog.id.desc()).limit(n)).all()
        if not rows:
            return pd.DataFrame(), np.array([])
        return self._features_frame(rows), np.array([r.probability for r in rows])

    def labeled_window(self, version: int) -> tuple[np.ndarray, np.ndarray]:
        n = self.cfg["monitoring"]["performance"]["window_labeled"]
        with self.Session() as db:
            rows = db.execute(select(PredictionLog.probability, GroundTruth.actual)
                              .join(GroundTruth, GroundTruth.request_id == PredictionLog.request_id)
                              .where(PredictionLog.model_version == version, PredictionLog.role == "champion")
                              .order_by(PredictionLog.id.desc()).limit(n)).all()
        return np.array([r.actual for r in rows], int), np.array([r.probability for r in rows], float)

    # --------------------------------------------------------------------- run
    def run(self, version: int | None = None, persist: bool = True) -> MonitoringReport:
        if version is None:
            champ = self.registry.champion()
            if champ is None:
                raise RuntimeError("no champion model to monitor")
            version = int(champ.version)
        profile, baseline = self.reference(version)
        df, scores = self.live_window(version)
        y, p = self.labeled_window(version)
        mc = self.cfg["monitoring"]

        drift = detect_drift(df, profile, self.cfg, scores) if len(df) >= mc["min_rows"] else None
        perf = performance_report(y, p, baseline, self.cfg)
        d_status = drift.status if drift else "insufficient_data"
        statuses = [s for s in (d_status, perf.status) if s in SEV]
        status = worst(*statuses) if statuses else "insufficient_data"

        reasons = []
        if drift and drift.status != "ok":
            reasons.append(f"data drift [{drift.status}]: {drift.summary}")
        for ck in perf.checks:
            if ck.severity != "ok":
                reasons.append(f"performance [{ck.severity}] {ck.name}: {ck.detail}")
        if status == "insufficient_data":
            reasons.append("not enough traffic yet")
        if status == "ok":
            reasons.append("all drift and performance checks within tolerance")

        rep = MonitoringReport(utcnow().isoformat(), version, status, status == "critical", reasons, len(df), perf.n_labeled,
                               drift.to_dict() if drift else None, perf.to_dict())
        if persist:
            max_psi = max([f.psi for f in drift.features if f.psi is not None], default=None) if drift else None
            with self.Session() as db:
                db.add(MonitoringRun(model_version=version, status=status, drift_status=d_status, perf_status=perf.status,
                                     n_rows=len(df), n_labeled=perf.n_labeled, max_psi=max_psi, report=rep.to_dict()))
                db.commit()
        return rep

    def history(self, limit: int = 50, version: int | None = None) -> list[dict]:
        q = select(MonitoringRun).order_by(MonitoringRun.id.desc()).limit(limit)
        if version is not None:
            q = q.where(MonitoringRun.model_version == version)
        with self.Session() as db:
            return [dict(id=r.id, created_at=r.created_at.isoformat(), model_version=r.model_version, status=r.status,
                         drift_status=r.drift_status, perf_status=r.perf_status, n_rows=r.n_rows,
                         n_labeled=r.n_labeled, max_psi=r.max_psi) for r in db.scalars(q)]

    def latest(self) -> dict | None:
        with self.Session() as db:
            r = db.scalars(select(MonitoringRun).order_by(MonitoringRun.id.desc()).limit(1)).first()
            return r.report if r else None
