"""The closed loop.  One `tick()` is one pass of the platform's autonomous control cycle:

    ┌─► MONITOR champion (drift + live quality) ─────────────────────────────┐
    │        │ critical?                                                      │
    │        ▼                                                                │
    │   experiment running? ──yes──► ADVANCE it (shadow → A/B → promote/stop) │
    │        │ no                                                             │
    │        ▼                                                                │
    │   enough fresh LABELED data + cooldown over? ──no──► wait, log why      │
    │        │ yes                                                            │
    │        ▼                                                                │
    │   RETRAIN on recent production data (validation + quality gates apply)  │
    │        ▼                                                                │
    │   REGISTER as challenger → start SHADOW ────────────────────────────────┘

Run it on a timer (`mlops worker`) or call it from a scheduler. Every decision is written to the
`lifecycle_event` table, so you can always answer "why did the model change at 3 am?".
"""
from __future__ import annotations
import datetime as dt

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from .config import feature_columns
from .monitoring import Monitor
from .registry import ModelRegistry
from .retention import prune_old_data
from .serving.db import GroundTruth, LifecycleEvent, PredictionLog, log_event, utcnow
from .training import DataValidationError, QualityGateError, run_training_pipeline


def build_training_frame(engine, cfg: dict) -> pd.DataFrame:
    """Most recent LABELED production rows -> a training frame (same columns as the original data)."""
    rc = cfg["retraining"]; num, cat = feature_columns(cfg)
    since = utcnow() - dt.timedelta(days=rc["lookback_days"])
    with sessionmaker(engine)() as db:
        rows = db.execute(select(PredictionLog.features, GroundTruth.actual)
                          .join(GroundTruth, GroundTruth.request_id == PredictionLog.request_id)
                          .where(PredictionLog.created_at >= since)
                          .order_by(PredictionLog.id.desc()).limit(rc["max_rows"])).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r.features for r in rows])[num + cat]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cfg["data"]["target"]] = [int(r.actual) for r in rows]
    return df.iloc[::-1].reset_index(drop=True)          # chronological


class Orchestrator:
    def __init__(self, cfg: dict, engine, api, registry: ModelRegistry | None = None):
        """api: any httpx-style client pointed at the serving API (httpx.Client or TestClient)."""
        self.cfg, self.engine, self.api = cfg, engine, api
        self.Session = sessionmaker(engine, expire_on_commit=False)
        self.registry = registry or ModelRegistry(cfg)
        self.monitor = Monitor(cfg, engine, self.registry)
        self._last_prune = 0.0

    # -------------------------------------------------------------- helpers
    def _last_event(self, *kinds):
        with self.Session() as db:
            q = select(LifecycleEvent).order_by(LifecycleEvent.id.desc()).limit(1)
            if kinds:
                q = q.where(LifecycleEvent.kind.in_(kinds))
            return db.scalars(q).first()

    def _log_once(self, kind: str, detail: dict, version=None):
        """Avoid flooding the timeline when the same blocking reason repeats every tick."""
        last = self._last_event()
        if last and last.kind == kind and last.detail.get("reason") == detail.get("reason"):
            return
        log_event(self.Session, kind, detail, version)

    # ----------------------------------------------------------------- tick
    def tick(self) -> dict:
        out: dict = {"steps": []}
        self._maybe_prune(out)
        champ = self.registry.champion()
        if champ is None:
            return {"action": "idle", "reason": "no champion model"}
        rep = self.monitor.run(int(champ.version))
        out.update(model_version=rep.model_version, status=rep.status, reasons=rep.reasons)
        out["steps"].append(f"monitored v{rep.model_version}: {rep.status}")

        dep = self.api.get("/deployment").json()
        if dep["mode"] != "champion_only":
            r = self.api.post("/experiment/advance")
            out["steps"].append(f"advanced experiment: {r.json().get('action')}")
            out["action"] = f"advance:{r.json().get('action')}"; out["advance"] = r.json()
            return out

        if not rep.trigger_retrain:
            out["action"] = "none"; return out
        if not self.cfg["retraining"]["enabled"]:
            out["action"] = "retraining_disabled"; return out
        return {**out, **self._maybe_retrain(rep)}

    def _maybe_prune(self, out: dict):
        import time
        if time.time() - self._last_prune < 3600:
            return
        self._last_prune = time.time()
        try:
            pruned = prune_old_data(self.engine, self.cfg)
            if any(pruned.values()):
                log_event(self.Session, "retention_pruned", {k: v for k, v in pruned.items() if v})
                out["pruned"] = pruned
        except Exception as e:           # housekeeping must never block the control loop
            out["prune_error"] = f"{type(e).__name__}: {e}"

    # -------------------------------------------------------------- retrain
    def _maybe_retrain(self, rep) -> dict:
        rc = self.cfg["retraining"]; steps = []
        last = self._last_event("retrain_started")
        if last and (utcnow() - last.created_at) < dt.timedelta(hours=rc["cooldown_hours"]):
            reason = f"cooldown ({rc['cooldown_hours']}h) active"
            self._log_once("retrain_skipped", {"reason": reason}, rep.model_version)
            return {"action": "skipped", "reason": reason}

        df = build_training_frame(self.engine, self.cfg)
        if len(df) < rc["min_labeled_rows"]:
            reason = f"only {len(df)} labeled production rows, need {rc['min_labeled_rows']} (waiting for ground truth)"
            self._log_once("retrain_skipped", {"reason": reason, "have": len(df), "need": rc["min_labeled_rows"]}, rep.model_version)
            return {"action": "skipped", "reason": reason}

        log_event(self.Session, "retrain_started", {"trigger": rep.reasons[:3], "rows": len(df),
                                                     "positive_rate": round(float(df[self.cfg["data"]["target"]].mean()), 3)}, rep.model_version)
        try:
            res = run_training_pipeline(df, self.cfg, run_name="auto-retrain",
                                        tags={"trigger": "monitoring", "trigger_status": rep.status, "prod_rows": str(len(df))})
        except DataValidationError as e:
            log_event(self.Session, "retrain_rejected_bad_data", {"reason": str(e)[:300]}, rep.model_version)
            return {"action": "rejected", "reason": f"production data failed validation: {e}"}
        except QualityGateError as e:
            log_event(self.Session, "retrain_rejected_quality", {"reason": str(e)}, rep.model_version)
            return {"action": "rejected", "reason": str(e)}

        mv = self.registry.register_run(res.run_id)
        log_event(self.Session, "retrain_completed", {"auc": round(res.metrics["roc_auc"], 4), "algorithm": res.best_model,
                                                      "rows": len(df)}, int(mv.version))
        r = self.api.post("/deployment", json={"mode": "shadow"})
        if r.status_code != 200:
            log_event(self.Session, "shadow_start_failed", {"reason": r.text[:300]}, int(mv.version))
            return {"action": "retrained_not_deployed", "reason": r.text, "challenger_version": int(mv.version)}
        return {"action": "retrained_and_shadowing", "challenger_version": int(mv.version), "rows": len(df)}
