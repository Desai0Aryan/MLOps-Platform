"""FastAPI model-serving service.

  POST /predict          single prediction (validated, logged)
  POST /predict/batch    up to `max_batch` predictions in one call
  POST /feedback[/batch] attach delayed ground-truth labels to past predictions
  GET  /model-info       which model version is live, and where it came from
  POST /reload           re-resolve the `champion` alias (zero-downtime hot swap)
  GET  /stats            live traffic + (when labels exist) live model quality
  GET  /health /ready    liveness vs. readiness probes (k8s / Docker friendly)

Logging happens in a background task, so the database never adds latency to a prediction.
"""
# NOTE: no `from __future__ import annotations` here: FastAPI must resolve the dynamically
# generated request model from the real annotation, not a string.
import asyncio, time, uuid
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score
from sqlalchemy import func, select

from ..config import feature_columns, load_config
from .db import GroundTruth, PredictionLog, init_db
from .schemas import FeedbackBatch, FeedbackRequest, PredictionResponse, build_request_model
from .store import ModelStore


def create_app(cfg: dict | None = None, database_url: str | None = None) -> FastAPI:
    import os
    cfg = cfg or load_config()
    scfg = cfg["serving"]
    db_url = database_url or os.getenv("DATABASE_URL") or scfg["database_url"]
    num, cat = feature_columns(cfg)
    features = num + cat
    PredictionRequest = build_request_model(cfg)

    class BatchRequest(BaseModel):
        instances: list[PredictionRequest] = Field(min_length=1, max_length=scfg["max_batch"])

    store = ModelStore(cfg)
    engine, SessionLocal = init_db(db_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(store.refresh, "champion")
        task = None
        if scfg["refresh_seconds"] > 0:
            async def poll():
                while True:
                    await asyncio.sleep(scfg["refresh_seconds"])
                    try:
                        await asyncio.to_thread(store.refresh, "champion")
                    except Exception:    # registry blip must never kill the service
                        pass
            task = asyncio.create_task(poll())
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Churn MLOps Serving API", version="0.3.0", lifespan=lifespan,
                  description="Production model-serving layer: validated inputs, versioned models, "
                              "full prediction logging, delayed-label feedback and hot reloads.")
    app.state.store, app.state.SessionLocal, app.state.cfg, app.state.engine = store, SessionLocal, cfg, engine

    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
        return resp

    def require_model():
        lm = store.get("champion")
        if lm is None:
            raise HTTPException(503, "No champion model is registered yet. Train + promote one first.")
        return lm

    def risk_band(p: float) -> str:
        lo, hi = scfg["risk_bands"]
        return "low" if p < lo else ("medium" if p < hi else "high")

    def run_inference(lm, rows: list[dict]):
        df = pd.DataFrame(rows)[features].astype({c: "float64" for c in num})
        t0 = time.perf_counter()
        proba = lm.model.predict_proba(df)[:, 1]
        per_row_ms = (time.perf_counter() - t0) * 1000 / len(rows)
        return proba, per_row_ms

    def to_records(lm, rows, proba, ms) -> list[dict]:
        return [dict(request_id=uuid.uuid4().hex, entity_id=r.get("entity_id"), model_name=lm.name,
                     model_version=lm.version, role="champion",
                     features={k: r.get(k) for k in features}, probability=float(p),
                     prediction=int(p >= scfg["threshold"]), latency_ms=round(ms, 3))
                for r, p in zip(rows, proba)]

    def persist(records: list[dict]):
        with SessionLocal() as db:
            db.add_all([PredictionLog(**r) for r in records])
            db.commit()

    def respond(rec: dict) -> PredictionResponse:
        return PredictionResponse(request_id=rec["request_id"], probability=round(rec["probability"], 6),
                                  prediction=rec["prediction"], risk=risk_band(rec["probability"]),
                                  model_name=rec["model_name"], model_version=rec["model_version"],
                                  latency_ms=rec["latency_ms"])

    # ------------------------------------------------------------------ probes
    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready():
        lm = store.get("champion")
        if lm is None:
            raise HTTPException(503, "no champion model loaded")
        return {"status": "ready", "model_version": lm.version}

    # ------------------------------------------------------------- prediction
    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(req: PredictionRequest, bg: BackgroundTasks):
        lm = require_model()
        rows = [req.model_dump()]
        proba, ms = run_inference(lm, rows)
        rec = to_records(lm, rows, proba, ms)[0]
        bg.add_task(persist, [rec])
        return respond(rec)

    @app.post("/predict/batch", response_model=list[PredictionResponse], tags=["inference"])
    def predict_batch(req: BatchRequest, bg: BackgroundTasks):
        lm = require_model()
        rows = [i.model_dump() for i in req.instances]
        proba, ms = run_inference(lm, rows)
        recs = to_records(lm, rows, proba, ms)
        bg.add_task(persist, recs)
        return [respond(r) for r in recs]

    # --------------------------------------------------------------- feedback
    def upsert_labels(db, items: list[FeedbackRequest]) -> tuple[int, list[str]]:
        known = set(db.scalars(select(PredictionLog.request_id).where(
            PredictionLog.request_id.in_([i.request_id for i in items]))))
        ok, missing = 0, []
        for it in items:
            if it.request_id not in known:
                missing.append(it.request_id); continue
            db.merge(GroundTruth(request_id=it.request_id, actual=it.actual)); ok += 1
        db.commit()
        return ok, missing

    @app.post("/feedback", tags=["feedback"])
    def feedback(fb: FeedbackRequest, db=Depends(get_db)):
        ok, missing = upsert_labels(db, [fb])
        if missing:
            raise HTTPException(404, f"unknown request_id {fb.request_id}")
        return {"stored": ok}

    @app.post("/feedback/batch", tags=["feedback"])
    def feedback_batch(fb: FeedbackBatch, db=Depends(get_db)):
        ok, missing = upsert_labels(db, fb.items)
        return {"stored": ok, "unknown_request_ids": missing[:50], "n_unknown": len(missing)}

    # ---------------------------------------------------------------- admin
    @app.get("/model-info", tags=["ops"])
    def model_info():
        lm = require_model()
        return {"name": lm.name, "version": lm.version, "alias": lm.alias, "algorithm": lm.algorithm,
                "run_id": lm.run_id, "loaded_at": lm.loaded_at, "features": features,
                "test_roc_auc": lm.tags.get("test_roc_auc"), "data_hash": lm.tags.get("data_hash"),
                "threshold": scfg["threshold"]}

    @app.post("/reload", tags=["ops"])
    def reload():
        changed, lm = store.refresh("champion")
        return {"changed": changed, "model_version": lm.version if lm else None}

    @app.get("/stats", tags=["ops"])
    def stats(window: int = Query(1000, ge=10, le=100000), db=Depends(get_db)):
        total = db.scalar(select(func.count(PredictionLog.id))) or 0
        rows = db.execute(select(PredictionLog.model_version, PredictionLog.probability, PredictionLog.prediction,
                                 PredictionLog.latency_ms, GroundTruth.actual)
                          .join(GroundTruth, GroundTruth.request_id == PredictionLog.request_id, isouter=True)
                          .order_by(PredictionLog.id.desc()).limit(window)).all()
        out = {"total_predictions": total, "window": len(rows), "by_version": {}}
        if not rows:
            return out
        lat = np.array([r.latency_ms for r in rows]); pred = np.array([r.prediction for r in rows])
        out.update(positive_rate=float(pred.mean()), mean_probability=float(np.mean([r.probability for r in rows])),
                   latency_ms={"mean": float(lat.mean()), "p95": float(np.percentile(lat, 95))})
        for r in rows:
            out["by_version"][str(r.model_version)] = out["by_version"].get(str(r.model_version), 0) + 1
        lab = [r for r in rows if r.actual is not None]
        out["labeled"] = len(lab)
        if lab:
            y = np.array([r.actual for r in lab]); p = np.array([r.probability for r in lab])
            pr = np.array([r.prediction for r in lab])
            out["live_quality"] = {"n": len(lab), "accuracy": float((pr == y).mean()),
                                   "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None}
        return out

    @app.get("/predictions/recent", tags=["ops"])
    def recent(limit: int = Query(20, ge=1, le=500), db=Depends(get_db)):
        q = db.scalars(select(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit)).all()
        return [dict(request_id=r.request_id, created_at=r.created_at.isoformat(), model_version=r.model_version,
                     probability=round(r.probability, 4), prediction=r.prediction, latency_ms=r.latency_ms,
                     features=r.features) for r in q]

    return app
