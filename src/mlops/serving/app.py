"""FastAPI model-serving service (v0.4: safe rollouts).

  POST /predict, /predict/batch   validated, versioned, logged inference (champion / shadow / A-B)
  POST /feedback[/batch]          delayed ground-truth labels
  GET  /deployment                current rollout state
  POST /deployment                champion_only | shadow | ab (traffic split)       [admin]
  GET  /experiment/analysis       live statistics + verdict for the running experiment
  POST /experiment/conclude       promote / stop / auto                             [admin]
  GET  /model-info /stats /predictions/recent /health /ready ;  POST /reload        [admin]

Safety properties:
  * shadow scoring runs AFTER the response is sent, in a background task: it can't slow or break a prediction
  * if the challenger throws during A/B, the request transparently falls back to the champion
  * a challenger must pass a golden-set safety floor before it sees any live traffic
  * a challenger can never receive a majority of traffic
"""
# NOTE: no `from __future__ import annotations` here: FastAPI must resolve the dynamically
# generated request model from the real annotation, not a string.
import asyncio, os, time, uuid
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score
from sqlalchemy import func, select

from ..config import feature_columns, load_config
from ..experiments import ab_report, load_ab_frame, load_shadow_frame, shadow_report
from ..registry import RegistryError
from .db import GroundTruth, PredictionLog, ShadowPrediction, init_db
from .deployment import DeploymentManager, assign_arm
from .schemas import FeedbackBatch, FeedbackRequest, PredictionResponse, build_request_model
from .store import ModelStore


class DeploymentRequest(BaseModel):
    mode: Literal["champion_only", "shadow", "ab"]
    traffic_split: float = Field(0.1, gt=0, lt=1, description="share of traffic for the challenger (ab mode)")


class ConcludeRequest(BaseModel):
    action: Literal["auto", "promote", "stop"] = "auto"


def create_app(cfg: dict | None = None, database_url: str | None = None) -> FastAPI:
    cfg = cfg or load_config()
    scfg, dcfg = cfg["serving"], cfg["deployment"]
    db_url = database_url or os.getenv("DATABASE_URL") or scfg["database_url"]
    num, cat = feature_columns(cfg)
    features = num + cat
    PredictionRequest = build_request_model(cfg)

    class BatchRequest(BaseModel):
        instances: list[PredictionRequest] = Field(min_length=1, max_length=scfg["max_batch"])

    store = ModelStore(cfg)
    engine, SessionLocal = init_db(db_url)
    deploy = DeploymentManager(SessionLocal, dcfg["state_cache_seconds"])

    def refresh_all():
        out = {}
        for role in ("champion", "challenger"):
            out[role] = store.refresh(role)
        return out

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(refresh_all)
        task = None
        if scfg["refresh_seconds"] > 0:
            async def poll():
                while True:
                    await asyncio.sleep(scfg["refresh_seconds"])
                    try:
                        await asyncio.to_thread(refresh_all)
                    except Exception:    # a registry blip must never kill the service
                        pass
            task = asyncio.create_task(poll())
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Churn MLOps Serving API", version="0.4.0", lifespan=lifespan,
                  description="Versioned model serving with shadow deployment, A/B testing, "
                              "statistical decision-making and zero-downtime rollouts.")
    app.state.store, app.state.SessionLocal, app.state.cfg = store, SessionLocal, cfg
    app.state.engine, app.state.deploy = engine, deploy

    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def require_admin(x_admin_token: str | None = Header(default=None)):
        token = os.getenv("ADMIN_TOKEN")
        if token and x_admin_token != token:
            raise HTTPException(401, "admin token required (X-Admin-Token header)")

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
        return resp

    # ------------------------------------------------------------------ helpers
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
        return proba, (time.perf_counter() - t0) * 1000 / len(rows)

    def make_record(lm, role, row, rid, p, ms) -> dict:
        return dict(request_id=rid, entity_id=row.get("entity_id"), model_name=lm.name, model_version=lm.version,
                    role=role, features={k: row.get(k) for k in features}, probability=float(p),
                    prediction=int(p >= scfg["threshold"]), latency_ms=round(ms, 3))

    def persist(records: list[dict]):
        with SessionLocal() as db:
            db.add_all([PredictionLog(**r) for r in records]); db.commit()

    def shadow_score(rows, ids, lm):
        """Runs after the response has been sent. Failures are recorded, never raised."""
        try:
            proba, ms = run_inference(lm, rows)
            out = [ShadowPrediction(request_id=i, model_version=lm.version, probability=float(p),
                                    prediction=int(p >= scfg["threshold"]), latency_ms=round(ms, 3))
                   for i, p in zip(ids, proba)]
        except Exception as e:
            out = [ShadowPrediction(request_id=i, model_version=lm.version, error=f"{type(e).__name__}: {e}"[:300]) for i in ids]
        with SessionLocal() as db:
            db.add_all(out); db.commit()

    def serve_rows(rows: list[dict], bg: BackgroundTasks) -> list[dict]:
        champ, state = require_model(), deploy.get()
        chal = store.get("challenger") if state.mode in ("shadow", "ab") else None
        if chal is not None and chal.version != state.challenger_version:
            chal = None                                    # stale challenger -> safest behaviour is champion only
        ids = [uuid.uuid4().hex for _ in rows]
        arms = ["champion"] * len(rows)
        if state.mode == "ab" and chal is not None:
            arms = [assign_arm(r.get("entity_id") or ids[i], state.traffic_split, dcfg["salt"]) for i, r in enumerate(rows)]
        recs: list[dict | None] = [None] * len(rows)
        for arm in ("champion", "challenger"):
            idx = [i for i, a in enumerate(arms) if a == arm]
            if not idx:
                continue
            lm, role = (chal, "challenger") if arm == "challenger" else (champ, "champion")
            try:
                proba, ms = run_inference(lm, [rows[i] for i in idx])
            except Exception:
                if arm == "champion":
                    raise
                lm, role = champ, "champion"               # challenger failed: fall back, never fail the request
                proba, ms = run_inference(lm, [rows[i] for i in idx])
            for j, i in enumerate(idx):
                recs[i] = make_record(lm, role, rows[i], ids[i], proba[j], ms)
        bg.add_task(persist, recs)
        if state.mode == "shadow" and chal is not None:
            bg.add_task(shadow_score, rows, ids, chal)
        return recs

    def respond(rec: dict) -> PredictionResponse:
        return PredictionResponse(request_id=rec["request_id"], probability=round(rec["probability"], 6),
                                  prediction=rec["prediction"], risk=risk_band(rec["probability"]),
                                  model_name=rec["model_name"], model_version=rec["model_version"],
                                  variant=rec["role"], latency_ms=rec["latency_ms"])

    # ------------------------------------------------------------------- probes
    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready():
        lm = store.get("champion")
        if lm is None:
            raise HTTPException(503, "no champion model loaded")
        return {"status": "ready", "model_version": lm.version}

    # --------------------------------------------------------------- prediction
    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(req: PredictionRequest, bg: BackgroundTasks):
        return respond(serve_rows([req.model_dump()], bg)[0])

    @app.post("/predict/batch", response_model=list[PredictionResponse], tags=["inference"])
    def predict_batch(req: BatchRequest, bg: BackgroundTasks):
        return [respond(r) for r in serve_rows([i.model_dump() for i in req.instances], bg)]

    # ----------------------------------------------------------------- feedback
    def upsert_labels(db, items):
        known = set(db.scalars(select(PredictionLog.request_id).where(PredictionLog.request_id.in_([i.request_id for i in items]))))
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

    # --------------------------------------------------------------- deployment
    def _state_payload():
        st = deploy.get(); chal = store.get("challenger")
        return {**st.to_dict(), "champion_version": (store.get("champion").version if store.get("champion") else None),
                "challenger_loaded_version": chal.version if chal else None}

    @app.get("/deployment", tags=["rollout"])
    def get_deployment():
        return _state_payload()

    @app.post("/deployment", tags=["rollout"], dependencies=[Depends(require_admin)])
    def set_deployment(req: DeploymentRequest):
        require_model()
        if req.mode == "champion_only":
            deploy.set("champion_only", 0.0, None); return _state_payload()
        changed, chal = store.refresh("challenger")
        if chal is None:
            raise HTTPException(409, "No challenger registered. Register a model first (it becomes the `challenger` alias).")
        ok, auc = store.registry.safety_check(chal.version)
        if not ok:
            raise HTTPException(422, f"Challenger v{chal.version} fails the safety floor (golden-set AUC {auc:.3f} < "
                                     f"{dcfg['safety_min_auc']}); it won't be exposed to live traffic.")
        if req.mode == "ab" and req.traffic_split > dcfg["max_ab_split"]:
            raise HTTPException(422, f"traffic_split {req.traffic_split} exceeds the safety cap {dcfg['max_ab_split']}")
        deploy.set(req.mode, req.traffic_split if req.mode == "ab" else 0.0, chal.version)
        return {**_state_payload(), "safety_auc": round(auc, 4)}

    # --------------------------------------------------------------- experiments
    def _analyze():
        st = deploy.get()
        if st.mode == "champion_only":
            raise HTTPException(409, "No experiment is running (mode=champion_only).")
        if st.mode == "shadow":
            df, errs = load_shadow_frame(engine, st.started_at, st.challenger_version)
            return shadow_report(df, cfg, errs), st
        return ab_report(load_ab_frame(engine, st.started_at), cfg, st.traffic_split), st

    @app.get("/experiment/analysis", tags=["rollout"])
    def experiment_analysis():
        rep, st = _analyze()
        return {"state": st.to_dict(), **rep.to_dict()}

    @app.post("/experiment/conclude", tags=["rollout"], dependencies=[Depends(require_admin)])
    def conclude(req: ConcludeRequest):
        rep, st = _analyze()
        v, action = st.challenger_version, req.action
        if action == "auto":
            action = {"promote": "promote", "stop": "stop", "challenger_worse": "stop",
                      "unhealthy": "stop", "invalid_srm": "stop"}.get(rep.verdict, "none")
        if action == "none":
            return {"applied": "none", "verdict": rep.verdict, "recommendation": rep.recommendation}
        if action == "promote":
            try:
                store.registry.promote_version(v, note=f"experiment verdict={rep.verdict}")
            except RegistryError as e:
                raise HTTPException(409, str(e))
        else:
            store.registry.reject_version(v, f"experiment verdict={rep.verdict}")
        deploy.set("champion_only", 0.0, None)
        refresh_all()
        return {"applied": action, "challenger_version": v, "verdict": rep.verdict, **_state_payload()}

    # -------------------------------------------------------------------- admin
    @app.get("/model-info", tags=["ops"])
    def model_info():
        lm = require_model()
        return {"name": lm.name, "version": lm.version, "alias": lm.alias, "algorithm": lm.algorithm,
                "run_id": lm.run_id, "loaded_at": lm.loaded_at, "features": features,
                "test_roc_auc": lm.tags.get("test_roc_auc"), "data_hash": lm.tags.get("data_hash"),
                "threshold": scfg["threshold"]}

    @app.post("/reload", tags=["ops"], dependencies=[Depends(require_admin)])
    def reload():
        res = refresh_all(); ch, lm = res["champion"]
        return {"changed": ch, "model_version": lm.version if lm else None,
                "challenger_version": res["challenger"][1].version if res["challenger"][1] else None}

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
            y = np.array([r.actual for r in lab]); p = np.array([r.probability for r in lab]); pr = np.array([r.prediction for r in lab])
            out["live_quality"] = {"n": len(lab), "accuracy": float((pr == y).mean()),
                                   "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
                                   "mean_predicted": float(p.mean()), "actual_rate": float(y.mean())}
        return out

    @app.get("/predictions/recent", tags=["ops"])
    def recent(limit: int = Query(20, ge=1, le=500), db=Depends(get_db)):
        q = db.scalars(select(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit)).all()
        return [dict(request_id=r.request_id, created_at=r.created_at.isoformat(), model_version=r.model_version,
                     variant=r.role, probability=round(r.probability, 4), prediction=r.prediction,
                     latency_ms=r.latency_ms, features=r.features) for r in q]

    return app
