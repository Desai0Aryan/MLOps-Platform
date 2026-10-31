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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score
from sqlalchemy import func, select

from ..config import feature_columns, load_config
from ..experiments import ab_report, load_ab_frame, load_shadow_frame, shadow_report
from ..registry import RegistryError
from ..monitoring import Monitor, render_html
from .db import GroundTruth, LifecycleEvent, PredictionLog, ShadowPrediction, init_db, log_event, utcnow
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
    monitor = Monitor(cfg, engine, store.registry)

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
        mon_task = None
        interval = int(os.getenv("MONITORING_INTERVAL_SECONDS", cfg["monitoring"]["interval_seconds"]))
        if interval > 0:
            async def mon_loop():
                while True:
                    await asyncio.sleep(interval)
                    try:
                        await asyncio.to_thread(monitor.run)
                    except Exception:
                        pass
            mon_task = asyncio.create_task(mon_loop())
        yield
        for t in (task, mon_task):
            if t:
                t.cancel()

    app = FastAPI(title="Churn MLOps Serving API", version="0.5.0", lifespan=lifespan,
                  description="Versioned model serving with shadow deployment, A/B testing, "
                              "statistical decision-making and zero-downtime rollouts.")
    app.state.store, app.state.SessionLocal, app.state.cfg = store, SessionLocal, cfg
    app.state.engine, app.state.deploy, app.state.monitor = engine, deploy, monitor

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
            if deploy.get().mode != "champion_only":
                log_event(SessionLocal, "experiment_stopped_manually", {}, deploy.get().challenger_version)
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
        log_event(SessionLocal, f"{req.mode}_started", {"traffic_split": req.traffic_split if req.mode == "ab" else 0.0,
                                                         "safety_auc": round(auc, 4)}, chal.version)
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

    def apply_action(action: str, v: int, verdict: str, reason: str | None = None):
        if action == "promote":
            try:
                store.registry.promote_version(v, note=f"experiment verdict={verdict}")
            except RegistryError as e:
                raise HTTPException(409, str(e))
            log_event(SessionLocal, "challenger_promoted", {"verdict": verdict}, v)
        else:
            store.registry.reject_version(v, reason or f"experiment verdict={verdict}")
            log_event(SessionLocal, "challenger_rejected", {"verdict": verdict, "reason": reason}, v)
        deploy.set("champion_only", 0.0, None)
        refresh_all()

    @app.post("/experiment/conclude", tags=["rollout"], dependencies=[Depends(require_admin)])
    def conclude(req: ConcludeRequest):
        rep, st = _analyze()
        v, action = st.challenger_version, req.action
        if action == "auto":
            action = {"promote": "promote", "stop": "stop", "challenger_worse": "stop",
                      "unhealthy": "stop", "invalid_srm": "stop"}.get(rep.verdict, "none")
        if action == "none":
            return {"applied": "none", "verdict": rep.verdict, "recommendation": rep.recommendation}
        apply_action(action, v, rep.verdict)
        return {"applied": action, "challenger_version": v, "verdict": rep.verdict, **_state_payload()}

    @app.post("/experiment/advance", tags=["rollout"], dependencies=[Depends(require_admin)])
    def advance():
        """One step of the rollout state machine: shadow -> A/B -> promote, or stop. Safe to call on a timer."""
        st = deploy.get()
        if st.mode == "champion_only":
            return {"action": "none", "reason": "no experiment running"}
        rep, st = _analyze(); v = st.challenger_version
        age_h = (utcnow() - st.started_at).total_seconds() / 3600
        if st.mode == "shadow":
            if rep.verdict == "challenger_better":
                deploy.set("ab", dcfg["auto_ab_split"], v)
                log_event(SessionLocal, "ab_started", {"traffic_split": dcfg["auto_ab_split"], "trigger": "shadow verdict"}, v)
                return {"action": "start_ab", "verdict": rep.verdict, "traffic_split": dcfg["auto_ab_split"]}
            if rep.verdict in ("challenger_worse", "unhealthy"):
                apply_action("stop", v, rep.verdict); return {"action": "stop", "verdict": rep.verdict}
        else:
            if rep.verdict == "promote":
                apply_action("promote", v, rep.verdict); return {"action": "promote", "verdict": rep.verdict, "challenger_version": v}
            if rep.verdict in ("stop", "invalid_srm"):
                apply_action("stop", v, rep.verdict); return {"action": "stop", "verdict": rep.verdict}
        if age_h >= dcfg["experiment"]["max_duration_hours"]:
            apply_action("stop", v, rep.verdict, reason=f"timed out after {age_h:.0f}h without a decisive result")
            return {"action": "stop", "verdict": rep.verdict, "reason": "timeout"}
        return {"action": "wait", "verdict": rep.verdict, "recommendation": rep.recommendation,
                "progress": rep.stats.get("progress")}

    # --------------------------------------------------------------- monitoring
    @app.post("/monitoring/run", tags=["monitoring"])
    def monitoring_run():
        require_model()
        return monitor.run().to_dict()

    @app.get("/monitoring/latest", tags=["monitoring"])
    def monitoring_latest():
        rep = monitor.latest()
        if rep is None:
            raise HTTPException(404, "no monitoring run yet; POST /monitoring/run")
        return rep

    @app.get("/monitoring/history", tags=["monitoring"])
    def monitoring_history(limit: int = Query(50, ge=1, le=500)):
        return monitor.history(limit)

    @app.get("/monitoring/report.html", response_class=HTMLResponse, tags=["monitoring"])
    def monitoring_report_html(refresh: bool = False):
        rep = None if refresh else monitor.latest()
        if rep is None:
            require_model(); rep = monitor.run().to_dict()
        return HTMLResponse(render_html(rep))

    @app.get("/events", tags=["monitoring"])
    def events(limit: int = Query(50, ge=1, le=500), db=Depends(get_db)):
        q = db.scalars(select(LifecycleEvent).order_by(LifecycleEvent.id.desc()).limit(limit)).all()
        return [dict(id=e.id, created_at=e.created_at.isoformat(), kind=e.kind, model_version=e.model_version, detail=e.detail) for e in q]

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
