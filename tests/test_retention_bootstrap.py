import copy, datetime as dt, pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from mlops.bootstrap import bootstrap, wait_for_mlflow
from mlops.registry import ModelRegistry
from mlops.retention import prune_old_data
from mlops.serving.db import GroundTruth, MonitoringRun, PredictionLog, ShadowPrediction, init_db, utcnow


def _n(db, model): return db.scalar(select(func.count()).select_from(model))


def test_retention_prunes_old_rows_and_keeps_recent_ones(cfg):
    engine, Session = init_db("sqlite://")
    now = utcnow(); old, fresh = now - dt.timedelta(days=200), now - dt.timedelta(days=1)
    with Session() as db:
        for i, ts in enumerate([old, old, fresh]):
            db.add(PredictionLog(request_id=f"r{i}", created_at=ts, model_name="m", model_version=1, features={}, probability=.5, prediction=0, latency_ms=1))
            db.add(ShadowPrediction(request_id=f"r{i}", created_at=ts, model_version=2, probability=.5, prediction=0, latency_ms=1))
            db.add(MonitoringRun(created_at=ts, model_version=1, status="ok", drift_status="ok", perf_status="ok", n_rows=1, n_labeled=0, report={}))
        db.flush(); db.add_all([GroundTruth(request_id=f"r{i}", actual=1) for i in range(3)]); db.commit()
    out = prune_old_data(engine, cfg)
    assert out == {"monitoring_runs": 2, "shadow_predictions": 2, "labels": 2, "predictions": 2}
    with Session() as db:
        assert [_n(db, m) for m in (PredictionLog, GroundTruth, ShadowPrediction, MonitoringRun)] == [1, 1, 1, 1]
        assert db.scalar(select(PredictionLog.request_id)) == "r2" and db.scalar(select(GroundTruth.request_id)) == "r2"   # label of the kept row survives


def test_retention_refuses_a_policy_that_would_starve_retraining(cfg):
    bad = copy.deepcopy(cfg); bad["retention"]["predictions_days"] = 5
    engine, _ = init_db("sqlite://")
    with pytest.raises(AssertionError):
        prune_old_data(engine, bad)


def test_bootstrap_is_idempotent(cfg, tmp_path):
    c = copy.deepcopy(cfg); c["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'b.db'}"
    c["training"]["candidates"] = {"logistic_regression": c["training"]["candidates"]["logistic_regression"]}
    first = bootstrap(c, rows=3000, log=lambda m: None)
    assert first["action"] == "created" and int(ModelRegistry(c).champion().version) == 1
    assert bootstrap(c, rows=3000, log=lambda m: None) == {"action": "exists", "version": 1}
    assert len(ModelRegistry(c).list_versions()) == 1                       # second run trained nothing


def test_wait_for_mlflow_behaviour():
    wait_for_mlflow("sqlite:///x.db", timeout=0)                           # local stores need no waiting
    with pytest.raises(TimeoutError):
        wait_for_mlflow("http://127.0.0.1:9", timeout=1, log=lambda m: None)   # nothing listens on port 9
