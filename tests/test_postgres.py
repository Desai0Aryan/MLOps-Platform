"""Runs against a REAL PostgreSQL when TEST_DATABASE_URL is set (CI service container / local docker):

    TEST_DATABASE_URL=postgresql+psycopg2://mlops:mlops@localhost:5432/mlops_test pytest tests/test_postgres.py

Everything else in the suite uses SQLite; these tests prove the Postgres path (JSON columns, advisory locks,
group-by/max queries, pandas read_sql, upserts) behaves identically.
"""
import os, threading
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from mlops.data import generate_dataset
from mlops.experiments import ab_report, load_ab_frame, load_shadow_frame
from mlops.serving import create_app
from mlops.serving.db import Base, GroundTruth, LifecycleEvent, PredictionLog, init_db, make_engine
from mlops.simulate import simulate
from mlops.training import run_training_pipeline
from test_monitoring import make_world

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TEST_DATABASE_URL to run the PostgreSQL tests")
VALID = {"tenure_months": 8, "monthly_charges": 92.5, "total_charges": 740.0, "num_support_calls": 4, "usage_gb": None,
         "age": 29, "contract_type": "month-to-month", "payment_method": "electronic_check", "internet_service": "fiber", "entity_id": "c-1"}


@pytest.fixture()
def pg():
    eng = make_engine(URL); Base.metadata.drop_all(eng); eng.dispose()
    return URL


def test_it_really_is_postgres(pg):
    eng, _ = init_db(pg)
    assert eng.dialect.name == "postgresql"
    with eng.connect() as c:
        assert "PostgreSQL" in c.scalar(text("select version()"))


def test_concurrent_startup_does_not_race_on_create_table(pg):
    """Six replicas booting at the same instant: the advisory lock must serialise CREATE TABLE."""
    errors = []
    def boot():
        try: init_db(pg)
        except Exception as e: errors.append(e)
    ts = [threading.Thread(target=boot) for _ in range(6)]; [t.start() for t in ts]; [t.join() for t in ts]
    assert not errors, errors[:1]


def test_serving_monitoring_and_metrics_on_postgres(cfg, tmp_path, pg):
    c, reg = make_world(cfg, tmp_path); c["serving"]["metrics_cache_seconds"] = 0
    with TestClient(create_app(c, database_url=pg)) as cl:
        r = cl.post("/predict", json=VALID); assert r.status_code == 200
        rid = r.json()["request_id"]
        recent = cl.get("/predictions/recent?limit=1").json()[0]
        assert recent["features"]["usage_gb"] is None and recent["features"]["age"] == 29          # JSON null round-trips
        assert cl.post("/feedback", json={"request_id": rid, "actual": 1}).status_code == 200
        assert cl.post("/feedback", json={"request_id": rid, "actual": 0}).status_code == 200      # upsert (merge), not duplicate
        with cl.app.state.SessionLocal() as db:
            assert db.scalar(select(func.count()).select_from(GroundTruth)) == 1
            assert db.scalar(select(GroundTruth.actual)) == 0

        simulate(cl, scenario="baseline", n=800, label_rate=1.0, seed=10)
        assert cl.get("/stats").json()["total_predictions"] == 801
        assert cl.post("/monitoring/run").json()["status"] == "ok"
        simulate(cl, scenario="market_shift", n=1500, label_rate=0.0, seed=11)
        rep = cl.post("/monitoring/run").json(); assert rep["status"] == "critical"
        assert [h["status"] for h in cl.get("/monitoring/history").json()] == ["critical", "ok"]

        # shadow analysis uses pandas.read_sql + joins on Postgres
        reg.register_run(run_training_pipeline(generate_dataset(5000, "baseline", seed=4), c).run_id)
        assert cl.post("/deployment", json={"mode": "shadow"}).status_code == 200
        simulate(cl, scenario="baseline", n=400, label_rate=1.0, seed=12)
        an = cl.get("/experiment/analysis").json()
        assert an["kind"] == "shadow" and an["stats"]["n_mirrored"] == 400 and an["stats"]["n_labeled"] == 400
        cl.post("/deployment", json={"mode": "ab", "traffic_split": 0.3}); simulate(cl, scenario="baseline", n=600, label_rate=1.0, seed=13)
        assert cl.get("/experiment/analysis").json()["kind"] == "ab"

        # DB-backed Prometheus collector: GROUP BY + MAX(created_at) on Postgres
        body = cl.get("/metrics").text
        assert 'mlops_lifecycle_events_total{kind="shadow_started"} 1.0' in body and 'mlops_lifecycle_events_total{kind="ab_started"} 1.0' in body
        assert 'mlops_monitoring_status 2.0' in body or 'mlops_monitoring_status 0.0' in body
        with cl.app.state.SessionLocal() as db:
            assert [e for e in db.scalars(select(LifecycleEvent.kind).order_by(LifecycleEvent.id))] == ["shadow_started", "ab_started"]
