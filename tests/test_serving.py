import copy, pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from mlops.data import generate_dataset
from mlops.registry import ModelRegistry, predict_proba
from mlops.serving import create_app
from mlops.serving.db import PredictionLog
from mlops.simulate import simulate
from mlops.training import run_training_pipeline

VALID = {"tenure_months": 8, "monthly_charges": 92.5, "total_charges": 740.0, "num_support_calls": 4,
         "usage_gb": 130.2, "age": 29, "contract_type": "month-to-month",
         "payment_method": "electronic_check", "internet_service": "fiber", "entity_id": "c-1"}


@pytest.fixture(scope="module")
def env(cfg, tmp_path_factory):
    c = copy.deepcopy(cfg); d = tmp_path_factory.mktemp("srv")
    c["mlflow"]["tracking_uri"] = f"sqlite:///{d / 'm.db'}"
    c["training"]["candidates"] = {"logistic_regression": c["training"]["candidates"]["logistic_regression"]}
    c["serving"]["refresh_seconds"] = 0
    reg = ModelRegistry(c)
    v1 = reg.register_run(run_training_pipeline(generate_dataset(2500, "baseline", seed=1), c).run_id)
    reg.promote_if_better(v1.version)
    return c, reg, d


@pytest.fixture()
def client(env):
    c, reg, d = env
    app = create_app(c, database_url="sqlite://")           # fresh in-memory log per test
    with TestClient(app) as cl:
        cl.app = app
        yield cl


def test_probes_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200
    info = client.get("/model-info").json()
    assert info["version"] == 1 and "tenure_months" in info["features"]


def test_predict_matches_direct_model_and_is_logged(client, env):
    c, reg, _ = env
    r = client.post("/predict", json=VALID); assert r.status_code == 200
    body = r.json()
    import pandas as pd
    direct = predict_proba(reg.load_model("champion"), pd.DataFrame([{k: v for k, v in VALID.items() if k != "entity_id"}]), c)[0]
    assert abs(body["probability"] - direct) < 1e-5
    assert body["risk"] in {"low", "medium", "high"} and "X-Process-Time-ms" in r.headers
    with client.app.state.SessionLocal() as db:               # background task ran -> row exists
        row = db.scalar(select(PredictionLog).where(PredictionLog.request_id == body["request_id"]))
        assert row and row.model_version == 1 and row.entity_id == "c-1" and row.features["age"] == 29


@pytest.mark.parametrize("patch", [{"age": 500}, {"contract_type": "lifetime"}, {"monthly_charges": -5},
                                   {"surprise_field": 1}, {"age": "old"}])
def test_invalid_requests_rejected(client, patch):
    assert client.post("/predict", json={**VALID, **patch}).status_code == 422


def test_missing_required_and_nullable_fields(client):
    bad = {k: v for k, v in VALID.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", json={**VALID, "usage_gb": None}).status_code == 200   # contract allows nulls


def test_batch_limits(client, env):
    c, *_ = env
    assert len(client.post("/predict/batch", json={"instances": [VALID] * 5}).json()) == 5
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422
    assert client.post("/predict/batch", json={"instances": [VALID] * (c["serving"]["max_batch"] + 1)}).status_code == 422


def test_feedback_flow_and_live_quality(client):
    sm = simulate(client, scenario="baseline", n=400, label_rate=1.0, seed=7)
    assert sm.sent == 400 and sm.labeled == 400 and sm.errors == 0
    st = client.get("/stats").json()
    assert st["total_predictions"] == 400 and st["labeled"] == 400
    assert 0.6 < st["live_quality"]["roc_auc"] < 0.95 and st["by_version"] == {"1": 400}
    rid = client.get("/predictions/recent?limit=1").json()[0]["request_id"]
    assert client.post("/feedback", json={"request_id": rid, "actual": 1}).status_code == 200     # relabel = upsert
    assert client.post("/feedback", json={"request_id": "nope", "actual": 1}).status_code == 404
    assert client.post("/feedback", json={"request_id": rid, "actual": 7}).status_code == 422


def test_hot_reload_swaps_model_without_restart(client, env):
    c, reg, _ = env
    r0 = client.post("/reload").json(); assert (r0["changed"], r0["model_version"]) == (False, 1)
    v2 = reg.register_run(run_training_pipeline(generate_dataset(9000, "baseline", seed=2), c).run_id)
    reg.promote_if_better(v2.version, force=True)   # this test is about hot reload, not the policy
    r1 = client.post("/reload").json(); assert (r1["changed"], r1["model_version"]) == (True, 2)
    assert client.post("/predict", json=VALID).json()["model_version"] == 2
    reg.rollback("test cleanup")                                       # leave registry as we found it
    assert client.post("/reload").json()["model_version"] == 1


def test_503_when_no_champion(cfg, tmp_path):
    c = copy.deepcopy(cfg); c["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'empty.db'}"; c["serving"]["refresh_seconds"] = 0
    with TestClient(create_app(c, database_url="sqlite://")) as cl:
        assert cl.get("/health").status_code == 200
        assert cl.get("/ready").status_code == 503
        assert cl.post("/predict", json=VALID).status_code == 503


def test_feedback_tolerates_the_background_persist_race(env):
    """Predictions are persisted AFTER the response is sent. A client that replies with feedback instantly must not get a
    spurious 'unknown request_id' (found by the post-deploy smoke test; TestClient hides it by running background tasks early)."""
    import threading, time as _t
    from mlops.serving.db import PredictionLog
    c, reg, d = env
    c2 = copy.deepcopy(c); c2["serving"]["feedback_grace_seconds"] = 1.5
    with TestClient(create_app(c2, database_url="sqlite://")) as cl:
        def late_insert():
            with cl.app.state.SessionLocal() as db:
                db.add(PredictionLog(request_id="late-1", model_name="m", model_version=1, features={}, probability=.5, prediction=0, latency_ms=1)); db.commit()
        threading.Timer(0.3, late_insert).start()
        t = _t.monotonic(); r = cl.post("/feedback", json={"request_id": "late-1", "actual": 1}); waited = _t.monotonic() - t
        assert r.status_code == 200 and 0.2 < waited < 1.4                       # waited for the row, but not the full grace period
        t = _t.monotonic(); r = cl.post("/feedback", json={"request_id": "never-existed", "actual": 1})
        assert r.status_code == 404 and _t.monotonic() - t >= 1.4                # a genuinely unknown id still fails, after the grace
        b = cl.post("/feedback/batch", json={"items": [{"request_id": "late-1", "actual": 0}, {"request_id": "nope", "actual": 0}]}).json()
        assert b["stored"] == 1 and b["n_unknown"] == 1
