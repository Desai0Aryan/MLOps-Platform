import copy, numpy as np, pandas as pd, pytest
from fastapi.testclient import TestClient

from mlops.data import generate_dataset, build_reference_profile
from mlops.monitoring import detect_drift, performance_report, psi, render_html
from mlops.orchestrator import Orchestrator
from mlops.registry import ModelRegistry
from mlops.serving import create_app
from mlops.simulate import simulate
from mlops.training import run_training_pipeline


# ------------------------------------------------------------------ pure detectors
@pytest.fixture(scope="module")
def profile(cfg):
    return build_reference_profile(generate_dataset(20000, "baseline", seed=1), cfg)


def test_psi_basic_properties():
    p = np.array([.25, .25, .25, .25])
    assert psi(p, p) == pytest.approx(0, abs=1e-9)
    assert psi(p, np.array([.5, .3, .1, .1])) > 0.25
    assert psi(p, np.array([.3, .25, .25, .2])) < 0.1


@pytest.mark.parametrize("scenario,expected", [("baseline", "ok"), ("mild_drift", "ok"), ("severe_drift", "critical"),
                                               ("market_shift", "critical"), ("concept_drift", "ok")])
def test_drift_status_per_scenario(cfg, profile, scenario, expected):
    rep = detect_drift(generate_dataset(1500, scenario, seed=77), profile, cfg)
    assert rep.status == expected, rep.summary


def test_concept_drift_is_invisible_to_feature_drift(cfg, profile):
    """Same inputs, different outcomes: only label-based monitoring can see this."""
    rep = detect_drift(generate_dataset(1500, "concept_drift", seed=77), profile, cfg)
    assert max(f.psi for f in rep.features) < 0.1


def test_names_the_features_that_moved(cfg, profile):
    rep = detect_drift(generate_dataset(1500, "severe_drift", seed=77), profile, cfg)
    crit = {f.name for f in rep.features if f.severity == "critical"}
    assert {"monthly_charges", "num_support_calls"} <= crit and "tenure_months" not in crit


def test_unseen_category_is_flagged(cfg, profile):
    df = generate_dataset(1500, "baseline", seed=3); df.loc[:200, "payment_method"] = "crypto"
    f = {x.name: x for x in detect_drift(df, profile, cfg).features}["payment_method"]
    assert f.severity != "ok" and "crypto" in f.note


def test_missing_value_spike_is_a_critical_incident(cfg, profile):
    df = generate_dataset(1500, "baseline", seed=3); df.loc[df.sample(frac=0.3, random_state=0).index, "age"] = np.nan
    rep = detect_drift(df, profile, cfg)
    assert rep.status == "critical"


def test_performance_report_states(cfg):
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 2000)
    good = np.clip(0.5 + (y - .5) * .5 + rng.normal(0, .2, 2000), .02, .98)
    base = {"roc_auc": 0.85, "log_loss": 0.50}
    assert performance_report(y, good, base, cfg).status in ("ok", "warning")
    assert performance_report(y, np.clip(good - 0.25, .01, .99), base, cfg).status == "critical"   # miscalibrated
    assert performance_report(y[:50], good[:50], base, cfg).status == "insufficient_data"


def test_html_report_renders(cfg, profile):
    from mlops.monitoring.monitor import MonitoringReport
    d = detect_drift(generate_dataset(1500, "severe_drift", seed=5), profile, cfg)
    rep = MonitoringReport("2026-01-01T00:00:00", 3, "critical", True, ["why <b>"], 1500, 0, d.to_dict(),
                           performance_report([], [], {}, cfg).to_dict()).to_dict()
    html = render_html(rep)
    assert "monthly_charges" in html and "retrain recommended" in html and "<svg" in html and "why &lt;b&gt;" in html  # escaped


# ------------------------------------------------------------ integration helpers
def make_world(cfg, tmp_path, **retrain):
    c = copy.deepcopy(cfg)
    c["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'm.db'}"
    c["training"]["candidates"] = {"logistic_regression": c["training"]["candidates"]["logistic_regression"]}
    c["serving"]["refresh_seconds"] = 0; c["deployment"]["state_cache_seconds"] = 0
    c["deployment"]["experiment"]["min_labeled_per_arm"] = 300
    c["retraining"].update({"cooldown_hours": 0, "max_rows": 2400, **retrain})
    reg = ModelRegistry(c)
    v1 = reg.register_run(run_training_pipeline(generate_dataset(8000, "baseline", seed=1), c).run_id)
    reg.promote_version(v1.version)
    return c, reg


@pytest.fixture()
def world(cfg, tmp_path):
    c, reg = make_world(cfg, tmp_path)
    with TestClient(create_app(c, database_url="sqlite://")) as cl:
        yield c, reg, cl


def kinds(cl):
    return [e["kind"] for e in reversed(cl.get("/events").json())]


# ------------------------------------------------------------------ monitoring API
def test_monitoring_api_detects_data_drift_without_labels(world):
    c, reg, cl = world
    assert cl.get("/monitoring/latest").status_code == 404
    simulate(cl, scenario="baseline", n=800, label_rate=1.0, seed=10)
    assert cl.post("/monitoring/run").json()["status"] == "ok"
    simulate(cl, scenario="market_shift", n=1500, label_rate=0.0, seed=11)
    rep = cl.post("/monitoring/run").json()
    assert rep["status"] == "critical" and rep["trigger_retrain"] and rep["drift"]["status"] == "critical"
    assert len(cl.get("/monitoring/history").json()) == 2 and cl.get("/monitoring/latest").json()["status"] == "critical"
    page = cl.get("/monitoring/report.html")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"] and "retrain recommended" in page.text


def test_concept_drift_is_caught_by_live_performance_only(world):
    c, reg, cl = world
    simulate(cl, scenario="concept_drift", n=1200, label_rate=1.0, seed=21)
    rep = cl.post("/monitoring/run").json()
    assert rep["drift"]["status"] == "ok"                                  # inputs look normal ...
    assert rep["performance"]["status"] == "critical" and rep["trigger_retrain"]   # ... but outcomes changed


# ----------------------------------------------------------------- orchestrator
def test_waits_for_labels_before_retraining(world):
    c, reg, cl = world
    orch = Orchestrator(c, cl.app.state.engine, cl, reg)
    simulate(cl, scenario="market_shift", n=1500, label_rate=0.0, seed=11)
    t = orch.tick()
    assert t["status"] == "critical" and t["action"] == "skipped" and "labeled" in t["reason"]
    assert reg.champion().version == "1" or reg.champion().version == 1 or int(reg.champion().version) == 1
    orch.tick()
    assert kinds(cl).count("retrain_skipped") == 1                        # timeline isn't spammed


def test_full_autonomous_lifecycle(world):
    c, reg, cl = world
    orch = Orchestrator(c, cl.app.state.engine, cl, reg)
    simulate(cl, scenario="baseline", n=800, label_rate=1.0, seed=10); assert orch.tick()["action"] == "none"
    simulate(cl, scenario="market_shift", n=1200, label_rate=0.0, seed=11); assert orch.tick()["action"] == "skipped"
    simulate(cl, scenario="market_shift", n=2200, label_rate=1.0, seed=12)
    t = orch.tick(); assert t["action"] == "retrained_and_shadowing" and cl.get("/deployment").json()["mode"] == "shadow"
    simulate(cl, scenario="market_shift", n=900, label_rate=1.0, seed=13); assert orch.tick()["action"] == "advance:start_ab"
    simulate(cl, scenario="market_shift", n=2500, label_rate=1.0, seed=14); assert orch.tick()["action"] == "advance:promote"
    assert int(reg.champion().version) == 2 and cl.get("/deployment").json()["mode"] == "champion_only"
    simulate(cl, scenario="market_shift", n=1200, label_rate=1.0, seed=15)
    t = orch.tick(); assert t["status"] == "ok" and t["model_version"] == 2     # healthy again, against v2's OWN baseline
    assert kinds(cl) == ["retrain_skipped", "retrain_started", "retrain_completed", "shadow_started", "ab_started", "challenger_promoted"]
    hist = [h["status"] for h in reversed(cl.get("/monitoring/history").json())]
    assert hist[0] == "ok" and "critical" in hist and hist[-1] == "ok"


def test_gate_refuses_to_retrain_on_contract_violating_data(world):
    """8 % missing usage_gb breaks the data contract: that's a data incident, not something to learn from."""
    c, reg, cl = world
    orch = Orchestrator(c, cl.app.state.engine, cl, reg)
    simulate(cl, scenario="baseline", n=800, label_rate=1.0, seed=10)
    simulate(cl, scenario="severe_drift", n=2200, label_rate=1.0, seed=12)
    t = orch.tick()
    assert t["action"] == "rejected" and "validation" in t["reason"]
    assert int(reg.champion().version) == 1 and cl.get("/deployment").json()["mode"] == "champion_only"
    assert "retrain_rejected_bad_data" in kinds(cl)


def test_cooldown_prevents_retrain_thrashing(cfg, tmp_path):
    c, reg = make_world(cfg, tmp_path, cooldown_hours=24)
    with TestClient(create_app(c, database_url="sqlite://")) as cl:
        orch = Orchestrator(c, cl.app.state.engine, cl, reg)
        simulate(cl, scenario="baseline", n=800, label_rate=1.0, seed=10)
        simulate(cl, scenario="severe_drift", n=2200, label_rate=1.0, seed=12)
        assert orch.tick()["action"] == "rejected"            # first attempt starts the cooldown clock
        t = orch.tick()
        assert t["action"] == "skipped" and "cooldown" in t["reason"]


def test_inconclusive_experiment_times_out(cfg, tmp_path):
    c, reg = make_world(cfg, tmp_path)
    c["deployment"]["experiment"]["max_duration_hours"] = 0
    with TestClient(create_app(c, database_url="sqlite://")) as cl:
        v2 = reg.register_run(run_training_pipeline(generate_dataset(6000, "baseline", seed=9), c).run_id)
        assert cl.post("/deployment", json={"mode": "shadow"}).status_code == 200
        out = cl.post("/experiment/advance").json()
        assert out["action"] == "stop" and out["reason"] == "timeout"
        assert cl.get("/deployment").json()["mode"] == "champion_only"
        assert reg.get_version(v2.version).tags["status"] == "rejected"
