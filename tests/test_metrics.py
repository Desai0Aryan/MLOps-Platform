import copy, glob, importlib.util, json, re, shutil, subprocess, sys, time
from pathlib import Path
import pytest, yaml
from fastapi.testclient import TestClient

from mlops.data import generate_dataset
from mlops.serving import create_app
from mlops.simulate import simulate
from mlops.training import run_training_pipeline
from test_monitoring import make_world

ROOT = Path(__file__).resolve().parents[1]
VALID = {"tenure_months": 8, "monthly_charges": 92.5, "total_charges": 740.0, "num_support_calls": 4, "usage_gb": 130.2,
         "age": 29, "contract_type": "month-to-month", "payment_method": "electronic_check", "internet_service": "fiber"}


@pytest.fixture()
def world(cfg, tmp_path):
    c, reg = make_world(cfg, tmp_path)
    c["serving"]["metrics_cache_seconds"] = 0                     # always read fresh state in tests
    with TestClient(create_app(c, database_url="sqlite://")) as cl:
        yield c, reg, cl


def val(cl, name, **labels):
    return cl.app.state.metrics.registry.get_sample_value(name, labels)


# ----------------------------------------------------------------- in-process instruments
def test_metrics_endpoint_is_prometheus_text(world):
    _, _, cl = world
    r = cl.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert "# TYPE mlops_predictions_total counter" in r.text and "mlops_model_info{" in r.text


def test_prediction_counters_are_exact(world):
    _, _, cl = world
    cl.post("/predict/batch", json={"instances": [VALID] * 7}); cl.post("/predict", json=VALID)
    assert val(cl, "mlops_predictions_total", model_version="1", variant="champion") == 8
    assert val(cl, "mlops_prediction_probability_count", model_version="1") == 8
    assert val(cl, "mlops_prediction_probability_bucket", model_version="1", le="+Inf") == 8
    assert val(cl, "mlops_inference_latency_seconds_count", variant="champion") == 2        # per call, not per row
    assert val(cl, "mlops_http_requests_total", method="POST", path="/predict/batch", status="200") == 1


def test_http_labels_use_route_templates_not_raw_urls(world):
    _, _, cl = world
    cl.get("/nope/abc-12345"); cl.post("/predict", json={**VALID, "age": 999}); cl.get("/metrics"); cl.get("/metrics")
    body = cl.get("/metrics").text
    assert "abc-12345" not in body                                                          # no cardinality explosion
    assert val(cl, "mlops_http_requests_total", method="GET", path="unmatched", status="404") == 1
    assert val(cl, "mlops_http_requests_total", method="POST", path="/predict", status="422") == 1
    assert "/metrics" not in "".join(l for l in body.splitlines() if l.startswith("mlops_http_requests_total"))


def test_feedback_counter(world):
    _, _, cl = world
    simulate(cl, scenario="baseline", n=200, label_rate=1.0, seed=3)
    assert val(cl, "mlops_feedback_labels_total") == 200


def test_each_app_has_an_isolated_registry(world, cfg, tmp_path):
    c, reg, cl = world
    with TestClient(create_app(c, database_url="sqlite://")) as other:
        cl.post("/predict", json=VALID)
        assert val(cl, "mlops_predictions_total", model_version="1", variant="champion") == 1
        assert val(other, "mlops_predictions_total", model_version="1", variant="champion") is None


# ------------------------------------------------------------ platform-state collector
def test_collector_exports_drift_and_status_from_the_database(world):
    _, _, cl = world
    simulate(cl, scenario="baseline", n=800, label_rate=1.0, seed=10)
    cl.post("/monitoring/run"); assert val(cl, "mlops_monitoring_status") == 0
    simulate(cl, scenario="market_shift", n=1500, label_rate=0.0, seed=11)
    cl.post("/monitoring/run")
    assert val(cl, "mlops_monitoring_status") == 2 and val(cl, "mlops_drift_status") == 2
    assert val(cl, "mlops_drift_psi", feature="monthly_charges") > 0.25
    assert val(cl, "mlops_drift_psi", feature="predicted_probability") > 0.25                 # prediction drift is exported too
    assert val(cl, "mlops_drift_feature_severity", feature="monthly_charges") == 2
    assert val(cl, "mlops_drift_feature_severity", feature="tenure_months") == 0
    assert val(cl, "mlops_performance_baseline", metric="roc_auc") > 0.7 and val(cl, "mlops_monitoring_model_version") == 1
    assert time.time() - val(cl, "mlops_monitoring_last_run_timestamp_seconds") < 60


def test_collector_tracks_rollout_state_and_lifecycle(world):
    c, reg, cl = world
    from mlops.registry import ModelRegistry
    reg.register_run(run_training_pipeline(generate_dataset(5000, "baseline", seed=4), c).run_id)
    assert val(cl, "mlops_deployment_mode", mode="champion_only") == 1
    assert cl.post("/deployment", json={"mode": "ab", "traffic_split": 0.25}).status_code == 200
    assert (val(cl, "mlops_deployment_mode", mode="ab"), val(cl, "mlops_deployment_mode", mode="champion_only")) == (1, 0)
    assert val(cl, "mlops_ab_traffic_split") == 0.25 and val(cl, "mlops_model_version", role="challenger") == 2
    assert val(cl, "mlops_lifecycle_events_total", kind="ab_started") == 1
    assert time.time() - val(cl, "mlops_lifecycle_last_event_timestamp_seconds", kind="ab_started") < 60


def test_challenger_failure_is_counted(world):
    c, reg, cl = world
    reg.register_run(run_training_pipeline(generate_dataset(5000, "baseline", seed=4), c).run_id)
    cl.post("/deployment", json={"mode": "ab", "traffic_split": 0.5})
    class Boom:
        def predict_proba(self, X): raise RuntimeError("x")
    cl.app.state.store.get("challenger").model = Boom()
    for i in range(30):
        assert cl.post("/predict", json={**VALID, "entity_id": f"c{i}"}).status_code == 200
    assert val(cl, "mlops_prediction_errors_total", stage="challenger_fallback") > 0
    assert val(cl, "mlops_predictions_total", model_version="1", variant="champion") == 30


def test_scrape_survives_a_broken_database(world):
    _, _, cl = world
    cl.app.state.monitor.latest = lambda: (_ for _ in ()).throw(RuntimeError("db down"))
    assert cl.get("/metrics").status_code == 200                       # observability must never take the service down


# ------------------------------------------------------ dashboards + alerts as code
def _exported_names(cl) -> set:
    names = set()
    for fam in cl.app.state.metrics.registry.collect():
        names.add(fam.name)
        for s in fam.samples:
            names.add(s.name)
        if fam.type == "counter": names.add(fam.name + "_total")
        if fam.type == "histogram": names |= {fam.name + x for x in ("_bucket", "_sum", "_count")}
    return names


def _dashboards():
    return {Path(f).name: json.load(open(f)) for f in sorted(glob.glob(str(ROOT / "deploy/grafana/dashboards/*.json")))}


def _exprs(d):
    return [t["expr"] for p in d["panels"] for t in p.get("targets", [])] + [a["expr"] for a in d["annotations"]["list"] if "expr" in a]


def test_dashboards_are_in_sync_with_the_generator():
    spec = importlib.util.spec_from_file_location("bd", ROOT / "scripts/build_dashboards.py"); bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)
    for make in (bd.service_board, bd.quality_board, bd.rollout_board):
        b = make(); assert json.load(open(ROOT / "deploy/grafana/dashboards" / f"{b.d['uid']}.json")) == b.d, f"{b.d['uid']} is stale: run scripts/build_dashboards.py"


def test_dashboard_structure_is_sound():
    for name, d in _dashboards().items():
        ids = [p["id"] for p in d["panels"]]; assert len(ids) == len(set(ids)), name
        boxes = [(p["gridPos"]["x"], p["gridPos"]["y"], p["gridPos"]["w"], p["gridPos"]["h"]) for p in d["panels"]]
        for i, a in enumerate(boxes):
            assert a[0] + a[2] <= 24, f"{name}: panel overflows the 24-column grid"
            for b in boxes[i + 1:]:
                assert not (a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]), f"{name}: overlapping panels"
        for p in d["panels"]:
            if p["type"] != "row":
                assert p["datasource"]["uid"] == "prometheus" and p["targets"], (name, p["title"])


def test_every_metric_the_dashboards_and_alerts_use_is_really_exported(world):
    _, _, cl = world
    exported = _exported_names(cl) | {"up", "time"}
    exprs = [e for d in _dashboards().values() for e in _exprs(d)]
    for g in yaml.safe_load(open(ROOT / "deploy/prometheus/alerts.yml"))["groups"]:
        exprs += [r["expr"] for r in g["rules"]]
    used = {m for e in exprs for m in re.findall(r"\b((?:mlops|process)_[a-z0-9_]+)", e)}
    assert used, "no metrics found: regex broken"
    assert used <= exported, f"queries reference metrics the API never emits: {sorted(used - exported)}"


def test_alert_rules_are_complete():
    rules = [r for g in yaml.safe_load(open(ROOT / "deploy/prometheus/alerts.yml"))["groups"] for r in g["rules"]]
    assert len(rules) >= 12 and len({r["alert"] for r in rules}) == len(rules)
    for r in rules:
        assert r["labels"]["severity"] in {"critical", "warning", "info"} and r["annotations"]["summary"] and len(r["annotations"]["description"]) > 30, r["alert"]


@pytest.mark.skipif(not shutil.which("promtool"), reason="promtool not installed")
def test_promtool_accepts_everything():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/validate_observability.py"), "--require"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
