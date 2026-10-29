import copy, numpy as np, pandas as pd, pytest
from fastapi.testclient import TestClient

from mlops.data import generate_dataset
from mlops.experiments import ab_report, shadow_report
from mlops.registry import ModelRegistry
from mlops.serving import create_app
from mlops.serving.deployment import assign_arm
from mlops.simulate import simulate
from mlops.training import run_training_pipeline


# --------------------------------------------------------------------- pure logic
def test_assignment_is_sticky_and_respects_split():
    keys = [f"cust-{i}" for i in range(20000)]
    arms = [assign_arm(k, 0.2, "s") for k in keys]
    assert arms == [assign_arm(k, 0.2, "s") for k in keys]                     # deterministic
    assert abs(arms.count("challenger") / len(arms) - 0.2) < 0.01              # ~20 %
    assert arms != [assign_arm(k, 0.2, "other-salt") for k in keys]            # salt reshuffles


def _frame_ab(n_c, n_h, split_bias=0.0, h_noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    def arm(name, n, noise):
        y = rng.integers(0, 2, n); p = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.2 + noise, n), 0.02, 0.98)
        return pd.DataFrame({"arm": name, "probability": p, "prediction": (p >= .5).astype(int),
                             "latency_ms": rng.gamma(2, 1, n), "actual": y})
    return pd.concat([arm("champion", n_c, 0.25), arm("challenger", n_h, h_noise)], ignore_index=True)


def test_ab_promotes_clearly_better_challenger(cfg):
    r = ab_report(_frame_ab(2000, 500, h_noise=-0.15), cfg, expected_split=0.2)
    assert r.verdict == "promote", r.checks


def test_ab_stops_clearly_worse_challenger(cfg):
    r = ab_report(_frame_ab(2000, 500, h_noise=0.5, seed=3), cfg, expected_split=0.2)
    assert r.verdict == "stop"


def test_ab_waits_for_enough_samples(cfg):
    r = ab_report(_frame_ab(400, 80), cfg, expected_split=0.2)
    assert r.verdict == "continue" and r.stats["progress"] < 1


def test_ab_detects_sample_ratio_mismatch(cfg):
    r = ab_report(_frame_ab(1000, 1000), cfg, expected_split=0.1)            # 50/50 observed, 10 % expected
    assert r.verdict == "invalid_srm"


def test_ab_latency_guardrail_holds_promotion(cfg):
    df = _frame_ab(2000, 500, h_noise=-0.15); df.loc[df.arm == "challenger", "latency_ms"] += 300
    assert ab_report(df, cfg, 0.2).verdict == "hold"


def test_shadow_paired_verdicts(cfg):
    rng = np.random.default_rng(1); n = 1500; y = rng.integers(0, 2, n)
    base = np.clip(0.5 + (y - .5) * .4 + rng.normal(0, .25, n), .02, .98)
    better = np.clip(0.5 + (y - .5) * .6 + rng.normal(0, .15, n), .02, .98)
    mk = lambda pc, ph: pd.DataFrame({"p_champ": pc, "p_chal": ph, "pred_champ": (pc >= .5).astype(int),
                                      "pred_chal": (ph >= .5).astype(int), "lat_champ": 1.0, "lat_chal": 1.0, "actual": y})
    assert shadow_report(mk(base, better), cfg).verdict == "challenger_better"
    assert shadow_report(mk(better, base), cfg).verdict == "challenger_worse"
    assert shadow_report(mk(base, base.copy()), cfg).verdict == "no_difference"
    assert shadow_report(mk(base, better), cfg, n_errors=500).verdict == "unhealthy"


# ------------------------------------------------------------------ API integration
@pytest.fixture(scope="module")
def world(cfg, tmp_path_factory):
    c = copy.deepcopy(cfg); d = tmp_path_factory.mktemp("roll")
    c["mlflow"]["tracking_uri"] = f"sqlite:///{d / 'm.db'}"
    c["training"]["candidates"] = {"logistic_regression": c["training"]["candidates"]["logistic_regression"]}
    c["serving"]["refresh_seconds"] = 0; c["deployment"]["state_cache_seconds"] = 0
    c["deployment"]["experiment"]["min_labeled_per_arm"] = 150
    reg = ModelRegistry(c)
    v1 = reg.register_run(run_training_pipeline(generate_dataset(5000, "baseline", seed=1), c).run_id)
    reg.promote_version(v1.version)
    v2 = reg.register_run(run_training_pipeline(generate_dataset(5000, "concept_drift", seed=2), c).run_id)  # fresher
    return c, reg, int(v1.version), int(v2.version)


@pytest.fixture()
def api(world):
    c, reg, v1, v2 = world
    cl = reg.client; n = reg.name                                  # reset registry to: v1 champion, v2 challenger
    cl.set_registered_model_alias(n, "champion", v1); cl.set_registered_model_alias(n, "challenger", v2)
    for v, s in ((v1, "champion"), (v2, "candidate")):
        cl.set_model_version_tag(n, v, "status", s)
    cl.delete_model_version_tag(n, v1, "replaced_version") if "replaced_version" in reg.get_version(v1).tags else None
    with TestClient(create_app(c, database_url="sqlite://")) as client:
        yield client, v1, v2


def test_cannot_start_experiment_without_challenger(world, cfg):
    c, reg, v1, v2 = world
    reg.client.delete_registered_model_alias(reg.name, "challenger")
    with TestClient(create_app(c, database_url="sqlite://")) as cl:
        assert cl.post("/deployment", json={"mode": "shadow"}).status_code == 409
    reg.client.set_registered_model_alias(reg.name, "challenger", v2)


def test_shadow_never_changes_the_response_but_logs_challenger(api):
    cl, v1, v2 = api
    assert cl.post("/deployment", json={"mode": "shadow"}).status_code == 200
    out = simulate(cl, scenario="concept_drift", n=300, label_rate=1.0, seed=5)
    assert out.versions == {v1: 300} and out.variants == {"champion": 300}      # callers only ever saw v1
    an = cl.get("/experiment/analysis").json()
    assert an["kind"] == "shadow" and an["stats"]["n_mirrored"] == 300 and an["stats"]["n_labeled"] == 300


def test_shadow_to_ab_to_promotion_on_drifted_traffic(api):
    cl, v1, v2 = api
    cl.post("/deployment", json={"mode": "shadow"})
    simulate(cl, scenario="concept_drift", n=600, label_rate=1.0, seed=6)
    assert cl.get("/experiment/analysis").json()["verdict"] == "challenger_better"
    assert cl.post("/deployment", json={"mode": "ab", "traffic_split": 0.3}).json()["mode"] == "ab"
    sm = simulate(cl, scenario="concept_drift", n=1500, label_rate=1.0, seed=7)
    assert set(sm.variants) == {"champion", "challenger"} and abs(sm.variants["challenger"] / 1500 - 0.3) < 0.05
    an = cl.get("/experiment/analysis").json()
    assert an["verdict"] == "promote" and an["stats"]["srm_p_value"] > 0.001
    res = cl.post("/experiment/conclude", json={"action": "auto"}).json()
    assert res["applied"] == "promote" and res["mode"] == "champion_only" and res["champion_version"] == v2
    assert cl.post("/predict", json=_VALID).json()["model_version"] == v2          # traffic now served by v2


_VALID = {"tenure_months": 8, "monthly_charges": 92.5, "total_charges": 740.0, "num_support_calls": 4,
          "usage_gb": 130.2, "age": 29, "contract_type": "month-to-month", "payment_method": "electronic_check",
          "internet_service": "fiber"}


def test_ab_assignment_is_sticky_per_customer(api):
    cl, v1, v2 = api
    cl.post("/deployment", json={"mode": "ab", "traffic_split": 0.5})
    seen = {}
    for rnd in range(3):
        for i in range(60):
            v = cl.post("/predict", json={**_VALID, "entity_id": f"cust-{i}"}).json()["variant"]
            assert seen.setdefault(i, v) == v


def test_challenger_failure_falls_back_to_champion(api):
    cl, v1, v2 = api
    cl.post("/deployment", json={"mode": "ab", "traffic_split": 0.5})
    class Boom:
        def predict_proba(self, X): raise RuntimeError("model exploded")
    cl.app.state.store.get("challenger").model = Boom()
    outs = [cl.post("/predict", json={**_VALID, "entity_id": f"c{i}"}) for i in range(40)]
    assert all(o.status_code == 200 for o in outs)
    assert {o.json()["variant"] for o in outs} == {"champion"}                    # every request still answered


def test_shadow_errors_are_recorded_not_raised(api):
    cl, v1, v2 = api
    cl.post("/deployment", json={"mode": "shadow"})
    class Boom:
        def predict_proba(self, X): raise RuntimeError("boom")
    cl.app.state.store.get("challenger").model = Boom()
    assert cl.post("/predict", json=_VALID).status_code == 200
    simulate(cl, scenario="baseline", n=100, label_rate=1.0, seed=2)
    an = cl.get("/experiment/analysis").json()
    assert an["stats"]["n_shadow_errors"] >= 100 and an["verdict"] == "unhealthy"
    assert cl.post("/experiment/conclude", json={"action": "auto"}).json()["applied"] == "stop"


def test_guard_rails_on_deployment_requests(api, world):
    cl, v1, v2 = api
    assert cl.post("/deployment", json={"mode": "ab", "traffic_split": 0.9}).status_code == 422     # > max_ab_split
    assert cl.post("/deployment", json={"mode": "ab", "traffic_split": 0}).status_code == 422
    assert cl.get("/experiment/analysis").status_code == 409                                         # nothing running


def test_safety_floor_blocks_bad_challenger(world):
    c, reg, v1, v2 = world
    strict = copy.deepcopy(c); strict["deployment"]["safety_min_auc"] = 0.99
    with TestClient(create_app(strict, database_url="sqlite://")) as cl:
        assert cl.post("/deployment", json={"mode": "shadow"}).status_code == 422


def test_admin_token_protects_mutations(api, monkeypatch):
    cl, *_ = api
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert cl.post("/deployment", json={"mode": "shadow"}).status_code == 401
    assert cl.post("/reload").status_code == 401
    assert cl.post("/deployment", json={"mode": "shadow"}, headers={"X-Admin-Token": "s3cret"}).status_code == 200
    assert cl.get("/deployment").status_code == 200                                                  # reads stay open
