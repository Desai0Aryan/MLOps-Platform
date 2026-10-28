import copy, numpy as np, pytest
from mlops.data import generate_dataset
from mlops.registry import ModelRegistry, RegistryError, compare_models
from mlops.training import run_training_pipeline


@pytest.fixture(scope="module")
def fast(cfg, tmp_path_factory):
    c = copy.deepcopy(cfg)
    c["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path_factory.mktemp('reg') / 'r.db'}"
    c["training"]["candidates"] = {"logistic_regression": c["training"]["candidates"]["logistic_regression"]}
    return c


def _train(c, rows, scen, seed):
    return run_training_pipeline(generate_dataset(rows, scen, seed=seed), c)


@pytest.fixture(scope="module")
def reg(fast):
    return ModelRegistry(fast)


# ---- pure policy logic (no MLflow) -----------------------------------------
def _y_p(n=4000, auc_noise=1.0, seed=0):
    rng = np.random.default_rng(seed); y = rng.integers(0, 2, n)
    return y, np.clip(0.5 + (y - 0.5) * 0.3 + rng.normal(0, 0.25 * auc_noise, n), 0.01, 0.99)


def test_policy_blocks_worse_model(cfg):
    y, good = _y_p(auc_noise=1.0); _, bad = _y_p(auc_noise=3.0, seed=1)
    d = compare_models(bad, good, y, cfg)
    assert not d.promote and any(r.name == "auc_improvement" and not r.passed for r in d.rules)


def test_policy_accepts_clearly_better_model(cfg):
    y, weak = _y_p(auc_noise=2.5); rng = np.random.default_rng(5)
    strong = np.clip(0.5 + (y - 0.5) * 0.3 + rng.normal(0, 0.1, len(y)), 0.01, 0.99)
    assert compare_models(strong, weak, y, cfg).promote


def test_first_model_only_needs_quality_floor(cfg):
    y, p = _y_p(); d = compare_models(p, None, y, cfg)
    assert d.promote and d.champion is None


def test_significance_rule_can_be_required(cfg):
    c = copy.deepcopy(cfg); c["registry"]["promotion"]["require_significance"] = True
    y, a = _y_p(seed=2); d = compare_models(a, a.copy(), y, c)        # identical models: no real gain
    assert not d.promote


# ---- full lifecycle ---------------------------------------------------------
def test_lifecycle_register_promote_reject_rollback(reg, fast):
    assert reg.champion() is None
    v1 = reg.register_run(_train(fast, 1500, "baseline", 1).run_id)
    assert reg.challenger().version == v1.version
    assert reg.promote_if_better(v1.version).promote
    assert reg.champion().version == v1.version and reg.challenger() is None

    v2 = reg.register_run(_train(fast, 8000, "baseline", 2).run_id)
    assert reg.promote_if_better(v2.version).promote
    assert reg.champion().version == v2.version
    assert reg.get_version(v1.version).tags["status"] == "archived"
    assert reg.champion().tags["replaced_version"] == str(v1.version)

    v3 = reg.register_run(_train(fast, 5000, "concept_drift", 3).run_id)
    d = reg.promote_if_better(v3.version)
    assert not d.promote
    assert reg.champion().version == v2.version                      # champion untouched
    assert reg.get_version(v3.version).tags["status"] == "rejected"

    assert reg.rollback("test") == (int(v2.version), int(v1.version))
    assert reg.champion().version == v1.version
    assert reg.get_version(v2.version).tags["status"] == "rolled_back"
    with pytest.raises(RegistryError):
        reg.rollback()                                               # v1 has no predecessor


def test_cannot_register_failed_runs(fast, reg):
    import mlflow
    from mlops.training import DataValidationError
    bad = generate_dataset(3000, "baseline"); bad["age"] = 999
    with pytest.raises(DataValidationError):
        run_training_pipeline(bad, fast)
    run_id = mlflow.search_runs(order_by=["start_time DESC"], max_results=1).iloc[0]["run_id"]
    with pytest.raises(RegistryError):
        reg.register_run(run_id)


def test_champion_model_serves_predictions(reg, cfg):
    from mlops.registry import predict_proba
    p = predict_proba(reg.load_model("champion"), generate_dataset(30, "baseline", seed=7), cfg)
    assert p.shape == (30,) and ((p >= 0) & (p <= 1)).all()
