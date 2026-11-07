"""The model-CI gate must (a) pass on good code and (b) actually FAIL on regressions."""
import copy, importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_ci", ROOT / "scripts/model_ci.py")
model_ci = importlib.util.module_from_spec(spec); spec.loader.exec_module(model_ci)


@pytest.fixture(scope="module")
def light(cfg):
    c = copy.deepcopy(cfg); c["data"]["n_train"] = 4000
    c["training"]["candidates"] = {"logistic_regression": c["training"]["candidates"]["logistic_regression"]}
    return c


@pytest.fixture(scope="module")
def baseline(light):
    rep = model_ci.evaluate(light, None, updating=True)
    assert rep["passed"], [c for c in rep["checks"] if not c["passed"]]                # regenerating a baseline must succeed
    m = rep["metrics"]
    return {"roc_auc": m["roc_auc"], "brier": m["brier"], "tolerance": {"roc_auc": 0.01, "brier": 0.01}}


def failed(rep): return {c["name"] for c in rep["checks"] if not c["passed"]}


def test_gate_passes_on_unchanged_code(light, baseline):
    rep = model_ci.evaluate(light, baseline)
    assert rep["passed"], failed(rep)
    names = {c["name"] for c in rep["checks"]}
    assert {"AUC vs baseline", "Brier vs baseline", "calibration on golden set", "training is reproducible",
            "data contract REJECTS corrupted data (negative control)"} <= names


def test_gate_blocks_a_quality_regression(light, baseline):
    """Simulates a PR that made the model worse: the committed baseline is higher than what the code now produces."""
    rep = model_ci.evaluate(light, {**baseline, "roc_auc": baseline["roc_auc"] + 0.05})
    assert not rep["passed"] and failed(rep) == {"AUC vs baseline"}


def test_gate_blocks_a_calibration_regression(light, baseline):
    rep = model_ci.evaluate(light, {**baseline, "calibration_gap_max": 0.0})
    assert "calibration on golden set" in failed(rep)


def test_gate_blocks_a_latency_regression(light, baseline):
    rep = model_ci.evaluate(light, {**baseline, "latency_p95_ms_max": 0.0001})
    assert "single-row inference latency" in failed(rep)


def test_missing_baseline_fails_loudly(light):
    assert "baseline file present" in failed(model_ci.evaluate(light, None))


def test_markdown_summary_renders(light, baseline):
    md = model_ci.to_markdown(model_ci.evaluate(light, baseline))
    assert md.startswith("### Model CI") and "| ✅ |" in md and "AUC" in md


def test_committed_baseline_is_well_formed():
    import json
    b = json.loads((ROOT / "ci/baseline_metrics.json").read_text())
    assert 0.7 < b["roc_auc"] < 1 and set(b["tolerance"]) == {"roc_auc", "brier"} and b["latency_p95_ms_max"] > 0
