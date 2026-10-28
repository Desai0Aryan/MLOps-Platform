import mlflow, pytest
import numpy as np
from mlops.data import generate_dataset
from mlops.training import run_training_pipeline, DataValidationError, QualityGateError


def test_pipeline_end_to_end(cfg):
    stages = []
    res = run_training_pipeline(generate_dataset(4000, "baseline"), cfg, progress=lambda s, f: stages.append(f))
    assert res.metrics["roc_auc"] > 0.7 and stages[-1] == 1.0
    assert set(res.candidates) == set(cfg["training"]["candidates"])
    run = mlflow.get_run(res.run_id)
    assert run.data.tags["pipeline_status"] == "success"
    # model is loadable and returns probabilities
    model = mlflow.sklearn.load_model(res.model_uri)
    p = model.predict_proba(generate_dataset(50, "baseline", seed=9).drop(columns="churned").astype({c: "float64" for c in cfg["schema"]["numeric"]}))[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_bad_data_is_rejected_before_training(cfg):
    df = generate_dataset(3000, "baseline"); df["age"] = 999
    with pytest.raises(DataValidationError):
        run_training_pipeline(df, cfg)


def test_quality_gate_blocks_weak_models(cfg):
    strict = {**cfg, "training": {**cfg["training"], "min_auc_to_accept": 0.999}}
    with pytest.raises(QualityGateError):
        run_training_pipeline(generate_dataset(3000, "baseline"), strict)
