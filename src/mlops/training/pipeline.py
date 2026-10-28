"""Automated training pipeline:

    load data -> VALIDATE (gate) -> split -> train N candidates (tracked in MLflow)
              -> pick best -> evaluate on holdout -> QUALITY GATE -> log model + artifacts

Every run is fully reproducible: data hash, config, seed, git-independent
reference profile and validation report are stored with the model.
"""
from __future__ import annotations
import hashlib, json, tempfile, time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import feature_columns
from ..data.profile import build_reference_profile
from ..data.validation import validate_dataframe, ValidationReport


class DataValidationError(RuntimeError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("Data validation failed: " +
                         "; ".join(f"{c.column}:{c.name}" for c in report.failures if c.severity == "error"))


class QualityGateError(RuntimeError):
    pass


@dataclass
class PipelineResult:
    run_id: str
    model_uri: str
    best_model: str
    metrics: dict
    candidates: dict
    validation: dict
    data_hash: str
    duration_s: float


def _build_estimator(name: str, params: dict, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(**params, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(**params, n_jobs=-1, random_state=seed)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def _build_pipeline(name: str, params: dict, cfg: dict) -> Pipeline:
    num, cat = feature_columns(cfg)
    scale = name == "logistic_regression"
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    return Pipeline([("pre", pre), ("clf", _build_estimator(name, params, cfg["project"]["seed"]))])


def hash_dataframe(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "log_loss": log_loss(y, proba), "brier": brier_score_loss(y, proba),
    }


def run_training_pipeline(df: pd.DataFrame, cfg: dict, *, run_name: str | None = None,
                          tags: dict | None = None,
                          progress: Callable[[str, float], None] | None = None) -> PipelineResult:
    """Execute the full pipeline. `progress(stage, fraction)` is optional UI hook."""
    t0 = time.time()
    emit = progress or (lambda *_: None)
    seed, target = cfg["project"]["seed"], cfg["data"]["target"]
    num, cat = feature_columns(cfg)
    features = num + cat

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=run_name or f"train-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.set_tags({"stage": "training", "framework": "scikit-learn", **(tags or {})})

        # ---- 1. DATA VALIDATION GATE ---------------------------------------
        emit("validating data", 0.05)
        report = validate_dataframe(df, cfg)
        mlflow.log_dict(report.to_dict(), "data_validation/report.json")
        mlflow.log_metrics({"validation_checks": len(report.checks),
                            "validation_failed": len(report.failures)})
        mlflow.set_tag("data_validation", "passed" if report.passed else "failed")
        if not report.passed:
            mlflow.set_tag("pipeline_status", "rejected_bad_data")
            raise DataValidationError(report)

        # ---- 2. SPLIT + PROFILE -------------------------------------------
        emit("splitting + profiling", 0.15)
        data_hash = hash_dataframe(df)
        train_df, test_df = train_test_split(df, test_size=cfg["data"]["test_size"],
                                             stratify=df[target], random_state=seed)
        # cast numerics to float64 so the model signature tolerates missing values at inference
        X_tr, y_tr = train_df[features].astype({c: "float64" for c in num}), train_df[target]
        X_te, y_te = test_df[features].astype({c: "float64" for c in num}), test_df[target]
        profile = build_reference_profile(train_df, cfg)   # fingerprint of *training* data only
        mlflow.log_dict(profile, "reference/reference_profile.json")
        mlflow.log_params({"data_hash": data_hash, "n_train": len(train_df), "n_test": len(test_df),
                           "seed": seed, "features": ",".join(features)})

        # ---- 3. TRAIN CANDIDATES (nested runs) -----------------------------
        cands_cfg = cfg["training"]["candidates"]
        cv = StratifiedKFold(cfg["training"]["cv_folds"], shuffle=True, random_state=seed)
        fitted, cand_scores = {}, {}
        for i, (name, params) in enumerate(cands_cfg.items()):
            emit(f"training {name}", 0.2 + 0.55 * i / len(cands_cfg))
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tag("candidate", name)
                mlflow.log_params({f"{name}.{k}": v for k, v in params.items()})
                pipe = _build_pipeline(name, params, cfg)
                cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
                pipe.fit(X_tr, y_tr)
                m = evaluate(pipe, X_te, y_te)
                mlflow.log_metrics({"cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
                                    **{f"test_{k}": v for k, v in m.items()}})
                fitted[name] = pipe
                cand_scores[name] = {"cv_auc": float(cv_auc.mean()), **{k: float(v) for k, v in m.items()}}

        # ---- 4. SELECT BEST (by cross-validated AUC, not the test set) -----
        emit("selecting best model", 0.80)
        best = max(cand_scores, key=lambda k: cand_scores[k]["cv_auc"])
        best_model, best_metrics = fitted[best], cand_scores[best]
        mlflow.set_tag("best_model", best)
        mlflow.log_params({"best_model": best})
        mlflow.log_metrics({f"test_{k}": v for k, v in best_metrics.items() if k != "cv_auc"})
        mlflow.log_metric("cv_auc_mean", best_metrics["cv_auc"])

        # ---- 5. QUALITY GATE -----------------------------------------------
        emit("quality gate", 0.88)
        min_auc = cfg["training"]["min_auc_to_accept"]
        if best_metrics["roc_auc"] < min_auc:
            mlflow.set_tag("pipeline_status", "rejected_low_quality")
            raise QualityGateError(f"AUC {best_metrics['roc_auc']:.3f} < required {min_auc}")

        # ---- 6. LOG MODEL (with signature + input example) -----------------
        emit("logging model", 0.94)
        sample = X_tr.head(5)
        signature = infer_signature(X_tr.head(200), best_model.predict_proba(X_tr.head(200))[:, 1])
        try:
            info = mlflow.sklearn.log_model(best_model, name="model", signature=signature, input_example=sample,
                                            serialization_format="cloudpickle")
        except TypeError:   # older MLflow versions
            info = mlflow.sklearn.log_model(best_model, artifact_path="model", signature=signature, input_example=sample,
                                            serialization_format="cloudpickle")
        mlflow.set_tags({"pipeline_status": "success", "model_uri": info.model_uri})
        emit("done", 1.0)

        return PipelineResult(run_id=run.info.run_id, model_uri=info.model_uri, best_model=best,
                              metrics=best_metrics, candidates=cand_scores,
                              validation=report.to_dict(), data_hash=data_hash,
                              duration_s=time.time() - t0)
