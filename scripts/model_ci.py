#!/usr/bin/env python3
"""Model CI: a regression test for the *training code*, run on every pull request.

Unit tests prove the code runs. This proves it still produces a GOOD, REPRODUCIBLE, FAST model:

  1. data contract         clean data passes AND corrupted data is rejected (negative control: the gate itself works)
  2. quality vs baseline   AUC / Brier within tolerance of the committed `ci/baseline_metrics.json`
  3. calibration           mean predicted churn ≈ actual churn on a held-out golden set
  4. slice quality         no customer segment is left behind (min AUC over contract / payment / internet slices)
  5. reproducibility       training twice with the same seed gives bit-identical metrics
  6. latency budget        single-row inference p95 under budget

    python scripts/model_ci.py                      # exit 1 on any failure, writes model-ci/report.{json,md}
    python scripts/model_ci.py --update-baseline    # deliberately accept the current metrics as the new baseline
"""
from __future__ import annotations
import argparse, copy, json, os, platform, sys, tempfile, time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlops.config import load_config  # noqa: E402
from mlops.data import generate_dataset, validate_dataframe  # noqa: E402
from mlops.registry import load_eval_set, predict_proba  # noqa: E402
from mlops.training import run_training_pipeline  # noqa: E402

BASELINE = ROOT / "ci" / "baseline_metrics.json"


def _isolated(cfg: dict, tmp: str, light: bool = False) -> dict:
    c = copy.deepcopy(cfg)
    c["mlflow"]["tracking_uri"] = f"sqlite:///{tmp}/mlflow.db"
    if light:
        c["training"]["candidates"] = {"logistic_regression": c["training"]["candidates"]["logistic_regression"]}
    return c


def evaluate(cfg: dict, baseline: dict | None, updating: bool = False) -> dict:
    checks: list[dict] = []
    def add(name, ok, detail): checks.append({"name": name, "passed": bool(ok), "detail": detail})
    metrics: dict = {}

    with tempfile.TemporaryDirectory() as tmp:
        c = _isolated(cfg, tmp)
        df = generate_dataset(c["data"]["n_train"], "baseline", seed=c["project"]["seed"])

        # 1. data contract + negative control
        add("data contract accepts clean data", validate_dataframe(df, c).passed, f"{len(df):,} rows")
        bad = df.copy(); bad["age"] = 999; bad.loc[:50, "contract_type"] = "lifetime"
        add("data contract REJECTS corrupted data (negative control)", not validate_dataframe(bad, c).passed, "age=999 + unknown category")

        # train the real pipeline
        t0 = time.time(); res = run_training_pipeline(df, c, run_name="model-ci"); train_s = time.time() - t0
        m = res.metrics; metrics.update(roc_auc=m["roc_auc"], brier=m["brier"], log_loss=m["log_loss"], best_model=res.best_model, train_seconds=round(train_s, 1))

        # 2. quality vs the committed baseline
        if baseline:
            tol = baseline["tolerance"]
            add("AUC vs baseline", m["roc_auc"] >= baseline["roc_auc"] - tol["roc_auc"],
                f"{m['roc_auc']:.4f} vs baseline {baseline['roc_auc']:.4f} (tolerance -{tol['roc_auc']})")
            add("Brier vs baseline", m["brier"] <= baseline["brier"] + tol["brier"],
                f"{m['brier']:.4f} vs baseline {baseline['brier']:.4f} (tolerance +{tol['brier']})")
        elif updating:
            add("baseline is being regenerated", True, "review the diff of ci/baseline_metrics.json before committing")
        else:
            add("baseline file present", False, "run with --update-baseline once and commit ci/baseline_metrics.json")

        # 3 + 4. golden-set calibration and slices (model reloaded from MLflow, as production would)
        import mlflow
        model = mlflow.sklearn.load_model(res.model_uri)
        gold = load_eval_set(c); y = gold[c["data"]["target"]].values; p = predict_proba(model, gold, c)
        gap = float(p.mean() - y.mean()); metrics["calibration_gap"] = gap
        lim = (baseline or {}).get("calibration_gap_max", 0.03)
        add("calibration on golden set", abs(gap) <= lim, f"predicted {p.mean():.3f} vs actual {y.mean():.3f} (|gap| <= {lim})")
        floor = (baseline or {}).get("slice_auc_min", 0.65); worst = (1.0, "")
        for col in ("contract_type", "payment_method", "internet_service"):
            for val, g in gold.groupby(col):
                if len(g) >= 300 and g[c["data"]["target"]].nunique() > 1:
                    s = roc_auc_score(g[c["data"]["target"]], p[g.index]); worst = min(worst, (s, f"{col}={val}"))
        metrics["worst_slice_auc"] = worst[0]
        add("no segment is left behind (min slice AUC)", worst[0] >= floor, f"worst slice {worst[1]}: AUC {worst[0]:.3f} (>= {floor})")

        # 6. latency budget
        one = gold.drop(columns=[c["data"]["target"]]).head(1)
        for _ in range(5): predict_proba(model, one, c)
        lat = []
        for _ in range(100):
            t = time.perf_counter(); predict_proba(model, one, c); lat.append((time.perf_counter() - t) * 1000)
        p95 = float(np.percentile(lat, 95)); metrics["latency_p95_ms"] = p95
        budget = (baseline or {}).get("latency_p95_ms_max", 100)
        add("single-row inference latency", p95 <= budget, f"p95 {p95:.1f} ms (budget {budget} ms)")

        # 5. reproducibility (light config: same seed twice => identical numbers)
        auc = []
        for i in range(2):
            with tempfile.TemporaryDirectory() as t2:
                auc.append(run_training_pipeline(df.head(6000), _isolated(cfg, t2, light=True), run_name=f"repro-{i}").metrics["roc_auc"])
        add("training is reproducible", abs(auc[0] - auc[1]) < 1e-12, f"AUC {auc[0]:.12f} vs {auc[1]:.12f}")

    return {"passed": all(k["passed"] for k in checks), "checks": checks, "metrics": metrics}


def to_markdown(rep: dict) -> str:
    m = rep["metrics"]
    out = [f"### Model CI: {'✅ passed' if rep['passed'] else '❌ FAILED'}", "",
           f"`{m['best_model']}` · AUC **{m['roc_auc']:.4f}** · Brier {m['brier']:.4f} · calibration gap {m['calibration_gap']:+.3f} · "
           f"worst slice AUC {m['worst_slice_auc']:.3f} · p95 {m['latency_p95_ms']:.1f} ms · trained in {m['train_seconds']}s", "",
           "| | check | detail |", "|---|---|---|"]
    out += [f"| {'✅' if k['passed'] else '❌'} | {k['name']} | {k['detail']} |" for k in rep["checks"]]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="model-ci"); ap.add_argument("--update-baseline", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()                       # resolve BEFORE changing directory
    cfg = load_config()
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() and not a.update_baseline else None
    with tempfile.TemporaryDirectory() as scratch:    # MLflow writes artifacts under ./mlruns: keep that out of the checkout
        cwd = os.getcwd(); os.chdir(scratch)
        try:
            rep = evaluate(cfg, baseline, updating=a.update_baseline)
        finally:
            os.chdir(cwd)
    if a.update_baseline:
        m = rep["metrics"]
        BASELINE.parent.mkdir(exist_ok=True)
        BASELINE.write_text(json.dumps({
            "roc_auc": round(m["roc_auc"], 4), "brier": round(m["brier"], 4), "log_loss": round(m["log_loss"], 4),
            "tolerance": {"roc_auc": 0.01, "brier": 0.01}, "calibration_gap_max": 0.03, "slice_auc_min": 0.65, "latency_p95_ms_max": 100,
            "best_model": m["best_model"], "n_train": cfg["data"]["n_train"], "seed": cfg["project"]["seed"],
            "python": platform.python_version()}, indent=2) + "\n")
        print(f"baseline written to {BASELINE}")
    out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2)); md = to_markdown(rep); (out / "report.md").write_text(md); print(md)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh: fh.write(md)
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
