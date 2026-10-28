# 🧠 End-to-End MLOps Platform with Model Drift Detection

A mini production ML platform: **train → track → register → deploy → monitor → detect drift → retrain.**

> Build status: **Part 2 / 8 complete** ✅

```
PART 1  ████████████████████████████████  Foundation: data, validation, training, tracking   ✅
PART 2  ████████████████████████████████  Model registry + champion/challenger promotion     ✅
PART 3  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  FastAPI serving + PostgreSQL prediction logging
PART 4  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  A/B testing + shadow deployment
PART 5  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Drift detection + performance monitoring + auto-retrain
PART 6  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Prometheus + Grafana dashboards
PART 7  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Docker Compose full stack
PART 8  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  GitHub Actions CI/CD + docs polish
```

## Target architecture

```mermaid
flowchart LR
  D[Data + Validation Gate] --> T[Training Pipeline]
  T --> M[(MLflow Tracking + Registry)]
  M --> API[FastAPI: champion / challenger / shadow]
  API --> PG[(PostgreSQL prediction log)]
  API --> P[Prometheus] --> G[Grafana]
  PG --> DR[Drift + Performance Monitor]
  DR -- drift alert --> T
```

## What Part 1 delivers

| Capability | Where |
|---|---|
| Synthetic churn data with **controllable covariate + concept drift** (4 scenarios) | `src/mlops/data/generator.py` |
| **Data contract** (YAML) + automated validation with structured, severity-ranked checks | `configs/config.yaml`, `data/validation.py` |
| **Reference profile** (quantile bins, frequencies, KS sample) saved with every model, the baseline for drift detection in Part 5 | `data/profile.py` |
| Training pipeline: validate → split → 3 candidates (nested MLflow runs) → CV-based selection → **quality gate** → signed model | `training/pipeline.py` |
| Reproducibility: data hash, seed, config, validation report, profile all logged to MLflow | `training/pipeline.py` |
| Rich CLI with live progress bar and leaderboard | `cli.py` |
| 10 unit/integration tests | `tests/` |

### Design decisions worth talking about in interviews
* **Fail fast on bad data.** Validation runs *before* any training; rejected runs are still logged to MLflow with `pipeline_status=rejected_bad_data`, so you have an audit trail.
* **Select on CV, report on holdout.** The test set never influences model choice.
* **Quality gate.** A model below `min_auc_to_accept` is never logged as a deployable artifact.
* **Per-model baselines.** The reference profile travels with the model so drift is measured against what *that* model saw.

## What Part 2 delivers

| Capability | Where |
|---|---|
| MLflow **model registry** using aliases (`champion`, `challenger`) + status/audit tags on every version | `registry/manager.py` |
| **Promotion policy** judged on a fixed golden set: AUC gain, recall guardrail, calibration (Brier) guardrail, paired-bootstrap significance | `registry/promotion.py` |
| Safe **rollback** with no external state (each champion records the version it replaced) | `ModelRegistry.rollback` |
| Rejected models are tagged with the reason and never touch the champion | `promote_if_better` |
| CLI: `registry list / register / promote / rollback`, `train --register --promote`, and a scripted `demo` | `cli.py`, `cli_registry.py`, `demo.py` |
| 17 tests total (policy logic + full lifecycle) | `tests/test_registry.py` |

### The story the demo tells
```
v1  1,500 rows            -> first model, PROMOTED (meets quality floor)
v2  12,000 rows           -> ΔAUC +0.0056, all guardrails pass, PROMOTED
v3  concept-drifted data  -> reports AUC 0.91 on ITS OWN test split, but 0.73 on the golden set -> REJECTED
incident -> ROLLBACK v2 -> v1
```
**Key insight:** v3 looked great on its own test split. A model must be judged on a *shared, fixed* evaluation set,
never on the data it was trained near. (In production the golden set would be curated, labeled real data.)

```bash
make demo                                 # watch the whole lifecycle
python -m mlops.cli train --register --promote
python -m mlops.cli registry list
python -m mlops.cli registry rollback --reason "latency regression"
```

In Part 4 this offline gate becomes stage 1 of a longer path: **offline gate → shadow → A/B → promote**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install
make data                   # generate data/train.csv
make train                  # full pipeline with progress bar
make test                   # 17 tests
make ui                     # MLflow UI on http://localhost:5000
```

Try the data gate:
```bash
python -m mlops.cli generate --scenario severe_drift --out data/drifted.csv
python -m mlops.cli validate data/drifted.csv -v
```
(Run commands with `PYTHONPATH=src`, which the Makefile sets for you.)
