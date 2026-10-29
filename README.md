# 🧠 End-to-End MLOps Platform with Model Drift Detection

A mini production ML platform: **train → track → register → deploy → monitor → detect drift → retrain.**

> Build status: **Part 4 / 8 complete** ✅

```
PART 1  ████████████████████████████████  Foundation: data, validation, training, tracking   ✅
PART 2  ████████████████████████████████  Model registry + champion/challenger promotion     ✅
PART 3  ████████████████████████████████  FastAPI serving + PostgreSQL prediction logging    ✅
PART 4  ████████████████████████████████  A/B testing + shadow deployment                    ✅
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

## What Part 3 delivers

| Capability | Where |
|---|---|
| FastAPI service: `/predict`, `/predict/batch`, `/feedback[/batch]`, `/model-info`, `/reload`, `/stats`, `/health`, `/ready` | `serving/app.py` |
| **Request schema generated from the data contract**: one source of truth for training validation, API validation and OpenAPI docs | `serving/schemas.py` |
| Every prediction logged (features, probability, version, latency, request id) via a **background task**, so the DB adds no latency | `serving/db.py` |
| **Delayed ground-truth labels** joined on `request_id` -> live accuracy/AUC (powers Part 5) | `/feedback`, `/stats` |
| **Zero-downtime hot reload**: new champion is loaded fully, then swapped atomically; also auto-polls the registry | `serving/store.py` |
| Liveness vs readiness probes; 503 (not a crash) when no champion exists | `/health`, `/ready` |
| SQLAlchemy 2.0: SQLite for tests, **PostgreSQL** via `DATABASE_URL` | `serving/db.py`, `docker-compose.postgres.yml` |
| Traffic simulator that can replay any drift scenario + send delayed labels | `simulate.py` |
| 29 tests total | `tests/test_serving.py` |

```bash
make train-register          # train, register, promote v1
make serve                   # API on :8000  (docs at /docs)
make simulate                # 1,000 baseline requests + labels
python -m mlops.cli simulate --scenario concept_drift --n 1000
curl localhost:8000/stats
```

**PostgreSQL:** `docker compose -f docker-compose.postgres.yml up -d`, then
`export DATABASE_URL=postgresql+psycopg2://mlops:mlops@localhost:5432/mlops` before `make serve`.
(Tests run on SQLite; the Postgres path uses the same SQLAlchemy models and is exercised end-to-end in Part 7.)

**What the simulator already reveals:** replaying `concept_drift` traffic, the model predicts ~35% churn
while the real rate is ~48%. That gap is exactly what Part 5's monitors will catch automatically.

## What Part 4 delivers: safe rollouts

```
 register ──► safety floor ──► SHADOW ──► A/B (≤50 %) ──► statistical verdict ──► promote / stop
              (golden AUC)    0 user risk   sticky split     SRM · log-loss · latency
```

| Capability | Where |
|---|---|
| **Shadow deployment**: challenger scores every request *after* the response is sent; its output is stored, never returned | `serving/app.py` (`shadow_score`) |
| **A/B routing**: SHA-256 sticky assignment (same customer, same model, on every replica, no lookup table) | `serving/deployment.py` |
| **Paired analysis** for shadow, **independent-arms analysis** for A/B: Welch t-test on log-loss, z-test on accuracy, bootstrap ΔAUC CIs | `experiments/analysis.py` |
| **Sample-Ratio-Mismatch check**: detects broken randomisation before trusting any result | `ab_report` |
| Guardrails: latency p95, shadow error rate, min labeled evidence per arm, max 50 % challenger share, golden-set safety floor | `configs/config.yaml → deployment` |
| **Fail-safe serving**: challenger exception in A/B transparently falls back to champion; shadow errors are recorded, never raised | tested |
| Rollout state in the DB (shared by all replicas); every change restarts the experiment clock | `DeploymentState` |
| Optional `ADMIN_TOKEN` protecting mutating endpoints | `require_admin` |
| Auto-conclusion: `promote` / `stop` applied to the registry + traffic in one call | `POST /experiment/conclude` |
| 45 tests total | `tests/test_rollout.py` |

```bash
make demo-deploy                              # the full production story, in-process
python -m mlops.cli deploy shadow             # against a running API
python -m mlops.cli simulate --scenario concept_drift --n 1000
python -m mlops.cli deploy analyze
python -m mlops.cli deploy ab --split 0.2
python -m mlops.cli deploy conclude --action auto
```

### Why primary metric = log-loss
Concept drift mostly breaks **calibration** (the model says 34 % churn while reality is 50 %), while AUC
barely moves. Log-loss punishes exactly that and is continuous, so a t-test applies. In the demo the champion's
AUC drops only ~0.01-0.04 but its log-loss is **0.57 vs 0.34** for the retrained challenger.

### Interview talking points
* Shadow gives a **paired** comparison (same requests) and needs far less data than an A/B test; A/B is then only a final confirmation.
* The golden set can't judge a model trained for *today's* world, so live evidence takes over. The golden set stays as a *safety floor*.
* SRM, sticky bucketing, a cap on challenger exposure and fallback-on-failure are the details that separate toy A/B code from production A/B code.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install
make data                   # generate data/train.csv
make train                  # full pipeline with progress bar
make test                   # 45 tests
make ui                     # MLflow UI on http://localhost:5000
```

Try the data gate:
```bash
python -m mlops.cli generate --scenario severe_drift --out data/drifted.csv
python -m mlops.cli validate data/drifted.csv -v
```
(Run commands with `PYTHONPATH=src`, which the Makefile sets for you.)
