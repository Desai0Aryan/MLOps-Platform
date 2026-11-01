# 🧠 End-to-End MLOps Platform with Model Drift Detection

A mini production ML platform: **train → track → register → deploy → monitor → detect drift → retrain.**

> Build status: **Part 7 / 8 complete** ✅

```
PART 1  ████████████████████████████████  Foundation: data, validation, training, tracking   ✅
PART 2  ████████████████████████████████  Model registry + champion/challenger promotion     ✅
PART 3  ████████████████████████████████  FastAPI serving + PostgreSQL prediction logging    ✅
PART 4  ████████████████████████████████  A/B testing + shadow deployment                    ✅
PART 5  ████████████████████████████████  Drift detection + performance monitoring + auto-retrain ✅
PART 6  ████████████████████████████████  Prometheus + Grafana dashboards                    ✅
PART 7  ████████████████████████████████  Docker Compose full stack                          ✅
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

## What Part 5 delivers: monitoring + the closed loop

```
 live traffic ─► DRIFT (no labels: PSI · KS · χ² · missing-rate · prediction drift) ─┐
 delayed labels ─► PERFORMANCE (AUC · log-loss · calibration gap · ECE) ────────────┼─► status ok/warning/critical
                                                                                    │
   critical ─► enough fresh LABELED data? ─► RETRAIN (validation + quality gates) ─► challenger
            ─► SHADOW ─► A/B ─► PROMOTE   (every decision logged to `lifecycle_event`)
```

| Capability | Where |
|---|---|
| Per-feature **PSI + KS / chi-square + missing-rate** drift vs. *each model's own* training profile (stored in MLflow) | `monitoring/drift.py` |
| **Prediction (score) drift**: early warning that needs no labels | `data/profile.py → build_score_profile` |
| **Live performance** on delayed labels: AUC, log-loss, Brier, ECE, calibration gap, each vs. its training baseline | `monitoring/performance.py` |
| Severity by *effect size* (PSI), not p-value, because with big samples everything is "significant" | `drift.py` docstring |
| Monitoring history + append-only **lifecycle timeline** in PostgreSQL | `MonitoringRun`, `LifecycleEvent` |
| **Orchestrator**: monitor → retrain on production data → register → shadow; advances A/B → promote on its own | `orchestrator.py`, `POST /experiment/advance` |
| Safeguards: waits for labels, refuses contract-violating data, cooldown, quality gate, experiment timeout | tested |
| Self-contained **HTML drift report** (light/dark) | `monitoring/report_html.py`, `docs/sample_drift_report.html` |
| 64 tests total | `tests/test_monitoring.py` |

![sample report](docs/sample_drift_report.png)

```bash
make demo-lifecycle                    # the hands-off story (≈1 min)
python -m mlops.cli monitor --html report.html
python -m mlops.cli worker --interval 300       # the autonomous loop, as a service
python -m mlops.cli events                      # why did the model change?
```

### Three lessons the tests encode
1. **Data drift ≠ model failure.** `severe_drift` trips 4 critical features, but live AUC can stay healthy. It's an early warning, not proof of damage.
2. **Concept drift is invisible to feature monitors.** `concept_drift` leaves all 9 features stable (PSI < 0.1); only label-based calibration monitoring catches it (`test_concept_drift_is_caught_by_live_performance_only`). Real platforms need both.
3. **Automation needs brakes.** With 8 % missing `usage_gb` the data contract fails, so the orchestrator *refuses to retrain*
   (`test_gate_refuses_to_retrain_on_contract_violating_data`): auto-retraining on a broken pipeline would bake the bug into the model.

## What Part 6 delivers: observability

| Capability | Where |
|---|---|
| `/metrics`: request rate/latency histograms, predictions by version+variant, probability distribution, inference time, error stages, shadow + feedback counters, process CPU/memory | `serving/metrics.py` |
| **Platform-state collector** computed at scrape time from the DB: drift PSI per feature, monitor verdicts, live AUC/calibration vs training baseline, rollout mode + A/B split, loaded model versions, lifecycle events | `PlatformCollector` |
| **13 alert rules** with severity + self-explaining descriptions | `deploy/prometheus/alerts.yml` |
| **3 provisioned Grafana dashboards** (40 panels): service health, model quality & drift, rollouts & lifecycle, with annotations on promotions / retrains | `deploy/grafana/`, `scripts/build_dashboards.py` |
| `docker-compose.observability.yml`: Prometheus + Grafana in one command | repo root |
| Traffic generator that injects drift mid-run to light up the dashboards | `simulate --loop-seconds 180 --drift-after 90` |
| 78 tests total (incl. "every metric used in a dashboard/alert is really exported") | `tests/test_metrics.py` |

```bash
make observability-up && make serve        # then in another shell:
make traffic                               # watch PSI climb, status flip to CRITICAL, worker retrain, new champion
```
See [`deploy/README.md`](deploy/README.md) for the dashboard guide and alert catalogue.

**How it was verified:** every alert rule and all 54 dashboard/annotation queries pass the real `promtool`; I then ran the real
API + a real Prometheus (2 s scrape) + the real worker through the full drift → retrain → shadow → A/B → promote cycle and executed every
dashboard query against it: **48/54 returned data, 0 errors** (the other 6 are legitimately empty: no failures, no challenger after
promotion, event-only annotations). Prometheus counted exactly the 9,400 predictions I sent, and `ModelPromoted` fired. That run caught two real
dashboard bugs (a path filter that missed `/predict/batch`; an error-rate stat that showed "n/a" instead of 0 %).
Not verifiable in my sandbox: rendering inside Grafana itself and the Docker compose files, since neither Grafana nor Docker is available there.

## What Part 7 delivers: one command, whole platform

```mermaid
flowchart LR
  subgraph data
    PG[(postgres 16<br/>mlops + mlflow DBs)]
  end
  ML[mlflow server<br/>proxied artifacts] --> PG
  BOOT[bootstrap job<br/>trains + promotes v1] --> ML
  API[api :8000<br/>FastAPI] --> PG
  API --> ML
  WORK[worker<br/>monitor → retrain → A/B → promote] --> API
  WORK --> PG
  WORK --> ML
  PROM[prometheus] -- scrapes /metrics --> API
  GRAF[grafana] --> PROM
  TRAFFIC[traffic generator<br/>profile: demo] --> API
  BOOT -. completed_successfully .-> API
```

```bash
cp .env.example .env            # change ADMIN_TOKEN + passwords
make up                         # builds one image, starts 7 services in dependency order
make up-demo                    # + traffic generator that drifts after 2 min and a fast worker: watch it heal itself
```
| URL | What |
|---|---|
| http://localhost:8000/docs | API (mutating endpoints need `X-Admin-Token`) |
| http://localhost:3000 | Grafana: 3 dashboards pre-provisioned |
| http://localhost:5000 | MLflow: experiments, registry, artifacts |
| http://localhost:9090 | Prometheus + alert rules |

| Design choice | Why |
|---|---|
| **One image, many roles** (api / worker / mlflow / bootstrap / traffic) | single build, single dependency set, multi-stage + non-root + healthcheck |
| **`bootstrap` one-shot job** gates the API (`service_completed_successfully`) | the API never starts without a model; idempotent, so safe on every `up` |
| **MLflow `--serve-artifacts`** | API/worker upload & download artifacts over HTTP: no shared volume, no S3 needed. `--allowed-hosts` set because MLflow 3 rejects unknown Host headers |
| **Separate databases** (`mlops`, `mlflow`) in one Postgres | independent schemas/backups; created by an init script |
| **Advisory lock around `CREATE TABLE`** | several replicas can boot at once without racing |
| **Retention job** (`retention:` in config) | prediction log / monitoring runs / shadow rows no longer grow forever; refuses a policy that would starve retraining |
| Postgres, MLflow, Prometheus bound to `127.0.0.1` | only the API and Grafana are reachable from the network |

### How it was verified (and what wasn't)
There is no Docker daemon in my build sandbox, so I verified everything *except the container runtime itself*:
* the real **`docker compose config`** accepts the file (default + `demo` profile) and **`hadolint`** passes the Dockerfile
* a **faithful rehearsal**: each service's command + environment was extracted from the *resolved compose file* and run as a process against a
  **real PostgreSQL 16**, a **real MLflow server** (same flags, proxied artifacts), the worker as configured, and the real Prometheus with the real
  config, using the same hostnames. Result: bootstrap (default config, 23 s) → API → worker auto-ran retrain → shadow → A/B → promote; **9,000
  predictions in Postgres**, admin-token enforced (401/200), v2 still served after an API restart, 18 artifacts uploaded through the MLflow proxy,
  and the `ApiDown` alert went *pending* during my deliberate restart
* **`tests/test_postgres.py`** (real Postgres, incl. 6 replicas creating tables simultaneously) and 10 static checks on the compose/Dockerfile
**Not verified:** `docker build` itself, container healthchecks, and named-volume ownership at runtime. Run `make up` and check `docker compose ps`.

## Quickstart (local, no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
make install
make data                   # generate data/train.csv
make train                  # full pipeline with progress bar
make test                   # 95 tests: 92 run anywhere, 3 need Postgres (make test-postgres)
make ui                     # MLflow UI on http://localhost:5000
```

Try the data gate:
```bash
python -m mlops.cli generate --scenario severe_drift --out data/drifted.csv
python -m mlops.cli validate data/drifted.csv -v
```
(Run commands with `PYTHONPATH=src`, which the Makefile sets for you.)
