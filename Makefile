.DEFAULT_GOAL := help
.PHONY: help install lock demo-lifecycle demo demo-deploy data train train-register serve simulate traffic monitor worker ui lint test test-postgres model-ci update-baseline dashboards validate-obs docker-validate up up-demo smoke logs ps down clean
PY ?= python
export PYTHONPATH := src

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## ---- setup
install: ## Install runtime + dev dependencies (locked versions)
	pip install -r requirements-dev.txt -c constraints.txt

lock: ## Re-resolve constraints.txt from requirements.txt (needs `uv`)
	uv pip compile requirements.txt --universal --python-version 3.12 -o constraints.txt

## ---- try it locally (no Docker)
demo-lifecycle: ## THE story: drift -> auto-retrain -> shadow -> A/B -> promote, hands-off (~1 min)
	$(PY) -m mlops.cli demo-lifecycle

demo: ## Registry lifecycle: promote, reject a bad model, roll back
	$(PY) -m mlops.cli demo

demo-deploy: ## Shadow + A/B rollout with statistical verdicts
	$(PY) -m mlops.cli demo-deploy

data: ## Generate data/train.csv
	$(PY) -m mlops.cli generate --scenario baseline --out data/train.csv

train: ## Run the training pipeline
	$(PY) -m mlops.cli train

train-register: ## Train, register and promote a model
	$(PY) -m mlops.cli train --register --promote

serve: ## Run the API on :8000 (docs at /docs, metrics at /metrics)
	$(PY) -m mlops.cli serve --port 8000

simulate: ## Send 1,000 synthetic requests (+ labels) to a running API
	$(PY) -m mlops.cli simulate --url http://localhost:8000 --n 1000

traffic: ## 3 min of live traffic that drifts after 90 s (feeds the dashboards)
	$(PY) -m mlops.cli simulate --url http://localhost:8000 --n 150 --loop-seconds 180 --pause 2 --drift-after 90 --drift-scenario market_shift

monitor: ## Run drift + performance monitoring; writes docs/latest_report.html
	$(PY) -m mlops.cli monitor --html docs/latest_report.html

worker: ## Run the autonomous loop (monitor -> retrain -> advance experiments)
	$(PY) -m mlops.cli worker --interval 300

ui: ## Open the MLflow UI on :5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

## ---- quality gates (what CI runs)
lint: ## ruff + shellcheck
	ruff check .
	shellcheck scripts/*.sh

test: ## Unit + integration tests
	$(PY) -m pytest -q

test-postgres: ## Run the PostgreSQL tests (set TEST_DATABASE_URL, or use the local default)
	TEST_DATABASE_URL=$${TEST_DATABASE_URL:-postgresql+psycopg2://mlops:mlops@localhost:5432/mlops_test} $(PY) -m pytest -q tests/test_postgres.py

model-ci: ## Model quality gate vs ci/baseline_metrics.json
	$(PY) scripts/model_ci.py

update-baseline: ## Accept the current model metrics as the new baseline (review the diff!)
	$(PY) scripts/model_ci.py --update-baseline

dashboards: ## Regenerate the Grafana dashboards from scripts/build_dashboards.py
	$(PY) scripts/build_dashboards.py

validate-obs: ## promtool-validate Prometheus config, alert rules and every dashboard query
	$(PY) scripts/validate_observability.py

docker-validate: ## Validate compose files + Dockerfile + observability configs
	docker compose --env-file .env.example --profile demo config -q
	hadolint Dockerfile
	$(PY) scripts/validate_observability.py

## ---- full stack (Docker)
up: ## Build and start the whole platform
	@test -f .env || cp .env.example .env
	docker compose up -d --build

up-demo: ## Start it with a traffic generator that drifts after 2 min and a fast worker
	@test -f .env || cp .env.example .env
	WORKER_INTERVAL=20 DRIFT_AFTER_SECONDS=120 docker compose --profile demo up -d --build

smoke: ## Smoke-test the running stack (API, MLflow, Prometheus, Grafana)
	set -a; . ./.env; set +a; $(PY) scripts/smoke_test.py --api http://localhost:8000 --mlflow http://localhost:5000 --prometheus http://localhost:9090 --grafana http://localhost:3000 --grafana-auth "admin:$${GRAFANA_ADMIN_PASSWORD}" --admin-token "$${ADMIN_TOKEN}"

logs: ## Tail API + worker logs
	docker compose logs -f --tail=50 api worker

ps: ## Show service status
	docker compose ps

down: ## Stop everything (add -v to the command to also delete data volumes)
	docker compose --profile demo down

clean: ## Remove local build/test artifacts
	rm -rf mlruns mlflow.db serving.db data/*.csv artifacts .pytest_cache model-ci docs/latest_report.html
	find . -name __pycache__ -prune -exec rm -rf {} +
