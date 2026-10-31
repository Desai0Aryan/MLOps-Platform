.PHONY: install data train train-register serve simulate test ui demo demo-deploy demo-lifecycle monitor worker dashboards validate-obs observability-up observability-down traffic clean
PY ?= python
export PYTHONPATH := src

install:  ; pip install -r requirements.txt
data:     ; $(PY) -m mlops.cli generate --scenario baseline --out data/train.csv
train:    ; $(PY) -m mlops.cli train
test:     ; $(PY) -m pytest -q
train-register: ; $(PY) -m mlops.cli train --register --promote
serve:    ; $(PY) -m mlops.cli serve --port 8000
simulate: ; $(PY) -m mlops.cli simulate --url http://localhost:8000 --n 1000
demo-lifecycle: ; $(PY) -m mlops.cli demo-lifecycle
monitor:  ; $(PY) -m mlops.cli monitor --html docs/latest_report.html
worker:   ; $(PY) -m mlops.cli worker --interval 300
dashboards: ; $(PY) scripts/build_dashboards.py
validate-obs: ; $(PY) scripts/validate_observability.py
observability-up: ; docker compose -f docker-compose.observability.yml up -d
observability-down: ; docker compose -f docker-compose.observability.yml down
traffic:  ; $(PY) -m mlops.cli simulate --url http://localhost:8000 --n 150 --loop-seconds 180 --pause 2 --drift-after 90 --drift-scenario market_shift
demo-deploy: ; $(PY) -m mlops.cli demo-deploy
demo:     ; $(PY) -m mlops.cli demo
ui:       ; mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
clean:    ; rm -rf mlruns mlflow.db serving.db data/*.csv artifacts .pytest_cache
