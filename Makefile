.PHONY: install data train test ui demo clean
PY ?= python
export PYTHONPATH := src

install:  ; pip install -r requirements.txt
data:     ; $(PY) -m mlops.cli generate --scenario baseline --out data/train.csv
train:    ; $(PY) -m mlops.cli train
test:     ; $(PY) -m pytest -q
demo:     ; $(PY) -m mlops.cli demo
ui:       ; mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
clean:    ; rm -rf mlruns mlflow.db data/*.csv artifacts .pytest_cache
