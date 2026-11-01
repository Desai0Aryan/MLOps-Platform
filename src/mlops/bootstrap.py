"""First-run bootstrap (the `bootstrap` container in docker-compose).

Idempotent by design: safe to run on every `docker compose up`.
  1. wait until the MLflow tracking server answers
  2. if a champion already exists -> do nothing
  3. otherwise train an initial model on baseline data, register it and promote it
so the API has something to serve the moment it starts.
"""
from __future__ import annotations
import time, urllib.error, urllib.request
from typing import Callable

from .data import generate_dataset
from .registry import ModelRegistry
from .training import run_training_pipeline


def wait_for_mlflow(uri: str, timeout: float = 120, log: Callable[[str], None] = print) -> None:
    if not uri.startswith("http"):
        return                                           # file/sqlite store: nothing to wait for
    deadline, last = time.time() + timeout, None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=3) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError) as e:
            last = e
        log(f"waiting for MLflow at {uri} ...")
        time.sleep(2)
    raise TimeoutError(f"MLflow not reachable at {uri} after {timeout}s: {last}")


def bootstrap(cfg: dict, rows: int | None = None, timeout: float = 120, log: Callable[[str], None] = print) -> dict:
    wait_for_mlflow(cfg["mlflow"]["tracking_uri"], timeout, log)
    reg = ModelRegistry(cfg)
    champ = reg.champion()
    if champ is not None:
        log(f"champion v{champ.version} already exists: nothing to do")
        return {"action": "exists", "version": int(champ.version)}
    n = rows or cfg["data"]["n_train"]
    log(f"no champion found: training an initial model on {n:,} baseline rows ...")
    df = generate_dataset(n, "baseline", seed=cfg["project"]["seed"])
    res = run_training_pipeline(df, cfg, run_name="bootstrap", tags={"trigger": "bootstrap"})
    mv = reg.register_run(res.run_id)
    reg.promote_version(mv.version, note="bootstrap")
    log(f"promoted v{mv.version} ({res.best_model}, AUC {res.metrics['roc_auc']:.3f}) to champion")
    return {"action": "created", "version": int(mv.version), "auc": res.metrics["roc_auc"]}
