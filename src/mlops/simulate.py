"""Traffic simulator: replays synthetic customers (any drift scenario) against the live API,
then sends delayed ground-truth labels for a fraction of them.

Works with a real server (httpx.Client(base_url=...)) or FastAPI's TestClient (same interface).
In Part 5 this is how we *cause* drift on purpose and watch the monitors catch it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd

from .data import generate_dataset


@dataclass
class SimulationSummary:
    sent: int
    labeled: int
    versions: dict
    mean_probability: float
    actual_churn_rate: float
    errors: int


def _records(df: pd.DataFrame, feats: list[str], prefix: str) -> list[dict]:
    out = []
    for i, row in enumerate(df[feats].to_dict("records")):
        row = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        for k in ("tenure_months", "total_charges", "monthly_charges", "usage_gb", "age", "num_support_calls"):
            if row.get(k) is not None:
                row[k] = float(row[k])
        row["entity_id"] = f"{prefix}-{i}"
        out.append(row)
    return out


def simulate(client, *, scenario: str = "baseline", n: int = 500, label_rate: float = 0.7,
             seed: int = 123, batch_size: int = 100, progress: Callable[[str, float], None] | None = None
             ) -> SimulationSummary:
    emit = progress or (lambda *_: None)
    df = generate_dataset(n, scenario, seed=seed)
    feats = [c for c in df.columns if c != "churned"]
    recs = _records(df, feats, f"{scenario}-{seed}")
    rng = np.random.default_rng(seed)
    results, errors = [], 0

    for s in range(0, n, batch_size):
        r = client.post("/predict/batch", json={"instances": recs[s:s + batch_size]})
        if r.status_code != 200:
            errors += len(recs[s:s + batch_size]); continue
        results.extend(zip(r.json(), df["churned"].iloc[s:s + batch_size]))
        emit("sending predictions", 0.7 * min(s + batch_size, n) / n)

    # delayed labels: only some customers' outcomes are ever observed
    items = [{"request_id": res["request_id"], "actual": int(y)} for res, y in results if rng.random() < label_rate]
    for s in range(0, len(items), 1000):
        client.post("/feedback/batch", json={"items": items[s:s + 1000]})
        emit("sending ground-truth labels", 0.7 + 0.3 * min(s + 1000, len(items)) / max(len(items), 1))
    emit("done", 1.0)

    versions: dict = {}
    for res, _ in results:
        versions[res["model_version"]] = versions.get(res["model_version"], 0) + 1
    return SimulationSummary(len(results), len(items), versions,
                             float(np.mean([r["probability"] for r, _ in results])) if results else 0.0,
                             float(df["churned"].mean()), errors)
