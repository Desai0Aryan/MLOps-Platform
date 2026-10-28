"""Reference profile: a compact statistical fingerprint of the training data.

Saved next to every model version. In Part 5 the drift detectors compare live
traffic against *this exact profile*, so each model is monitored against the
data it was actually trained on - not whatever the data looks like today.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

N_BINS = 10


def build_reference_profile(df: pd.DataFrame, cfg: dict) -> dict:
    prof: dict = {"n_rows": int(len(df)), "numeric": {}, "categorical": {}}
    for col in cfg["schema"]["numeric"]:
        s = df[col].dropna().astype(float)
        edges = np.unique(np.quantile(s, np.linspace(0, 1, N_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(s, bins=edges)
        prof["numeric"][col] = {
            "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
            "null_rate": float(df[col].isna().mean()),
            "quantiles": {str(q): float(v) for q, v in zip([0.05, 0.25, 0.5, 0.75, 0.95],
                                                           np.quantile(s, [0.05, 0.25, 0.5, 0.75, 0.95]))},
            "bin_edges": [None if np.isinf(e) else float(e) for e in edges],
            "bin_probs": (counts / counts.sum()).tolist(),
            "sample": s.sample(min(2000, len(s)), random_state=0).round(3).tolist(),  # for KS tests
        }
    for col in cfg["schema"]["categorical"]:
        vc = df[col].value_counts(normalize=True)
        prof["categorical"][col] = {"freqs": {str(k): float(v) for k, v in vc.items()},
                                    "null_rate": float(df[col].isna().mean())}
    tgt = cfg["data"]["target"]
    if tgt in df.columns:
        prof["target_positive_rate"] = float(df[tgt].mean())
    return prof
