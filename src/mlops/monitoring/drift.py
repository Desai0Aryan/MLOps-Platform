"""Data-drift detectors. No labels required: this is the EARLY-WARNING system.

Per feature we compute, against the model's own training-time reference profile:
  * PSI (Population Stability Index): practical effect size, drives severity
  * KS test (numeric) / chi-square (categorical): statistical significance
  * missing-value rate change: catches broken upstream pipelines
Statistical significance alone is NOT used for severity: with thousands of rows,
trivial shifts are "significant". PSI measures whether the shift actually matters.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from scipy import stats as st

SEV = {"ok": 0, "warning": 1, "critical": 2}
EPS = 1e-4


def worst(*levels: str) -> str:
    return max(levels, key=lambda s: SEV.get(s, 0)) if levels else "ok"


def psi(ref_p, live_p) -> float:
    r = np.clip(np.asarray(ref_p, float), EPS, None); l = np.clip(np.asarray(live_p, float), EPS, None)
    r, l = r / r.sum(), l / l.sum()
    return float(np.sum((l - r) * np.log(l / r)))


def _psi_sev(v: float, c: dict) -> str:
    return "critical" if v >= c["psi_crit"] else ("warning" if v >= c["psi_warn"] else "ok")


def _null_sev(delta: float, c: dict) -> str:
    return "critical" if delta >= c["null_delta_crit"] else ("warning" if delta >= c["null_delta_warn"] else "ok")


@dataclass
class FeatureDrift:
    name: str
    kind: str                      # numeric | categorical | score
    severity: str
    psi: float | None
    p_value: float | None
    test: str
    significant: bool
    null_rate_ref: float
    null_rate_live: float
    shift: float | None            # numeric: (live mean - ref mean) / ref std
    labels: list = field(default_factory=list)
    ref_dist: list = field(default_factory=list)
    live_dist: list = field(default_factory=list)
    note: str = ""

    def to_dict(self): return asdict(self)


@dataclass
class DriftReport:
    status: str
    n_rows: int
    features: list[FeatureDrift]
    score: FeatureDrift | None
    summary: str
    counts: dict

    def to_dict(self):
        return {"status": self.status, "n_rows": self.n_rows, "summary": self.summary, "counts": self.counts,
                "features": [f.to_dict() for f in self.features], "score": self.score.to_dict() if self.score else None}


def _edges(prof: dict) -> np.ndarray:
    e = [(-np.inf if v is None else v) for v in prof["bin_edges"]]
    e[0], e[-1] = -np.inf, np.inf
    return np.array(e, float)


def _bin_labels(edges: np.ndarray) -> list[str]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(f"≤{hi:.3g}" if np.isinf(lo) else (f">{lo:.3g}" if np.isinf(hi) else f"{lo:.3g}–{hi:.3g}"))
    return out


def numeric_drift(name: str, series: pd.Series, prof: dict, cfg: dict, kind: str = "numeric") -> FeatureDrift:
    c = cfg["monitoring"]["drift"]
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float).values
    null_live = float(series.isna().mean()) if len(series) else 0.0
    null_sev = _null_sev(null_live - prof.get("null_rate", 0.0), c)
    if len(vals) < 2:
        return FeatureDrift(name, kind, worst("critical", null_sev), None, None, "ks", False, prof.get("null_rate", 0.0),
                            null_live, None, note="no usable values")
    edges = _edges(prof)
    counts, _ = np.histogram(vals, bins=edges)
    live_p = counts / counts.sum()
    ref_p = np.array(prof["bin_probs"], float)
    v = psi(ref_p, live_p)
    ks = st.ks_2samp(np.array(prof["sample"], float), vals)
    sd = prof["std"] if prof["std"] > 0 else 1.0
    shift = float((vals.mean() - prof["mean"]) / sd)
    sev = worst(_psi_sev(v, c), null_sev)
    note = f"mean shifted {shift:+.2f}σ" if abs(shift) >= 0.2 else ""
    if null_sev != "ok":
        note = (note + " · " if note else "") + f"missing {prof.get('null_rate', 0):.1%}→{null_live:.1%}"
    return FeatureDrift(name, kind, sev, v, float(ks.pvalue), "ks", bool(ks.pvalue < c["ks_alpha"]),
                        prof.get("null_rate", 0.0), null_live, shift, _bin_labels(edges), ref_p.tolist(), live_p.tolist(), note)


def categorical_drift(name: str, series: pd.Series, prof: dict, cfg: dict) -> FeatureDrift:
    c = cfg["monitoring"]["drift"]
    ref = prof["freqs"]; live = series.dropna().astype(str).value_counts()
    cats = list(ref) + sorted(set(live.index) - set(ref))
    unseen = sorted(set(live.index) - set(ref))
    n = int(live.sum()); null_live = float(series.isna().mean()) if len(series) else 0.0
    if n == 0:
        return FeatureDrift(name, "categorical", "critical", None, None, "chi2", False, prof.get("null_rate", 0.0), null_live, None, note="no usable values")
    ref_p = np.array([ref.get(k, 0.0) for k in cats]); counts = np.array([live.get(k, 0) for k in cats], float)
    live_p = counts / counts.sum()
    v = psi(ref_p, live_p)
    expected = np.clip(ref_p, EPS, None); expected = expected / expected.sum() * n
    p = float(st.chisquare(counts, expected).pvalue)
    sev = worst(_psi_sev(v, c), _null_sev(null_live - prof.get("null_rate", 0.0), c))
    if unseen:
        sev = worst(sev, "warning")
    top = max(range(len(cats)), key=lambda i: abs(live_p[i] - ref_p[i]))
    note = f"{cats[top]}: {ref_p[top]:.0%}→{live_p[top]:.0%}" + (f" · NEW categories {unseen}" if unseen else "")
    return FeatureDrift(name, "categorical", sev, v, p, "chi2", bool(p < c["ks_alpha"]), prof.get("null_rate", 0.0),
                        null_live, None, cats, ref_p.tolist(), live_p.tolist(), note)


def detect_drift(df: pd.DataFrame, profile: dict, cfg: dict, scores=None) -> DriftReport:
    c = cfg["monitoring"]["drift"]
    feats = [numeric_drift(col, df[col], p, cfg) for col, p in profile["numeric"].items() if col in df]
    feats += [categorical_drift(col, df[col], p, cfg) for col, p in profile["categorical"].items() if col in df]
    score = None
    if scores is not None and "score" in profile and len(scores):
        score = numeric_drift("predicted_probability", pd.Series(scores), profile["score"], cfg, kind="score")
    counts = {s: sum(f.severity == s for f in feats) for s in ("ok", "warning", "critical")}
    share = counts["critical"] / max(len(feats), 1)
    null_incident = any(f.null_rate_live - f.null_rate_ref >= c["null_delta_crit"] for f in feats)
    status = "ok"
    if counts["warning"] or counts["critical"] or (score and score.severity != "ok"):
        status = "warning"
    if share >= c["critical_feature_share"] or (score and score.severity == "critical") or null_incident:
        status = "critical"
    top = sorted(feats, key=lambda f: -(f.psi or 0))[:3]
    summary = (f"{counts['critical']} critical / {counts['warning']} warning / {counts['ok']} stable features"
               + (f" · prediction drift PSI {score.psi:.2f}" if score and score.psi is not None else "")
               + (" · top: " + ", ".join(f"{f.name} ({f.psi:.2f})" for f in top if f.psi) if status != "ok" else ""))
    return DriftReport(status, len(df), feats, score, summary, counts)
