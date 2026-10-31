"""Live model-quality monitoring on delayed ground truth.

This catches what feature drift cannot: CONCEPT drift, where inputs look identical
but their relationship with the outcome changed. Calibration is tracked explicitly
because that is what breaks first (ranking quality / AUC often survives).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .drift import worst


@dataclass
class PerfCheck:
    name: str
    severity: str
    detail: str


@dataclass
class PerformanceReport:
    status: str                 # ok | warning | critical | insufficient_data
    n_labeled: int
    metrics: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    checks: list[PerfCheck] = field(default_factory=list)
    summary: str = ""

    def to_dict(self): return asdict(self)


def expected_calibration_error(y, p, bins=10) -> float:
    idx = np.minimum((np.asarray(p) * bins).astype(int), bins - 1); ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(np.mean(p[m]) - np.mean(y[m]))
    return float(ece)


def performance_report(y, p, baseline: dict, cfg: dict) -> PerformanceReport:
    c = cfg["monitoring"]["performance"]
    y = np.asarray(y, int); p = np.asarray(p, float); n = len(y)
    if n < c["min_labeled"] or len(np.unique(y)) < 2:
        return PerformanceReport("insufficient_data", n, baseline=baseline,
                                 summary=f"{n}/{c['min_labeled']} labeled predictions: waiting for ground truth")
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    m = {"roc_auc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, p >= 0.5)),
         "log_loss": float(log_loss(y, pc)), "brier": float(brier_score_loss(y, p)),
         "ece": expected_calibration_error(y, p), "mean_predicted": float(p.mean()), "actual_rate": float(y.mean()),
         "calibration_gap": float(p.mean() - y.mean())}
    checks = []
    if "roc_auc" in baseline:
        drop = baseline["roc_auc"] - m["roc_auc"]
        sev = "critical" if drop >= c["auc_drop_crit"] else ("warning" if drop >= c["auc_drop_warn"] else "ok")
        checks.append(PerfCheck("auc_vs_training", sev, f"AUC {m['roc_auc']:.3f} vs {baseline['roc_auc']:.3f} at training (drop {drop:+.3f})"))
    gap = abs(m["calibration_gap"])
    sev = "critical" if gap >= c["calib_gap_crit"] else ("warning" if gap >= c["calib_gap_warn"] else "ok")
    checks.append(PerfCheck("calibration_gap", sev, f"predicted churn {m['mean_predicted']:.1%} vs actual {m['actual_rate']:.1%} (gap {m['calibration_gap']:+.1%})"))
    if baseline.get("log_loss"):
        inc = m["log_loss"] / baseline["log_loss"] - 1
        sev = "critical" if inc >= c["logloss_increase_crit"] else ("warning" if inc >= c["logloss_increase_warn"] else "ok")
        checks.append(PerfCheck("log_loss_vs_training", sev, f"log-loss {m['log_loss']:.3f} vs {baseline['log_loss']:.3f} at training ({inc:+.0%})"))
    status = worst(*[k.severity for k in checks])
    return PerformanceReport(status, n, m, baseline, checks,
                             f"AUC {m['roc_auc']:.3f} · accuracy {m['accuracy']:.1%} · calibration gap {m['calibration_gap']:+.1%}")
