"""Statistics for online model experiments.

SHADOW (paired design)  - both models score the SAME requests, so we compare them
    request-by-request. Paired tests are far more powerful than comparing two groups.
A/B    (independent arms) - each customer sees ONE model. We compare groups, and first
    verify the experiment itself is valid (Sample Ratio Mismatch).

Primary metric is per-prediction LOG LOSS: continuous, sensitive to calibration
(the failure mode concept drift causes) and well suited to t-tests.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import roc_auc_score
from sqlalchemy import func, select

from ..serving.db import GroundTruth, PredictionLog, ShadowPrediction


# ----------------------------------------------------------------------------- types
@dataclass
class Check:
    name: str
    status: str            # "pass" | "fail" | "info" | "wait"
    detail: str


@dataclass
class ExperimentReport:
    kind: str                       # shadow | ab
    verdict: str
    recommendation: str
    checks: list[Check] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _p95(x):
    return float(np.percentile(x, 95)) if len(x) else float("nan")


def _boot_auc_delta(y, p_new, p_old, n, conf, paired, seed=0, y_old=None):
    rng = np.random.default_rng(seed); out = []
    if paired:
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) > 1:
                out.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]))
    else:
        for _ in range(n):
            i, j = rng.integers(0, len(y), len(y)), rng.integers(0, len(y_old), len(y_old))
            if len(np.unique(y[i])) > 1 and len(np.unique(y_old[j])) > 1:
                out.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y_old[j], p_old[j]))
    a = (1 - conf) / 2
    return (float(np.quantile(out, a)), float(np.quantile(out, 1 - a))) if out else (float("nan"), float("nan"))


# ----------------------------------------------------------------------------- shadow
def shadow_report(df: pd.DataFrame, cfg: dict, n_errors: int = 0) -> ExperimentReport:
    """df columns: p_champ, p_chal, pred_champ, pred_chal, lat_champ, lat_chal, actual (NaN if unlabeled)."""
    ex = cfg["deployment"]["experiment"]; rep = ExperimentReport("shadow", "continue", "")
    n = len(df); lab = df.dropna(subset=["actual"]); nl = len(lab)
    rep.stats = {"n_mirrored": n, "n_labeled": nl, "n_shadow_errors": n_errors}
    if n == 0 and n_errors == 0:
        rep.recommendation = "No mirrored traffic yet."; rep.checks.append(Check("traffic", "wait", "0 requests mirrored")); return rep

    err_rate = n_errors / (n + n_errors)
    rep.checks.append(Check("shadow_health", "pass" if err_rate <= ex["max_shadow_error_rate"] else "fail",
                            f"{n_errors} challenger errors ({err_rate:.2%}, max {ex['max_shadow_error_rate']:.0%})"))
    if n == 0:   # every single shadow call failed
        rep.verdict, rep.recommendation = "unhealthy", "Challenger failed on every mirrored request. Stop and investigate."
        return rep
    agree = float((df.pred_champ == df.pred_chal).mean()); corr = float(np.corrcoef(df.p_champ, df.p_chal)[0, 1])
    rep.stats.update(agreement_rate=agree, mean_abs_prob_diff=float(np.abs(df.p_champ - df.p_chal).mean()),
                     prob_correlation=corr, mean_p_champ=float(df.p_champ.mean()), mean_p_chal=float(df.p_chal.mean()),
                     latency_ms={"champion_p95": _p95(df.lat_champ), "challenger_p95": _p95(df.lat_chal)})
    rep.checks.append(Check("agreement", "info",
                            f"models agree on {agree:.1%} of decisions · prob. correlation {corr:.3f} · "
                            f"mean P(churn) champion {df.p_champ.mean():.3f} vs challenger {df.p_chal.mean():.3f}"))

    if err_rate > ex["max_shadow_error_rate"]:
        rep.verdict, rep.recommendation = "unhealthy", "Challenger is erroring on live traffic. Stop and investigate."
        return rep
    if nl < ex["min_labeled_per_arm"]:
        rep.checks.append(Check("sample_size", "wait", f"{nl}/{ex['min_labeled_per_arm']} labeled requests"))
        rep.recommendation = f"Keep collecting: need {ex['min_labeled_per_arm'] - nl} more labeled requests."
        rep.stats["progress"] = nl / ex["min_labeled_per_arm"]; return rep
    rep.stats["progress"] = 1.0

    y = lab.actual.values.astype(int)
    d = _logloss(y, lab.p_champ.values) - _logloss(y, lab.p_chal.values)          # >0 => challenger better
    mean_d, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(nl))
    t, pval = st.ttest_1samp(d, 0.0) if d.std() > 0 else (0.0, 1.0)
    lo, hi = st.t.interval(ex["confidence"], nl - 1, loc=mean_d, scale=se) if se > 0 else (mean_d, mean_d)
    a_lo, a_hi = _boot_auc_delta(y, lab.p_chal.values, lab.p_champ.values, ex["bootstrap_samples"], ex["confidence"], True)
    rep.stats.update(logloss_delta=mean_d, logloss_p_value=float(pval),
                     auc_champion=_auc(y, lab.p_champ), auc_challenger=_auc(y, lab.p_chal), auc_delta_ci=[a_lo, a_hi],
                     calibration_gap_champion=float(lab.p_champ.mean() - y.mean()),
                     calibration_gap_challenger=float(lab.p_chal.mean() - y.mean()))
    rep.checks.append(Check("paired_logloss", "pass" if (pval < ex["alpha"] and mean_d > 0) else ("fail" if pval < ex["alpha"] else "info"),
                            f"Δlog-loss {mean_d:+.4f} (positive = challenger better) · {int(ex['confidence']*100)}% CI [{lo:+.4f}, {hi:+.4f}] · p={pval:.2g}"))
    rep.checks.append(Check("paired_auc", "info", f"AUC champion {_auc(y, lab.p_champ):.4f} → challenger {_auc(y, lab.p_chal):.4f} · ΔAUC CI [{a_lo:+.4f}, {a_hi:+.4f}]"))
    rep.checks.append(Check("calibration", "info", f"predicted−actual churn rate: champion {lab.p_champ.mean() - y.mean():+.3f} · challenger {lab.p_chal.mean() - y.mean():+.3f}"))

    if pval < ex["alpha"] and mean_d > 0:
        rep.verdict, rep.recommendation = "challenger_better", "Challenger is significantly better on live traffic. Advance to an A/B test."
    elif pval < ex["alpha"]:
        rep.verdict, rep.recommendation = "challenger_worse", "Challenger is significantly WORSE on live traffic. Stop the experiment."
    else:
        rep.verdict, rep.recommendation = "no_difference", "No statistically significant difference. Keep collecting or stop."
    return rep


# ----------------------------------------------------------------------------- A/B
def ab_report(df: pd.DataFrame, cfg: dict, expected_split: float) -> ExperimentReport:
    """df columns: arm ('champion'|'challenger'), probability, prediction, latency_ms, actual (NaN if unlabeled)."""
    ex = cfg["deployment"]["experiment"]; rep = ExperimentReport("ab", "continue", "")
    c, h = df[df.arm == "champion"], df[df.arm == "challenger"]
    n_c, n_h = len(c), len(h); n = n_c + n_h
    rep.stats = {"n": {"champion": n_c, "challenger": n_h}, "expected_split": expected_split}
    if n == 0 or n_h == 0:
        rep.recommendation = "No challenger traffic yet."; rep.checks.append(Check("traffic", "wait", "challenger has 0 requests")); return rep

    # 1) Sample Ratio Mismatch: is randomisation even working?
    chi, p_srm = st.chisquare([n_c, n_h], [n * (1 - expected_split), n * expected_split])
    srm_ok = p_srm >= ex["srm_alpha"]
    rep.checks.append(Check("sample_ratio", "pass" if srm_ok else "fail",
                            f"observed {n_h / n:.1%} challenger vs expected {expected_split:.1%} (χ² p={p_srm:.3g})"))
    rep.stats["srm_p_value"] = float(p_srm)
    if not srm_ok:
        rep.verdict, rep.recommendation = "invalid_srm", "Traffic split is broken (sample-ratio mismatch). Results cannot be trusted; fix routing and restart."
        return rep

    cl, hl = c.dropna(subset=["actual"]), h.dropna(subset=["actual"])
    for name, g in (("champion", c), ("challenger", h)):
        rep.stats[name] = {"n": len(g), "mean_probability": float(g.probability.mean()),
                           "positive_rate": float(g.prediction.mean()), "latency_p95_ms": _p95(g.latency_ms)}
    # 2) latency guardrail
    lc, lh = rep.stats["champion"]["latency_p95_ms"], rep.stats["challenger"]["latency_p95_ms"]
    lat_ok = lh <= max(lc * ex["max_latency_ratio"], lc + ex["max_latency_slack_ms"])
    rep.checks.append(Check("latency_guardrail", "pass" if lat_ok else "fail", f"p95 champion {lc:.2f} ms · challenger {lh:.2f} ms"))

    # 3) enough labeled evidence?
    mn = ex["min_labeled_per_arm"]; short = min(len(cl), len(hl))
    rep.stats["labeled"] = {"champion": len(cl), "challenger": len(hl)}
    if short < mn:
        rep.checks.append(Check("sample_size", "wait", f"labeled per arm: champion {len(cl)}, challenger {len(hl)} (need {mn})"))
        rep.stats["progress"] = short / mn
        rep.recommendation = f"Keep running: the smaller arm has {short}/{mn} labeled outcomes."; return rep
    rep.stats["progress"] = 1.0

    yc, yh = cl.actual.values.astype(int), hl.actual.values.astype(int)
    llc, llh = _logloss(yc, cl.probability.values), _logloss(yh, hl.probability.values)
    diff = float(llc.mean() - llh.mean())                                   # >0 => challenger better
    t, p_ll = st.ttest_ind(llc, llh, equal_var=False)
    se = float(np.sqrt(llc.var(ddof=1) / len(llc) + llh.var(ddof=1) / len(llh)))
    z = st.norm.ppf(1 - (1 - ex["confidence"]) / 2)
    rep.checks.append(Check("logloss_welch_t", "pass" if (p_ll < ex["alpha"] and diff > 0) else ("fail" if p_ll < ex["alpha"] else "info"),
                            f"log-loss champion {llc.mean():.4f} vs challenger {llh.mean():.4f} · Δ {diff:+.4f} "
                            f"(CI [{diff - z * se:+.4f}, {diff + z * se:+.4f}]) · p={p_ll:.2g}"))
    ac, ah = float((cl.prediction.values == yc).mean()), float((hl.prediction.values == yh).mean())
    pp = (ac * len(yc) + ah * len(yh)) / (len(yc) + len(yh))
    zacc = (ah - ac) / np.sqrt(pp * (1 - pp) * (1 / len(yc) + 1 / len(yh))) if 0 < pp < 1 else 0.0
    p_acc = float(2 * (1 - st.norm.cdf(abs(zacc))))
    rep.checks.append(Check("accuracy_z_test", "info", f"accuracy champion {ac:.3f} vs challenger {ah:.3f} · p={p_acc:.2g}"))
    a_lo, a_hi = _boot_auc_delta(yh, hl.probability.values, cl.probability.values, ex["bootstrap_samples"], ex["confidence"], False, y_old=yc)
    rep.checks.append(Check("auc_bootstrap", "info", f"AUC champion {_auc(yc, cl.probability):.4f} vs challenger {_auc(yh, hl.probability):.4f} · ΔAUC CI [{a_lo:+.4f}, {a_hi:+.4f}]"))
    rep.checks.append(Check("calibration", "info", f"predicted−actual churn: champion {cl.probability.mean() - yc.mean():+.3f} · challenger {hl.probability.mean() - yh.mean():+.3f}"))
    rep.stats.update(logloss_delta=diff, logloss_p_value=float(p_ll), accuracy={"champion": ac, "challenger": ah},
                     auc={"champion": _auc(yc, cl.probability), "challenger": _auc(yh, hl.probability)}, auc_delta_ci=[a_lo, a_hi])

    if p_ll < ex["alpha"] and diff > 0:
        if lat_ok:
            rep.verdict, rep.recommendation = "promote", "Challenger wins with statistical significance and passes all guardrails. Promote it."
        else:
            rep.verdict, rep.recommendation = "hold", "Challenger is better but fails the latency guardrail. Do not promote until fixed."
    elif p_ll < ex["alpha"]:
        rep.verdict, rep.recommendation = "stop", "Challenger is significantly worse. Stop the experiment and keep the champion."
    else:
        rep.verdict, rep.recommendation = "no_difference", "No significant difference detected. The extra risk of switching isn't justified yet."
    return rep


# ----------------------------------------------------------------------------- DB loaders
def load_shadow_frame(engine, since, challenger_version) -> tuple[pd.DataFrame, int]:
    pl, sp, gt = PredictionLog, ShadowPrediction, GroundTruth
    q = (select(pl.request_id, pl.probability.label("p_champ"), pl.prediction.label("pred_champ"),
                pl.latency_ms.label("lat_champ"), sp.probability.label("p_chal"), sp.prediction.label("pred_chal"),
                sp.latency_ms.label("lat_chal"), gt.actual)
         .join(sp, sp.request_id == pl.request_id).join(gt, gt.request_id == pl.request_id, isouter=True)
         .where(pl.created_at >= since, sp.model_version == challenger_version, sp.error.is_(None)))
    df = pd.read_sql(q, engine)
    with engine.connect() as conn:
        errs = conn.scalar(select(func.count(sp.id)).where(sp.created_at >= since, sp.model_version == challenger_version, sp.error.is_not(None))) or 0
    return df, int(errs)


def load_ab_frame(engine, since) -> pd.DataFrame:
    pl, gt = PredictionLog, GroundTruth
    q = (select(pl.role.label("arm"), pl.probability, pl.prediction, pl.latency_ms, gt.actual)
         .join(gt, gt.request_id == pl.request_id, isouter=True)
         .where(pl.created_at >= since, pl.role.in_(["champion", "challenger"])))
    return pd.read_sql(q, engine)
