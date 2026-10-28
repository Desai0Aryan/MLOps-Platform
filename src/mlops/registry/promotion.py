"""Promotion policy: should challenger replace champion?

Both models score the same fixed golden set. We then apply explicit,
auditable rules instead of "the number looks bigger":

  1. AUC improvement >= threshold           (is it actually better?)
  2. Recall guardrail                       (did we sacrifice catching churners?)
  3. Calibration guardrail (Brier)          (are the probabilities still honest?)
  4. Optional: paired-bootstrap significance (is the gain more than noise?)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score

from ..config import feature_columns
from ..data import generate_dataset


@dataclass
class Rule:
    name: str
    passed: bool
    detail: str


@dataclass
class PromotionDecision:
    promote: bool
    reason: str
    challenger: dict
    champion: dict | None
    auc_delta: float = 0.0
    ci_low: float | None = None
    ci_high: float | None = None
    rules: list[Rule] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self); return d


def load_eval_set(cfg: dict) -> pd.DataFrame:
    r = cfg["registry"]
    return generate_dataset(r["eval_rows"], "baseline", seed=r["eval_seed"])


def predict_proba(model, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    num, cat = feature_columns(cfg)
    X = df[num + cat].astype({c: "float64" for c in num})
    return model.predict_proba(X)[:, 1]


def _metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    return {"roc_auc": float(roc_auc_score(y, p)), "recall": float(recall_score(y, pred)),
            "brier": float(brier_score_loss(y, p))}


def _bootstrap_auc_delta(y, p_new, p_old, n, conf, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx_all, deltas = np.arange(len(y)), []
    for _ in range(n):
        idx = rng.choice(idx_all, len(y), replace=True)          # paired resample
        if y[idx].min() == y[idx].max():
            continue
        deltas.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_old[idx]))
    a = (1 - conf) / 2
    return float(np.quantile(deltas, a)), float(np.quantile(deltas, 1 - a))


def compare_models(challenger_p: np.ndarray, champion_p: np.ndarray | None, y: np.ndarray,
                   cfg: dict, min_auc_floor: float | None = None) -> PromotionDecision:
    pol = cfg["registry"]["promotion"]
    ch = _metrics(y, challenger_p)

    if champion_p is None:   # bootstrapping: nothing to beat, only the absolute floor applies
        floor = min_auc_floor if min_auc_floor is not None else cfg["training"]["min_auc_to_accept"]
        ok = ch["roc_auc"] >= floor
        return PromotionDecision(ok, "first model: " + ("meets quality floor" if ok else "below quality floor"),
                                 ch, None, rules=[Rule("quality_floor", ok, f"AUC {ch['roc_auc']:.4f} vs floor {floor}")])

    cp = _metrics(y, champion_p)
    delta = ch["roc_auc"] - cp["roc_auc"]
    lo, hi = _bootstrap_auc_delta(y, challenger_p, champion_p, pol["bootstrap_samples"], pol["confidence"])
    rules = [
        Rule("auc_improvement", delta >= pol["min_auc_improvement"],
             f"ΔAUC {delta:+.4f} (need ≥ {pol['min_auc_improvement']:+.4f})"),
        Rule("recall_guardrail", cp["recall"] - ch["recall"] <= pol["max_recall_drop"],
             f"Δrecall {ch['recall'] - cp['recall']:+.4f} (max drop {pol['max_recall_drop']})"),
        Rule("calibration_guardrail", ch["brier"] - cp["brier"] <= pol["max_brier_increase"],
             f"ΔBrier {ch['brier'] - cp['brier']:+.4f} (max rise {pol['max_brier_increase']})"),
        Rule("significance", (lo > 0) if pol["require_significance"] else True,
             f"{int(pol['confidence']*100)}% CI of ΔAUC [{lo:+.4f}, {hi:+.4f}]"
             + ("" if pol["require_significance"] else " (informational)")),
    ]
    ok = all(r.passed for r in rules)
    reason = "all promotion rules passed" if ok else "blocked by: " + ", ".join(r.name for r in rules if not r.passed)
    return PromotionDecision(ok, reason, ch, cp, delta, lo, hi, rules)
