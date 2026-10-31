"""Synthetic telecom-churn data with *controllable* drift.

Real production data changes in two distinct ways, and the generator can
produce both so the monitoring in later parts has something real to detect:

* **Data (covariate) drift**  - P(X) changes: customers get older, prices rise,
  the payment mix shifts. The model's inputs look different.
* **Concept drift**           - P(y|X) changes: the *relationship* between the
  features and churn changes (e.g. a competitor makes fiber customers leave).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

CONTRACTS = ["month-to-month", "one-year", "two-year"]
PAYMENTS = ["credit_card", "bank_transfer", "electronic_check", "mailed_check"]
INTERNET = ["fiber", "dsl", "none"]


@dataclass(frozen=True)
class DriftScenario:
    name: str
    description: str
    age_shift: float = 0.0               # years added to mean age
    price_multiplier: float = 1.0        # inflation on monthly charges
    support_call_multiplier: float = 1.0
    contract_probs: tuple = (0.55, 0.25, 0.20)
    payment_probs: tuple = (0.30, 0.25, 0.25, 0.20)
    internet_probs: tuple = (0.45, 0.40, 0.15)
    usage_missing_rate: float = 0.01     # data-quality degradation
    # concept drift: extra churn pressure that the original model never saw
    fiber_churn_boost: float = 0.0
    price_sensitivity_boost: float = 0.0


SCENARIOS: dict[str, DriftScenario] = {
    "baseline": DriftScenario("baseline", "Training-time world. No drift."),
    "mild_drift": DriftScenario(
        "mild_drift", "Slight demographic + pricing shift (covariate drift only).",
        age_shift=3, price_multiplier=1.08, support_call_multiplier=1.15,
    ),
    "severe_drift": DriftScenario(
        "severe_drift", "Big covariate shift: younger users, inflation, new payment mix.",
        age_shift=-9, price_multiplier=1.35, support_call_multiplier=1.8,
        contract_probs=(0.75, 0.15, 0.10), payment_probs=(0.15, 0.10, 0.55, 0.20),
        usage_missing_rate=0.08,
    ),
    "market_shift": DriftScenario(
        "market_shift", "Regime change: new customer mix, inflation AND a competitor (covariate + concept drift).",
        age_shift=-6, price_multiplier=1.25, support_call_multiplier=1.5,
        contract_probs=(0.70, 0.18, 0.12), payment_probs=(0.18, 0.12, 0.50, 0.20),
        usage_missing_rate=0.015, fiber_churn_boost=0.7, price_sensitivity_boost=0.018,
    ),
    "concept_drift": DriftScenario(
        "concept_drift", "Competitor launches: fiber + price-sensitive customers churn far more.",
        fiber_churn_boost=1.6, price_sensitivity_boost=0.035,
    ),
}


def generate_dataset(n: int = 20_000, scenario: str | DriftScenario = "baseline",
                     seed: int = 42, with_target: bool = True) -> pd.DataFrame:
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACTS, n, p=sc.contract_probs)
    payment = rng.choice(PAYMENTS, n, p=sc.payment_probs)
    internet = rng.choice(INTERNET, n, p=sc.internet_probs)

    age = np.clip(rng.normal(46 + sc.age_shift, 14, n), 18, 95).round()
    tenure = np.clip(rng.gamma(2.2, 14, n), 0, 119).round()

    base_price = np.where(internet == "fiber", 85, np.where(internet == "dsl", 55, 25))
    monthly = np.clip(rng.normal(base_price, 12, n) * sc.price_multiplier, 10, 249)
    total = np.clip(monthly * tenure * rng.uniform(0.9, 1.05, n), 0, 29_999)
    calls = np.clip(rng.poisson(1.6 * sc.support_call_multiplier, n), 0, 29)
    usage = np.clip(rng.lognormal(4.2, 0.9, n), 0, 1999)
    usage = np.where(internet == "none", usage * 0.05, usage)

    df = pd.DataFrame({
        "tenure_months": tenure, "monthly_charges": monthly.round(2),
        "total_charges": total.round(2), "num_support_calls": calls,
        "usage_gb": usage.round(1), "age": age,
        "contract_type": contract, "payment_method": payment, "internet_service": internet,
    })
    # data-quality degradation: random missing usage values
    miss = rng.random(n) < sc.usage_missing_rate
    df.loc[miss, "usage_gb"] = np.nan

    if with_target:
        z = (-1.1
             - 0.035 * tenure
             + 0.020 * (monthly - 60) + sc.price_sensitivity_boost * (monthly - 60)
             + 0.28 * calls
             + np.where(contract == "month-to-month", 1.1, np.where(contract == "one-year", -0.2, -1.0))
             + np.where(payment == "electronic_check", 0.55, 0.0)
             + np.where(internet == "fiber", 0.45 + sc.fiber_churn_boost, 0.0)
             - 0.012 * (age - 46)
             + rng.normal(0, 0.35, n))
        p = 1 / (1 + np.exp(-z))
        df["churned"] = (rng.random(n) < p).astype(int)
    return df
