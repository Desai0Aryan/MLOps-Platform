"""Automated data validation against a declarative data contract.

Runs as the first gate of the training pipeline (and later on live traffic).
Every check yields a structured result so failures are explainable, loggable
to MLflow, and renderable in a dashboard.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import numpy as np
import pandas as pd


@dataclass
class Check:
    name: str
    column: str | None
    passed: bool
    severity: str          # "error" blocks the pipeline, "warning" is reported only
    detail: str


@dataclass
class ValidationReport:
    n_rows: int
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any((not c.passed) and c.severity == "error" for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def add(self, name, column, passed, detail, severity="error"):
        self.checks.append(Check(name, column, bool(passed), severity, detail))

    def to_dict(self) -> dict:
        return {"passed": self.passed, "n_rows": self.n_rows,
                "n_checks": len(self.checks), "n_failed": len(self.failures),
                "checks": [asdict(c) for c in self.checks]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def validate_dataframe(df: pd.DataFrame, cfg: dict, require_target: bool = True) -> ValidationReport:
    schema, min_rows = cfg["schema"], cfg["data"]["min_rows"]
    rep = ValidationReport(n_rows=len(df))

    rep.add("min_rows", None, len(df) >= min_rows, f"{len(df)} rows (need >= {min_rows})")
    dup = float(df.duplicated().mean()) if len(df) else 0.0
    rep.add("duplicate_rows", None, dup < 0.05, f"{dup:.2%} duplicate rows", "warning")

    for col, rule in schema["numeric"].items():
        if col not in df.columns:
            rep.add("column_present", col, False, "missing column"); continue
        s = pd.to_numeric(df[col], errors="coerce")
        bad_type = int(s.isna().sum() - df[col].isna().sum())
        rep.add("numeric_dtype", col, bad_type == 0, f"{bad_type} non-numeric values")
        null = float(s.isna().mean())
        rep.add("null_fraction", col, null <= rule["max_null"], f"{null:.2%} nulls (max {rule['max_null']:.2%})")
        valid = s.dropna()
        if len(valid):
            lo, hi = float(valid.min()), float(valid.max())
            rep.add("value_range", col, lo >= rule["min"] and hi <= rule["max"],
                    f"observed [{lo:.1f}, {hi:.1f}] vs allowed [{rule['min']}, {rule['max']}]")

    for col, rule in schema["categorical"].items():
        if col not in df.columns:
            rep.add("column_present", col, False, "missing column"); continue
        null = float(df[col].isna().mean())
        rep.add("null_fraction", col, null <= rule["max_null"], f"{null:.2%} nulls")
        unseen = sorted(set(df[col].dropna().unique()) - set(rule["allowed"]))
        rep.add("allowed_categories", col, not unseen,
                f"unseen categories: {unseen}" if unseen else "all categories known")

    tgt = schema["target"]
    if require_target:
        if tgt["name"] not in df.columns:
            rep.add("target_present", tgt["name"], False, "target column missing")
        else:
            vals = set(df[tgt["name"]].dropna().unique())
            rep.add("target_values", tgt["name"], vals <= set(tgt["allowed"]), f"values: {sorted(vals)}")
            rate = float(df[tgt["name"]].mean())
            rep.add("target_balance", tgt["name"],
                    tgt["min_positive_rate"] <= rate <= tgt["max_positive_rate"],
                    f"positive rate {rate:.2%}")
    return rep
