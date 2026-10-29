"""Request schemas are GENERATED from the same data contract used in training.

One source of truth: change a range or category in configs/config.yaml and the
training validator, the API validator and the OpenAPI docs all change together.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, create_model

EXAMPLE = {"tenure_months": 8, "monthly_charges": 92.5, "total_charges": 740.0, "num_support_calls": 4,
           "usage_gb": 130.2, "age": 29, "contract_type": "month-to-month",
           "payment_method": "electronic_check", "internet_service": "fiber", "entity_id": "cust-10492"}


def build_request_model(cfg: dict):
    fields: dict = {}
    for col, r in cfg["schema"]["numeric"].items():
        if r["max_null"] > 0:
            fields[col] = (Optional[float], Field(None, ge=r["min"], le=r["max"], description="may be null"))
        else:
            fields[col] = (float, Field(..., ge=r["min"], le=r["max"]))
    for col, r in cfg["schema"]["categorical"].items():
        fields[col] = (Literal[tuple(r["allowed"])], Field(...))
    fields["entity_id"] = (Optional[str], Field(None, max_length=64, description="your customer id (optional)"))
    return create_model("PredictionRequest",
                        __config__=ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE}), **fields)


class PredictionResponse(BaseModel):
    request_id: str
    probability: float = Field(description="P(churn)")
    prediction: int
    risk: Literal["low", "medium", "high"]
    model_name: str
    model_version: int
    variant: Literal["champion", "challenger"] = "champion"
    latency_ms: float


class FeedbackRequest(BaseModel):
    request_id: str
    actual: Literal[0, 1]


class FeedbackBatch(BaseModel):
    items: list[FeedbackRequest] = Field(max_length=5000)
