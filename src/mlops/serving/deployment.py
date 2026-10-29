"""Rollout state + deterministic traffic assignment."""
from __future__ import annotations
import hashlib, threading, time
from dataclasses import dataclass
import datetime as dt

from .db import DeploymentState, utcnow


def assign_arm(key: str, split: float, salt: str) -> str:
    """Sticky, stateless assignment: the same customer always lands in the same arm.
    SHA-256 -> uniform [0,1). No lookup table, works identically on every replica."""
    h = int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
    return "challenger" if h < split else "champion"


@dataclass(frozen=True)
class State:
    mode: str
    traffic_split: float
    challenger_version: int | None
    started_at: dt.datetime
    updated_at: dt.datetime

    def to_dict(self) -> dict:
        return {"mode": self.mode, "traffic_split": self.traffic_split,
                "challenger_version": self.challenger_version,
                "experiment_started_at": self.started_at.isoformat(), "updated_at": self.updated_at.isoformat()}


def _to_state(r: DeploymentState) -> State:
    return State(r.mode, r.traffic_split, r.challenger_version, r.started_at, r.updated_at)


class DeploymentManager:
    def __init__(self, SessionLocal, cache_seconds: float = 2.0):
        self.Session, self.ttl = SessionLocal, cache_seconds
        self._cache: tuple[float, State] | None = None
        self._lock = threading.Lock()

    def get(self) -> State:
        c = self._cache
        if c and self.ttl > 0 and time.monotonic() - c[0] < self.ttl:
            return c[1]
        with self.Session() as db:
            row = db.get(DeploymentState, 1)
            if row is None:
                row = DeploymentState(id=1); db.add(row); db.commit()
            st = _to_state(row)
        self._cache = (time.monotonic(), st)
        return st

    def set(self, mode: str, split: float, challenger_version: int | None) -> State:
        """Any change restarts the experiment clock, because mixing data from different
        splits would invalidate the statistics."""
        with self._lock, self.Session() as db:
            row = db.get(DeploymentState, 1) or DeploymentState(id=1)
            row.mode, row.traffic_split, row.challenger_version = mode, split, challenger_version
            row.started_at = row.updated_at = utcnow()
            db.merge(row); db.commit()
            st = _to_state(db.get(DeploymentState, 1))
        self._cache = (time.monotonic(), st)
        return st
