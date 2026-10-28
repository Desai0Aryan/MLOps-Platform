"""Thread-safe, hot-swappable model holder.

Zero-downtime reloads: the NEW model is fully loaded first, then swapped in with a
single dict assignment, so in-flight requests keep using the old model.
Keyed by role so Part 4 can hold champion + challenger + shadow simultaneously.
"""
from __future__ import annotations
import threading, time
from dataclasses import dataclass, field

from ..registry import ModelRegistry


@dataclass
class LoadedModel:
    model: object
    name: str
    version: int
    alias: str
    run_id: str
    algorithm: str
    tags: dict = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)


class ModelStore:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.registry = ModelRegistry(cfg)
        self._models: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    def get(self, role: str = "champion") -> LoadedModel | None:
        return self._models.get(role)

    def refresh(self, role: str = "champion") -> tuple[bool, LoadedModel | None]:
        """Re-resolve the alias. Returns (changed, current)."""
        alias = self.cfg["registry"][f"{role}_alias"]
        mv = self.registry.get_by_alias(alias)
        cur = self._models.get(role)
        if mv is None:
            if cur is not None:
                with self._lock:
                    self._models.pop(role, None)
            return cur is not None, None
        if cur is not None and cur.version == int(mv.version):
            return False, cur
        model = self.registry.load_model(int(mv.version))          # load OUTSIDE the lock
        loaded = LoadedModel(model, self.registry.name, int(mv.version), alias, mv.tags.get("run_id", "?"),
                             mv.tags.get("algorithm", "?"), dict(mv.tags))
        with self._lock:
            self._models[role] = loaded                               # atomic swap
        return True, loaded
