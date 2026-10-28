"""Model registry built on MLflow aliases (the modern replacement for stages).

    aliases : champion   -> serves production traffic
              challenger -> candidate under evaluation (shadow / A-B in Part 4)
    tags    : status, replaced_version, promoted_at, decision ... (full audit trail)

Rollback needs no external state: every promoted version records the version it
replaced (`replaced_version`), so rollback is just "walk the chain back one step".
"""
from __future__ import annotations
import json, time
from functools import lru_cache

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .promotion import PromotionDecision, compare_models, load_eval_set, predict_proba


class RegistryError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        uri = cfg["mlflow"]["tracking_uri"]
        mlflow.set_tracking_uri(uri); mlflow.set_registry_uri(uri)
        self.client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        self.name = cfg["mlflow"]["registered_model_name"]
        self.champion_alias = cfg["registry"]["champion_alias"]
        self.challenger_alias = cfg["registry"]["challenger_alias"]
        self._eval_df = None

    # ---------------------------------------------------------------- reads
    def get_by_alias(self, alias: str):
        try:
            return self.client.get_model_version_by_alias(self.name, alias)
        except MlflowException:
            return None

    def champion(self): return self.get_by_alias(self.champion_alias)
    def challenger(self): return self.get_by_alias(self.challenger_alias)

    def get_version(self, version: int | str):
        try:
            return self.client.get_model_version(self.name, str(version))
        except MlflowException as e:
            raise RegistryError(f"version {version} not found") from e

    def list_versions(self):
        try:
            vs = self.client.search_model_versions(f"name='{self.name}'")
        except MlflowException:
            return []
        # search results omit aliases, so re-fetch each version to get the full object
        return [self.client.get_model_version(self.name, v.version) for v in sorted(vs, key=lambda v: int(v.version))]

    def aliases_of(self, version) -> list[str]:
        return list(self.get_version(version).aliases)

    def load_model(self, ref: str | int):
        """ref: an alias ('champion') or a version number."""
        uri = f"models:/{self.name}@{ref}" if isinstance(ref, str) and not str(ref).isdigit() \
            else f"models:/{self.name}/{ref}"
        return mlflow.sklearn.load_model(uri)

    # -------------------------------------------------------------- register
    def register_run(self, run_id: str, *, set_challenger: bool = True):
        run = self.client.get_run(run_id)
        if run.data.tags.get("pipeline_status") != "success":
            raise RegistryError("only successful pipeline runs can be registered "
                                f"(status={run.data.tags.get('pipeline_status')})")
        source = run.data.tags.get("model_uri") or f"runs:/{run_id}/model"
        try:
            self.client.create_registered_model(self.name, description="Customer churn classifier")
        except MlflowException:
            pass
        mv = mlflow.register_model(source, self.name)
        m, t = run.data.metrics, run.data.tags
        for k, v in {"run_id": run_id, "status": "candidate", "algorithm": t.get("best_model", "?"),
                     "test_roc_auc": f"{m.get('test_roc_auc', float('nan')):.4f}",
                     "cv_auc": f"{m.get('cv_auc_mean', float('nan')):.4f}",
                     "data_hash": run.data.params.get("data_hash", "?"),
                     "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")}.items():
            self.client.set_model_version_tag(self.name, mv.version, k, str(v))
        if set_challenger:
            self.client.set_registered_model_alias(self.name, self.challenger_alias, mv.version)
        return self.get_version(mv.version)

    # ------------------------------------------------------------- promotion
    def _eval(self):
        if self._eval_df is None:
            self._eval_df = load_eval_set(self.cfg)
        return self._eval_df

    def evaluate_promotion(self, version: int | str) -> PromotionDecision:
        df = self._eval(); y = df[self.cfg["data"]["target"]].values
        cand_p = predict_proba(self.load_model(int(version)), df, self.cfg)
        champ = self.champion()
        champ_p = predict_proba(self.load_model(int(champ.version)), df, self.cfg) if champ else None
        return compare_models(cand_p, champ_p, y, self.cfg)

    def promote_if_better(self, version: int | str, *, force: bool = False) -> PromotionDecision:
        version = int(version); mv = self.get_version(version)
        champ = self.champion()
        if champ and int(champ.version) == version:
            raise RegistryError(f"v{version} is already champion")
        d = self.evaluate_promotion(version)
        self.client.set_model_version_tag(self.name, version, "last_decision", json.dumps(
            {"promote": d.promote, "reason": d.reason, "auc": round(d.challenger["roc_auc"], 4),
             "auc_delta": round(d.auc_delta, 4)}))
        if d.promote or force:
            self._set_champion(version, replaced=int(champ.version) if champ else None,
                               note="forced" if (force and not d.promote) else "passed policy")
        else:
            self.client.set_model_version_tag(self.name, version, "status", "rejected")
            self.client.set_model_version_tag(self.name, version, "rejection_reason", d.reason)
            cur = self.challenger()
            if cur and int(cur.version) == version:
                self.client.delete_registered_model_alias(self.name, self.challenger_alias)
        return d

    def _set_champion(self, version: int, replaced: int | None, note: str):
        c, n = self.client, self.name
        c.set_registered_model_alias(n, self.champion_alias, version)
        c.set_model_version_tag(n, version, "status", "champion")
        c.set_model_version_tag(n, version, "promoted_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        c.set_model_version_tag(n, version, "promotion_note", note)
        if replaced is not None:
            c.set_model_version_tag(n, version, "replaced_version", str(replaced))
            c.set_model_version_tag(n, replaced, "status", "archived")
        ch = self.challenger()
        if ch and int(ch.version) == version:      # it graduated, so it's no longer a challenger
            c.delete_registered_model_alias(n, self.challenger_alias)

    # -------------------------------------------------------------- rollback
    def rollback(self, reason: str = "manual rollback") -> tuple[int, int]:
        """Restore the previous champion. Returns (rolled_back_version, restored_version)."""
        cur = self.champion()
        if not cur:
            raise RegistryError("no champion to roll back")
        prev = cur.tags.get("replaced_version")
        if not prev:
            raise RegistryError(f"v{cur.version} has no predecessor to roll back to")
        c, n = self.client, self.name
        c.set_registered_model_alias(n, self.champion_alias, int(prev))
        c.set_model_version_tag(n, prev, "status", "champion")
        c.set_model_version_tag(n, prev, "restored_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        c.set_model_version_tag(n, cur.version, "status", "rolled_back")
        c.set_model_version_tag(n, cur.version, "rollback_reason", reason)
        return int(cur.version), int(prev)
