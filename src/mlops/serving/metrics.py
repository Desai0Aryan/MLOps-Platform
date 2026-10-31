"""Prometheus instrumentation.

Two kinds of metrics, on purpose:

1. IN-PROCESS instruments (counters / histograms): things only this replica can see, such as request
   counts, latencies, predictions served. Prometheus sums across replicas.
2. DB-BACKED collector: platform state that is produced by OTHER processes (the worker's monitoring
   runs, the lifecycle timeline) or lives in the shared database (rollout state). Computed at scrape
   time, so the numbers are right even though the worker is a different process.

Each app gets its own CollectorRegistry (no global state), so tests and multiple apps never collide.
Run one uvicorn worker per container and scale with replicas; that keeps these counters simple.
"""
from __future__ import annotations
import datetime as dt
import threading, time

from prometheus_client import CollectorRegistry, Counter, Histogram, ProcessCollector
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from sqlalchemy import func, select

from .db import LifecycleEvent

LATENCY_BUCKETS = (.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5)
PROB_BUCKETS = tuple(round(i / 10, 1) for i in range(1, 11))
STATUS_CODE = {"ok": 0, "warning": 1, "critical": 2, "insufficient_data": -1}
MODES = ("champion_only", "shadow", "ab")


class Metrics:
    def __init__(self):
        self.registry = r = CollectorRegistry()
        self.http_requests = Counter("mlops_http_requests_total", "HTTP requests", ["method", "path", "status"], registry=r)
        self.http_latency = Histogram("mlops_http_request_duration_seconds", "End-to-end HTTP latency",
                                      ["method", "path"], buckets=LATENCY_BUCKETS, registry=r)
        self.predictions = Counter("mlops_predictions_total", "Predictions served", ["model_version", "variant"], registry=r)
        self.probability = Histogram("mlops_prediction_probability", "Distribution of predicted P(churn)",
                                     ["model_version"], buckets=PROB_BUCKETS, registry=r)
        self.inference_latency = Histogram("mlops_inference_latency_seconds", "Model inference time per row",
                                           ["variant"], buckets=LATENCY_BUCKETS, registry=r)
        self.errors = Counter("mlops_prediction_errors_total", "Failures inside the prediction path", ["stage"], registry=r)
        self.shadow = Counter("mlops_shadow_predictions_total", "Mirrored challenger predictions", ["status"], registry=r)
        self.feedback = Counter("mlops_feedback_labels_total", "Ground-truth labels received", registry=r)
        ProcessCollector(registry=r)


def _epoch(iso: str) -> float:
    return dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc).timestamp()


class PlatformCollector:
    """Computes platform-state gauges at scrape time (cached a few seconds to protect the database)."""

    def __init__(self, store, deploy, monitor, SessionLocal, ttl: float = 5.0):
        self.store, self.deploy, self.monitor, self.Session, self.ttl = store, deploy, monitor, SessionLocal, ttl
        self._cache: tuple[float, dict] | None = None
        self._lock = threading.Lock()

    def _snapshot(self) -> dict:
        with self._lock:
            if self._cache and time.monotonic() - self._cache[0] < self.ttl:
                return self._cache[1]
            snap: dict = {"events": {}, "report": None, "state": None}
            try:
                snap["state"] = self.deploy.get()
                snap["report"] = self.monitor.latest()
                with self.Session() as db:
                    snap["events"] = {k: (n, last) for k, n, last in db.execute(
                        select(LifecycleEvent.kind, func.count(), func.max(LifecycleEvent.created_at)).group_by(LifecycleEvent.kind))}
            except Exception:         # a metrics scrape must never take the service down
                pass
            self._cache = (time.monotonic(), snap)
            return snap

    def collect(self):
        snap = self._snapshot()
        info = GaugeMetricFamily("mlops_model_info", "Loaded model (value is always 1)", labels=["role", "version", "algorithm"])
        ver = GaugeMetricFamily("mlops_model_version", "Loaded model version number", labels=["role"])
        loaded = GaugeMetricFamily("mlops_model_loaded_timestamp_seconds", "When the model was loaded", labels=["role"])
        for role in ("champion", "challenger"):
            lm = self.store.get(role)
            if lm:
                info.add_metric([role, str(lm.version), lm.algorithm], 1)
                ver.add_metric([role], lm.version); loaded.add_metric([role], lm.loaded_at)
        yield from (info, ver, loaded)

        mode = GaugeMetricFamily("mlops_deployment_mode", "Active rollout mode (one-hot)", labels=["mode"])
        split = GaugeMetricFamily("mlops_ab_traffic_split", "Share of traffic routed to the challenger")
        st = snap["state"]
        if st:
            for m in MODES:
                mode.add_metric([m], 1 if st.mode == m else 0)
            split.add_metric([], st.traffic_split)
        yield mode; yield split

        ev = CounterMetricFamily("mlops_lifecycle_events", "Automated lifecycle decisions", labels=["kind"])
        ev_ts = GaugeMetricFamily("mlops_lifecycle_last_event_timestamp_seconds",
                                  "When each kind of lifecycle event last happened (robust for alerting)", labels=["kind"])
        for kind, (n, last) in sorted(snap["events"].items()):
            ev.add_metric([kind], n)
            ev_ts.add_metric([kind], last.replace(tzinfo=dt.timezone.utc).timestamp())
        yield ev; yield ev_ts

        rep = snap["report"]
        status = GaugeMetricFamily("mlops_monitoring_status", "Overall monitor verdict: 0 ok, 1 warning, 2 critical, -1 insufficient data")
        d_status = GaugeMetricFamily("mlops_drift_status", "Data-drift verdict (same encoding)")
        p_status = GaugeMetricFamily("mlops_performance_status", "Live-performance verdict (same encoding)")
        psi = GaugeMetricFamily("mlops_drift_psi", "Population Stability Index vs training reference", labels=["feature"])
        sev = GaugeMetricFamily("mlops_drift_feature_severity", "Per-feature drift severity (0/1/2)", labels=["feature"])
        perf = GaugeMetricFamily("mlops_performance_metric", "Live model-quality metric on labeled traffic", labels=["metric"])
        base = GaugeMetricFamily("mlops_performance_baseline", "Same metric measured at training time", labels=["metric"])
        rows = GaugeMetricFamily("mlops_monitoring_rows", "Rows examined by the last monitoring run", labels=["kind"])
        last = GaugeMetricFamily("mlops_monitoring_last_run_timestamp_seconds", "When the monitor last ran")
        mver = GaugeMetricFamily("mlops_monitoring_model_version", "Model version the last run examined")
        if rep:
            status.add_metric([], STATUS_CODE.get(rep["status"], -1))
            d = rep.get("drift")
            d_status.add_metric([], STATUS_CODE.get(d["status"], -1) if d else -1)
            p_status.add_metric([], STATUS_CODE.get(rep["performance"]["status"], -1))
            if d:
                for f in ([d["score"]] if d.get("score") else []) + d["features"]:
                    if f and f["psi"] is not None:
                        psi.add_metric([f["name"]], f["psi"])
                    sev.add_metric([f["name"]], STATUS_CODE.get(f["severity"], 0))
            for k, v in rep["performance"].get("metrics", {}).items():
                perf.add_metric([k], v)
            for k, v in rep["performance"].get("baseline", {}).items():
                if v is not None:
                    base.add_metric([k], v)
            rows.add_metric(["live"], rep["n_rows"]); rows.add_metric(["labeled"], rep["n_labeled"])
            last.add_metric([], _epoch(rep["created_at"])); mver.add_metric([], rep["model_version"])
        yield from (status, d_status, p_status, psi, sev, perf, base, rows, last, mver)
