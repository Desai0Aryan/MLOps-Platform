"""Generates the Grafana dashboards (deploy/grafana/dashboards/*.json).

Dashboards-as-code: layout, thresholds and queries live here, so they're reviewable in a PR, consistent
across panels, and machine-checkable (tests/test_metrics.py verifies every query only references metrics
the API really exports, and CI validates the PromQL with promtool).

    python scripts/build_dashboards.py
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "deploy" / "grafana" / "dashboards"
DS = {"type": "prometheus", "uid": "prometheus"}
RI = "$__rate_interval"
STATUS_MAP = [{"type": "value", "options": {
    "0": {"text": "OK", "color": "green", "index": 0}, "1": {"text": "WARNING", "color": "yellow", "index": 1},
    "2": {"text": "CRITICAL", "color": "red", "index": 2}, "-1": {"text": "NO DATA", "color": "text", "index": 3}}}]


def thr(*steps):
    """thr('green', (0.1,'yellow'), (0.25,'red'))"""
    out = [{"color": steps[0], "value": None}] + [{"color": c, "value": v} for v, c in steps[1:]]
    return {"mode": "absolute", "steps": out}


def targets(*qs):
    out = []
    for i, q in enumerate(qs):
        e, leg = (q, "") if isinstance(q, str) else q
        out.append({"refId": chr(65 + i), "datasource": DS, "expr": e, "legendFormat": leg, "instant": False})
    return out


class Board:
    def __init__(self, uid, title, desc, tags):
        self.d = {"uid": uid, "title": title, "description": desc, "tags": ["mlops"] + tags, "schemaVersion": 39, "version": 1,
                  "editable": True, "graphTooltip": 1, "refresh": "15s", "time": {"from": "now-30m", "to": "now"},
                  "timezone": "browser", "templating": {"list": []}, "links": [], "panels": [],
                  "annotations": {"list": [{"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                                            "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)", "name": "Annotations & Alerts", "type": "dashboard"}]}}
        self._id, self._y = 0, 0

    def annotate(self, name, expr, color, title):
        self.d["annotations"]["list"].append({"datasource": DS, "enable": True, "name": name, "iconColor": color, "expr": expr,
                                              "titleFormat": title, "step": "15s", "useValueForTime": False})

    def row(self, title):
        self._id += 1
        self.d["panels"].append({"id": self._id, "type": "row", "title": title, "collapsed": False, "panels": [],
                                 "gridPos": {"h": 1, "w": 24, "x": 0, "y": self._y}})
        self._y += 1

    def line(self, panels, h=8, widths=None):
        widths = widths or [24 // len(panels)] * len(panels); x = 0
        for p, w in zip(panels, widths):
            self._id += 1
            p.update(id=self._id, gridPos={"h": h, "w": w, "x": x, "y": self._y}, datasource=DS)
            self.d["panels"].append(p); x += w
        self._y += h

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{self.d['uid']}.json").write_text(json.dumps(self.d, indent=2) + "\n")


def stat(title, expr, unit="none", th=None, mappings=None, desc="", legend="", text="auto", nodata="n/a", decimals=None, graph="none", color="value"):
    f = {"defaults": {"unit": unit, "thresholds": th or thr("green"), "mappings": mappings or [], "noValue": nodata, "color": {"mode": "thresholds"}}, "overrides": []}
    if decimals is not None:
        f["defaults"]["decimals"] = decimals
    return {"type": "stat", "title": title, "description": desc, "targets": targets((expr, legend)), "fieldConfig": f,
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}, "colorMode": color,
                        "graphMode": graph, "textMode": text, "justifyMode": "center", "orientation": "auto"}}


def ts(title, qs, unit="none", desc="", th=None, stack=None, minv=None, maxv=None, fill=12, step=False, th_style="off"):
    c = {"drawStyle": "line", "lineWidth": 2, "fillOpacity": fill, "showPoints": "never", "spanNulls": True,
         "lineInterpolation": "stepAfter" if step else "linear", "thresholdsStyle": {"mode": th_style}}
    if stack:
        c["stacking"] = {"mode": stack, "group": "A"}
    d = {"unit": unit, "custom": c, "thresholds": th or thr("green"), "color": {"mode": "palette-classic"}}
    if minv is not None: d["min"] = minv
    if maxv is not None: d["max"] = maxv
    return {"type": "timeseries", "title": title, "description": desc, "targets": targets(*qs), "fieldConfig": {"defaults": d, "overrides": []},
            "options": {"legend": {"displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull", "max"]}, "tooltip": {"mode": "multi", "sort": "desc"}}}


def bargauge(title, expr, legend, th, unit="none", maxv=1, desc=""):
    return {"type": "bargauge", "title": title, "description": desc, "targets": targets((expr, legend)),
            "fieldConfig": {"defaults": {"unit": unit, "min": 0, "max": maxv, "thresholds": th, "color": {"mode": "thresholds"}}, "overrides": []},
            "options": {"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True, "minVizHeight": 12, "minVizWidth": 8,
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}


def timeline(title, qs, mappings, desc=""):
    return {"type": "state-timeline", "title": title, "description": desc, "targets": targets(*qs),
            "fieldConfig": {"defaults": {"mappings": mappings, "custom": {"fillOpacity": 80, "lineWidth": 0}, "color": {"mode": "thresholds"},
                                         "thresholds": thr("green")}, "overrides": []},
            "options": {"mergeValues": True, "showValue": "auto", "alignValue": "left", "rowHeight": 0.8, "legend": {"showLegend": False}}}


def heatmap(title, expr, desc=""):
    t = targets((expr, "{{le}}")); t[0]["format"] = "heatmap"
    return {"type": "heatmap", "title": title, "description": desc, "targets": t,
            "fieldConfig": {"defaults": {"custom": {"hideFrom": {"legend": False, "tooltip": False, "viz": False}}}, "overrides": []},
            "options": {"calculate": False, "yAxis": {"axisPlacement": "left", "unit": "percentunit"}, "color": {"mode": "scheme", "scheme": "Spectral", "fill": "dark-orange", "scale": "exponential", "exponent": 0.5, "steps": 64},
                        "cellGap": 1, "legend": {"show": True}, "tooltip": {"show": True, "yHistogram": False}, "rowsFrame": {"layout": "auto"}}}


def table(title, expr, rename, hide=("Time", "__name__", "Value", "job", "instance", "service"), desc=""):
    t = targets(expr); t[0].update(format="table", instant=True)
    return {"type": "table", "title": title, "description": desc, "targets": t, "fieldConfig": {"defaults": {}, "overrides": []},
            "options": {"showHeader": True, "cellHeight": "sm"},
            "transformations": [{"id": "organize", "options": {"excludeByName": {h: True for h in hide}, "renameByName": rename}}]}


# =====================================================================================================
def service_board():
    b = Board("mlops-service", "MLOps · Service health", "Traffic, errors, latency and resources of the serving layer.", ["service"])
    b.row("Golden signals")
    b.line([
        stat("Prediction requests / s", f'sum(rate(mlops_http_requests_total{{path=~"/predict.*"}}[{RI}]))', "reqps", graph="area", decimals=2),
        stat("Error rate (5xx)", f'(sum(rate(mlops_http_requests_total{{status=~"5.."}}[{RI}])) or vector(0)) / sum(rate(mlops_http_requests_total[{RI}]))', "percentunit",
             thr("green", (0.01, "yellow"), (0.05, "red")), desc="Alert fires above 5 % for 5 minutes."),
        stat("p95 latency · /predict", f'histogram_quantile(0.95, sum by (le) (rate(mlops_http_request_duration_seconds_bucket{{path=~"/predict.*"}}[{RI}])))', "s",
             thr("green", (0.1, "yellow"), (0.5, "red")), desc="Alert fires above 500 ms."),
        stat("Predictions / s (all models)", f"sum(rate(mlops_predictions_total[{RI}]))", "ops", graph="area", decimals=2),
        stat("Champion version", 'mlops_model_version{role="champion"}', "none", thr("blue"), text="value", decimals=0),
        stat("Uptime", "time() - process_start_time_seconds", "s", thr("green"), desc="Time since this replica started."),
    ])
    b.row("Traffic & latency")
    b.line([
        ts("Requests / s by route and status", [(f"sum by (path, status) (rate(mlops_http_requests_total[{RI}]))", "{{path}} · {{status}}")], "reqps", stack="normal"),
        ts("Latency percentiles · /predict", [
            (f'histogram_quantile(0.50, sum by (le) (rate(mlops_http_request_duration_seconds_bucket{{path=~"/predict.*"}}[{RI}])))', "p50"),
            (f'histogram_quantile(0.95, sum by (le) (rate(mlops_http_request_duration_seconds_bucket{{path=~"/predict.*"}}[{RI}])))', "p95"),
            (f'histogram_quantile(0.99, sum by (le) (rate(mlops_http_request_duration_seconds_bucket{{path=~"/predict.*"}}[{RI}])))', "p99")], "s", fill=5),
    ])
    b.line([
        ts("Model inference time per row (p95) by variant", [(f"histogram_quantile(0.95, sum by (le, variant) (rate(mlops_inference_latency_seconds_bucket[{RI}])))", "{{variant}}")], "s",
           desc="Compare champion vs challenger cost before promoting a heavier model."),
        ts("Predictions / s by model version and variant", [(f"sum by (model_version, variant) (rate(mlops_predictions_total[{RI}]))", "v{{model_version}} · {{variant}}")], "ops", stack="normal"),
    ])
    b.row("Failures & feedback")
    b.line([
        ts("Prediction-path errors / s by stage", [(f"sum by (stage) (rate(mlops_prediction_errors_total[{RI}]))", "{{stage}}")], "ops",
           desc="challenger_fallback = a challenger failed during A/B and the champion answered instead (users unaffected)."),
        ts("Ground-truth labels received / s", [(f"rate(mlops_feedback_labels_total[{RI}])", "labels")], "ops"),
        ts("Process memory & CPU", [("process_resident_memory_bytes", "resident memory"), (f"rate(process_cpu_seconds_total[{RI}])", "cpu cores")], "none", fill=5),
    ], widths=[8, 8, 8])
    b.annotate("Champion changed", 'changes(mlops_model_version{role="champion"}[1m]) > 0', "green", "Champion model changed")
    return b


def quality_board():
    b = Board("mlops-quality", "MLOps · Model quality & drift",
              "Data drift (needs no labels) and live performance (needs delayed labels), measured against each model's own training reference.", ["drift", "quality"])
    b.row("Verdict")
    b.line([
        stat("Overall", "mlops_monitoring_status", mappings=STATUS_MAP, th=thr("green"), color="background", text="value", desc="Worst of data drift and live performance."),
        stat("Data drift", "mlops_drift_status", mappings=STATUS_MAP, th=thr("green"), color="background", text="value"),
        stat("Live performance", "mlops_performance_status", mappings=STATUS_MAP, th=thr("green"), color="background", text="value"),
        stat("Model monitored", "mlops_monitoring_model_version", th=thr("blue"), text="value", decimals=0),
        stat("Last monitor run", "time() - mlops_monitoring_last_run_timestamp_seconds", "s", thr("green", (900, "yellow"), (3600, "red")), desc="Alert fires if older than 1 h."),
        stat("Labeled outcomes in window", 'mlops_monitoring_rows{kind="labeled"}', th=thr("red", (150, "yellow"), (500, "green")), decimals=0,
             desc="Live-performance checks need >= 150 labels."),
    ])
    b.line([timeline("Monitor verdict over time", [("mlops_monitoring_status", "overall"), ("mlops_drift_status", "drift"), ("mlops_performance_status", "performance")], STATUS_MAP)], h=5)
    b.row("Data drift (PSI vs training reference): < 0.10 stable · 0.10-0.25 moderate · > 0.25 major")
    drift_th = thr("green", (0.10, "yellow"), (0.25, "red"))
    b.line([
        bargauge("Current PSI by feature", "sort_desc(mlops_drift_psi)", "{{feature}}", drift_th, maxv=0.6,
                 desc="'predicted_probability' is prediction drift: the model's own output distribution."),
        ts("PSI over time", [("mlops_drift_psi", "{{feature}}")], "none", th=drift_th, th_style="line", minv=0, fill=0),
    ], h=10, widths=[10, 14])
    b.row("Live performance (delayed labels)")
    b.line([
        ts("ROC-AUC: live vs training", [('mlops_performance_metric{metric="roc_auc"}', "live"), ('mlops_performance_baseline{metric="roc_auc"}', "at training")], "none", minv=0.5, maxv=1, fill=0),
        ts("Calibration gap (predicted − actual churn rate)", [('mlops_performance_metric{metric="calibration_gap"}', "gap")], "percentunit",
           th=thr("red", (-0.10, "yellow"), (-0.05, "green"), (0.05, "yellow"), (0.10, "red")), th_style="line",
           desc="The first thing concept drift breaks. Alert fires beyond ±10 %."),
    ])
    b.line([
        ts("Predicted vs actual churn rate", [('mlops_performance_metric{metric="mean_predicted"}', "predicted"), ('mlops_performance_metric{metric="actual_rate"}', "actual")], "percentunit", fill=0, minv=0),
        ts("Log-loss: live vs training", [('mlops_performance_metric{metric="log_loss"}', "live"), ('mlops_performance_baseline{metric="log_loss"}', "at training")], "none", fill=0, minv=0),
    ])
    b.line([
        heatmap("Distribution of predicted churn probability (over time)", f"sum by (le) (increase(mlops_prediction_probability_bucket[{RI}]))",
                desc="A shifting colour band is prediction drift you can see with your eyes."),
        ts("Expected calibration error & accuracy", [('mlops_performance_metric{metric="ece"}', "ECE"), ('mlops_performance_metric{metric="accuracy"}', "accuracy")], "percentunit", fill=0, minv=0, maxv=1),
    ])
    b.annotate("Model promoted", 'changes(mlops_model_version{role="champion"}[1m]) > 0', "green", "New champion")
    b.annotate("Retraining started", 'changes(mlops_lifecycle_last_event_timestamp_seconds{kind="retrain_started"}[1m]) > 0', "orange", "Auto-retrain started")
    return b


def rollout_board():
    b = Board("mlops-rollout", "MLOps · Rollouts & lifecycle", "Shadow / A-B state, challenger health and everything the platform decided on its own.", ["rollout", "lifecycle"])
    b.row("Where are we?")
    b.line([
        stat("Rollout mode", "mlops_deployment_mode == 1", legend="{{mode}}", text="name", th=thr("blue"), color="background", desc="champion_only → shadow → ab"),
        stat("Challenger traffic share", "mlops_ab_traffic_split", "percentunit", thr("green"), decimals=0),
        stat("Champion", 'mlops_model_version{role="champion"}', th=thr("green"), text="value", decimals=0, color="background"),
        stat("Challenger", 'mlops_model_version{role="challenger"}', th=thr("yellow"), text="value", decimals=0, color="background", nodata="none"),
        stat("Auto-promotions", 'sum(mlops_lifecycle_events_total{kind="challenger_promoted"})', th=thr("green"), decimals=0, nodata="0"),
        stat("Blocked / rejected", 'sum(mlops_lifecycle_events_total{kind=~"retrain_rejected_.*|challenger_rejected|shadow_start_failed"})', th=thr("green", (1, "yellow")), decimals=0, nodata="0"),
    ])
    b.line([table("Loaded models", "mlops_model_info", {"role": "Role", "version": "Version", "algorithm": "Algorithm"})], h=5)
    b.row("Traffic")
    b.line([
        ts("Traffic share by variant", [(f"sum by (variant) (rate(mlops_predictions_total[{RI}]))", "{{variant}}")], "ops", stack="percent", fill=40,
           desc="In A/B mode the challenger band should match the configured split (sample-ratio check)."),
        ts("Shadow predictions mirrored / s", [(f"sum by (status) (rate(mlops_shadow_predictions_total[{RI}]))", "{{status}}")], "ops", stack="normal"),
        ts("Challenger failures / s", [(f'sum by (stage) (rate(mlops_prediction_errors_total{{stage=~"challenger_fallback|shadow"}}[{RI}]))', "{{stage}}")], "ops"),
    ], widths=[10, 7, 7])
    b.row("Lifecycle")
    b.line([
        ts("Model versions over time", [("mlops_model_version", "{{role}}")], "none", step=True, fill=0, minv=0),
        ts("Automated decisions (cumulative)", [("mlops_lifecycle_events_total", "{{kind}}")], "none", step=True, fill=0, minv=0,
           desc="retrain_started → retrain_completed → shadow_started → ab_started → challenger_promoted."),
    ])
    b.annotate("Champion changed", 'changes(mlops_model_version{role="champion"}[1m]) > 0', "green", "Champion changed")
    b.annotate("Rollout step", 'changes(mlops_lifecycle_last_event_timestamp_seconds{kind=~"shadow_started|ab_started"}[1m]) > 0', "blue", "Rollout: {{kind}}")
    return b


if __name__ == "__main__":
    for make in (service_board, quality_board, rollout_board):
        board = make(); board.save(); print("wrote", OUT / f"{board.d['uid']}.json", f"({len(board.d['panels'])} panels)")
