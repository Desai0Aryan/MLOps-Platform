#!/usr/bin/env python3
"""Post-deploy smoke test for the whole platform. Standard library only, so it runs on any CI runner.

    python scripts/smoke_test.py --api http://localhost:8000 [--prometheus URL] [--grafana URL] [--mlflow URL]
                                 [--admin-token T] [--read-only] [--wait 300]

--read-only is for PRODUCTION: it never sends labels (synthetic labels would corrupt live performance
monitoring and retraining data) and sends at most one unlabeled prediction.
Exit code 0 = healthy, 1 = at least one check failed.
"""
import argparse, base64, json, sys, time, urllib.error, urllib.parse, urllib.request

VALID = {"tenure_months": 8, "monthly_charges": 92.5, "total_charges": 740.0, "num_support_calls": 4, "usage_gb": 130.2,
         "age": 29, "contract_type": "month-to-month", "payment_method": "electronic_check", "internet_service": "fiber"}
RESULTS: list[tuple[str, bool, str]] = []


def call(url, method="GET", data=None, headers=None, timeout=10, raw=False):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method, headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
            return r.status, (payload.decode() if raw else (json.loads(payload) if payload else None))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:   # noqa: BLE001   (connection refused, timeout, DNS ...): a failed check, never a crash
        return 0, None


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail)); print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def wait_for(fn, seconds, every=2.0):
    end = time.time() + seconds
    while time.time() < end:
        if fn():
            return True
        time.sleep(every)
    return False


def metric_sum(text: str, name: str) -> float:
    return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines() if line.startswith(name + "{") or line.startswith(name + " "))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True); ap.add_argument("--prometheus"); ap.add_argument("--grafana"); ap.add_argument("--mlflow")
    ap.add_argument("--admin-token"); ap.add_argument("--grafana-auth", default="admin:admin"); ap.add_argument("--read-only", action="store_true")
    ap.add_argument("--wait", type=int, default=120, help="seconds to wait for the API to become ready")
    a = ap.parse_args(); api = a.api.rstrip("/")

    print(f"API {api}")
    check("API becomes ready (champion model loaded)", wait_for(lambda: call(f"{api}/ready")[0] == 200, a.wait), f"waited up to {a.wait}s")
    st, info = call(f"{api}/model-info")
    check("model-info reports a champion", st == 200 and info and info["version"] >= 1, f"v{info['version']} {info['algorithm']}" if st == 200 and info else str(st))
    st, before = call(f"{api}/metrics", raw=True); before = before or ""
    check("/metrics is Prometheus text", st == 200 and ("mlops_predictions_total" in before or "mlops_model_info" in before))
    n_before = metric_sum(before or "", "mlops_predictions_total")

    st, p = call(f"{api}/predict", "POST", {**VALID, "entity_id": "smoke-test"})
    check("POST /predict returns a valid prediction", st == 200 and p and 0 <= p["probability"] <= 1 and p["request_id"], f"p={p['probability']:.3f} variant={p['variant']}" if st == 200 else str(st))
    st, _ = call(f"{api}/predict", "POST", {**VALID, "age": 999})
    check("invalid input is rejected with 422", st == 422)

    sent = 1
    if not a.read_only:
        st, batch = call(f"{api}/predict/batch", "POST", {"instances": [{**VALID, "entity_id": f"smoke-{i}"} for i in range(20)]})
        check("POST /predict/batch (20 rows)", st == 200 and batch and len(batch) == 20); sent += 20
        if st == 200:
            st, fb = call(f"{api}/feedback/batch", "POST", {"items": [{"request_id": r["request_id"], "actual": 0} for r in batch]})
            check("labels are accepted and linked", st == 200 and fb and fb["stored"] == 20, f"stored {fb['stored']}/20" if fb else str(st))
        st, stats = call(f"{api}/stats")
        check("/stats reflects logged predictions", st == 200 and stats and stats["total_predictions"] >= 21, f"total={stats['total_predictions']}" if stats else str(st))
        st, rep = call(f"{api}/monitoring/run", "POST")
        check("monitoring run succeeds", st == 200 and rep and rep["status"] in ("ok", "warning", "critical", "insufficient_data"), f"status={rep['status']}" if st == 200 else str(st))

    # counter must have moved by what we sent (proves instrumentation + the prediction path agree)
    st, after = call(f"{api}/metrics", raw=True); after = after or ""
    check("prediction counter advanced by what we sent", st == 200 and metric_sum(after, "mlops_predictions_total") - n_before >= sent, f"+{metric_sum(after, 'mlops_predictions_total') - n_before:.0f}")
    st, dep = call(f"{api}/deployment")
    check("rollout state is readable", st == 200 and dep and dep["mode"] in ("champion_only", "shadow", "ab"), f"mode={dep['mode']}" if st == 200 else str(st))
    if a.admin_token:
        st, _ = call(f"{api}/deployment", "POST", {"mode": "champion_only"})
        check("admin endpoints reject requests without the token", st == 401, f"got {st}")
        st, _ = call(f"{api}/deployment", "POST", {"mode": "champion_only"}, headers={"X-Admin-Token": a.admin_token})
        check("admin endpoints accept the right token", st == 200, f"got {st}")

    if a.mlflow:
        print(f"MLflow {a.mlflow}")
        check("MLflow server healthy", call(f"{a.mlflow}/health", raw=True)[0] == 200)
        st, models = call(f"{a.mlflow.rstrip('/')}/api/2.0/mlflow/registered-models/search")
        check("model registry is populated", st == 200 and models and len(models.get("registered_models", [])) >= 1)

    if a.prometheus:
        prom = a.prometheus.rstrip("/"); print(f"Prometheus {prom}")
        def target_up():
            st, d = call(f"{prom}/api/v1/query?" + urllib.parse.urlencode({"query": 'up{job="mlops-api"}'}))
            return st == 200 and d and d["data"]["result"] and d["data"]["result"][0]["value"][1] == "1"
        check("Prometheus is scraping the API (target up)", wait_for(target_up, 90, 3), "waited up to 90s")
        st, rules = call(f"{prom}/api/v1/rules")
        n_rules = sum(len(g["rules"]) for g in rules["data"]["groups"]) if st == 200 and rules else 0
        check("alert rules are loaded", n_rules >= 12, f"{n_rules} rules")

    if a.grafana:
        gr = a.grafana.rstrip("/"); print(f"Grafana {gr}")
        auth = {"Authorization": "Basic " + base64.b64encode(a.grafana_auth.encode()).decode()}
        check("Grafana healthy", wait_for(lambda: call(f"{gr}/api/health")[0] == 200, 60, 3))
        st, dbs = call(f"{gr}/api/search?type=dash-db&tag=mlops", headers=auth)
        check("3 MLOps dashboards are provisioned", st == 200 and dbs and len(dbs) >= 3, f"found {len(dbs) if dbs else '?'}")
        st, ds = call(f"{gr}/api/datasources/uid/prometheus", headers=auth)
        check("Prometheus datasource is provisioned", st == 200)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
