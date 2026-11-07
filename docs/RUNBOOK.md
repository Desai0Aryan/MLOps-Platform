# Runbook: what to do when an alert fires

Every alert in [`deploy/prometheus/alerts.yml`](../deploy/prometheus/alerts.yml) has an entry here (a test enforces it).
Commands assume the Docker stack; replace `docker compose exec api` with nothing if you run locally.
Useful everywhere: `python -m mlops.cli events` (what the platform decided, and why) · `docker compose logs --tail=100 api worker`.

### ApiDown
**Meaning:** Prometheus cannot scrape the API for 1 minute. **Check:** `docker compose ps`, `curl -s localhost:8000/health`, `docker compose logs api`.
**Likely causes:** container crash-loop (bad config / DB unreachable), OOM kill (limit is 1 GB). **Fix:** `docker compose up -d api`; if a bad release,
`./scripts/deploy.sh --rollback`.

### HighErrorRate
**Meaning:** > 5 % of requests return 5xx. **Check:** `curl -s localhost:8000/ready` (503 = no champion loaded), API logs for tracebacks, Postgres health.
**Likely causes:** champion failed to load from MLflow, database down. **Fix:** restore MLflow/Postgres, then `curl -X POST -H "X-Admin-Token: $ADMIN_TOKEN" localhost:8000/reload`.

### HighPredictionLatencyP95
**Meaning:** p95 `/predict*` latency above 500 ms. **Check:** Service dashboard → *Model inference time by variant*; is a heavier challenger live (`mlops deploy status`)?
**Fix:** stop the experiment (`mlops deploy stop`), scale replicas, or roll back the model.

### NoPredictionTraffic
**Meaning:** API is up but served nothing for 15 minutes. **Likely cause:** an upstream caller or network path broke, not this platform. **Check:** caller logs, load balancer.

### ChallengerFallbacks
**Meaning:** during A/B the challenger threw and the champion answered instead. **Users are unaffected, but the experiment is now invalid.**
**Fix:** `mlops deploy stop`, inspect `docker compose logs api | grep -i challenger`, fix or reject the challenger version.

### ShadowErrors
**Meaning:** the shadow challenger errors on mirrored traffic (no user impact). **Fix:** do not promote it. `mlops deploy analyze` shows `unhealthy`; the worker stops the experiment itself.

### ModelDriftCritical
**Meaning:** the monitor's overall verdict is CRITICAL for 5 minutes. **Check:** Quality dashboard; `mlops monitor --html report.html` for the per-feature breakdown.
**What happens automatically:** the worker retrains if enough labeled production data exists (≥ 1,500 rows). **If nothing happens:** see *AutoRetrainBlocked* and `mlops events`
(`retrain_skipped` tells you why: usually "waiting for labels"). **Manual override:** `mlops train --register`, then `mlops deploy shadow`.

### FeatureDriftHigh
**Meaning:** a feature's PSI > 0.25 versus the training reference. Early warning only: it does not prove the model is wrong. **Check:** is it an upstream change
(new pricing, new channel, broken ETL)? A missing-value spike points at a data pipeline bug: fix upstream, do not retrain.

### CalibrationGapHigh
**Meaning:** predicted churn rate differs from the actual rate by > 10 points: classic concept drift (AUC can still look fine). **Check:** Quality dashboard → calibration gap.
**Fix:** retrain on recent labeled data (automatic if labels exist). Consider recalibrating the threshold in the meantime.

### AucDegraded
**Meaning:** live AUC is > 0.07 below the model's training AUC. **Check:** labeled-sample size (`mlops_monitoring_rows{kind="labeled"}`); a small window is noisy. **Fix:** as above.

### MonitoringStale
**Meaning:** no monitoring run for over an hour: the **worker is down or failing**, so drift goes unnoticed and nothing retrains. **Check:** `docker compose logs worker`, `docker compose ps worker`.
**Fix:** `docker compose up -d worker`.

### AutoRetrainBlocked
**Meaning:** an automatic retrain was refused (`retrain_rejected_bad_data`, `retrain_rejected_quality`) or the shadow start failed. **Check:** `mlops events` shows the reason.
**Most common:** production data violates the data contract (e.g. > 2 % missing `usage_gb`): that is a *data incident*. Fix the pipeline upstream; do not loosen the contract to make the alert go away.

### ModelPromoted
**Meaning (info):** a challenger won its A/B test and became champion automatically. **Check:** Rollouts dashboard annotations and `mlops events`.
**If you disagree:** `python -m mlops.cli registry rollback --reason "..."` then `curl -X POST ... /reload`. (Image rollback is `./scripts/deploy.sh --rollback`; that reverts *code*, not the model.)

---
## Generic procedures
| Task | Command |
|---|---|
| Revert a bad **model** | `python -m mlops.cli registry rollback --reason "..."` then `POST /reload` (the API also polls every 30 s) |
| Revert a bad **release** | `./scripts/deploy.sh --rollback` |
| Stop any experiment | `python -m mlops.cli deploy stop` |
| Pause auto-retraining | set `retraining.enabled: false` in `configs/config.yaml`, restart the worker |
| Inspect everything the platform decided | `python -m mlops.cli events` |
