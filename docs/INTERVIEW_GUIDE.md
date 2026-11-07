# Interview guide: talking about this project

Each answer points at evidence you can show (a test, a demo, a file). Lead with the *why*, then the *proof*.

**1. "What's the one-sentence pitch?"**
A mini production ML platform that doesn't just train a model but versions it, serves it, rolls changes out safely (shadow → A/B), watches it for drift, and, with guard-rails, retrains and promotes a better one by itself. `make demo-lifecycle` shows the whole story in a minute.

**2. "Why is log-loss your primary online metric?"**
Concept drift breaks *calibration* before ranking: in my simulator the model kept AUC ≈ 0.8 while predicting 35 % churn against a real 48 %. Log-loss punishes exactly that and is continuous, so a t-test applies. Proof: `test_concept_drift_is_caught_by_live_performance_only`, `docs/DESIGN.md §3`.

**3. "Covariate drift vs concept drift: how do you catch each?"**
Covariate drift (P(X) changes) needs no labels: PSI per feature plus prediction drift. Concept drift (P(y|X) changes) leaves every feature stable, so only label-based monitoring (calibration gap, AUC vs training) catches it. I have a test proving feature monitors are blind to it and the performance monitor isn't.

**4. "Why PSI instead of just KS p-values?"**
With thousands of rows everything is statistically significant. PSI is an effect size, so severity means "does this shift matter". KS/χ² are still reported as supporting evidence.

**5. "Why shadow *and* A/B?"**
Shadow scores the *same* requests with both models, so the comparison is paired (much more powerful) and has zero user risk. A/B is the final confirmation under real routing, with a sample-ratio-mismatch check first. Together they need less exposure than A/B alone.

**6. "What is SRM and why check it?"**
Sample Ratio Mismatch: the observed traffic split differs from the configured one, meaning randomisation is broken and any result is untrustworthy. I test it with a χ² on arm counts before looking at metrics (`test_ab_detects_sample_ratio_mismatch`).

**7. "How do you deal with labels arriving weeks later?"**
Predictions are immutable events; labels live in a separate table joined on `request_id`. Monitoring reports `insufficient_data` until enough labels exist; the orchestrator *waits* rather than retraining on a thin slice (it logs `retrain_skipped: waiting for ground truth`). Honest caveat: peeking at a running test inflates false positives; I gate on minimum sample size and would add sequential testing next.

**8. "Why not retrain automatically whenever drift fires?"**
Because drift can be a data bug, not a world change. If production data violates the data contract (e.g. 6 % missing values against a 2 % limit) the orchestrator **refuses to retrain**, since it would bake the bug into the model. I hit this in a real run and kept it as a test (`test_gate_refuses_to_retrain_on_contract_violating_data`). Other brakes: cooldown, quality gate, min labeled rows, experiment timeout.

**9. "How do you roll back?"**
Three independent levels: *model* (`registry rollback`: a chain of `replaced_version` tags, no external state), *experiment* (stop → champion untouched), *release* (`deploy.sh` redeploys the previous image tag automatically if `/ready` never turns healthy, tested with a fake docker: old → bad → old). Note rolling back an image does not roll back a model.

**10. "What happens if the challenger crashes in production?"**
In A/B the request falls back to the champion (user never sees it) and the failure is counted/alerted; in shadow it's recorded, never raised, and the verdict becomes `unhealthy`. Tests: `test_challenger_failure_falls_back_to_champion`, `test_shadow_errors_are_recorded_not_raised`.

**11. "Tell me about a bug you found."**
Pick one: (a) the **smoke test** found that clients replying with feedback instantly raced my background-task persistence, which unit tests never saw because TestClient runs background tasks early; fixed with a grace period plus a regression test. (b) `actionlint` caught unquoted `: ` in step names that would have broken CI on its first run. (c) Running the dashboards' queries against a real Prometheus showed a path filter that missed `/predict/batch`. The lesson: validate with the real thing, not just parse.

**12. "How do you know your CI works if you never ran it?"**
I'm explicit that the workflows weren't executed on GitHub from my sandbox. What I did: `actionlint` (incl. shellcheck on every `run:` block), and I ran each job's logic locally: tests on a clean-room install of the latest dependency versions, real PostgreSQL 16, real `promtool`/`hadolint`, `docker compose config`, the smoke test against a faithful rehearsal of the stack, the deploy script against a fake docker, and the model gate in both directions (passes on good code, fails on injected regressions).

**13. "What breaks first at 100× traffic?"**
The synchronous-ish pieces: single-process counters (→ scale replicas, already designed for it), prediction-log writes (→ batch/queue, partition the table), `DELETE`-based retention (→ partition drop), per-scrape DB queries (cached 5 s). The model itself isn't the bottleneck (p95 ≈ 7 ms single-row).

**14. "What would you not trust in this project?"**
The thresholds (tuned on synthetic data), univariate-only drift, pickled model artifacts, and the absence of auth on MLflow. They're listed in `docs/DESIGN.md §6`; saying so up front is part of the answer.
