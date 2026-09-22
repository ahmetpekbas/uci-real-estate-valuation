# Operations

Each metric is marked **live** (collectable from what the service emits today) or with the change it
needs. The service emits one JSON line per request — `request_id`, `method`, `path`, `status`,
`latency_ms` ([app/main.py:52](../app/main.py#L52)) — and ships three artifacts. Predictions and
input values are not in that line, which is what constrains the model rows below and §3.

## 1. Monitoring

**Service** — all live; every row derives from the existing log line or the orchestrator.

| Metric | Why it matters | Source |
|---|---|---|
| `/ready` state and flap count | Readiness is the only signal that separates "process up" from "model loaded". A pod oscillating ready/not-ready is depooling under load, which the load balancer hides from error rate entirely | Readiness probe |
| Error rate split 4xx vs 5xx | 4xx rising means a caller changed their payload; 5xx rising means we broke. Different owners, different responses — so they are never aggregated into one "error rate" | `status` in the structured log |
| Latency p99 | The number users feel. Inference here is microseconds, so a p99 rise almost always means the process is starved, not that the model is slow | `latency_ms` in the structured log |

**Model** — `/predict` returns the prediction to the caller but does not log it, so live model
behaviour is currently unobservable. Rows 2–4 are one `log_event` in the predict handler away.

| Metric | Why it matters | Status |
|---|---|---|
| Holdout MAE / RMSE per retrain | The release gate (§4). Tracked over retrains, not over requests | Live — `metadata.json` and the `training_complete` line |
| Rolling prediction mean vs the training target mean | Ground truth is a completed sale, so true error arrives weeks later; the prediction mean is the leading indicator available without labels | ~2 lines |
| Rolling prediction standard deviation | Catches what the mean cannot. If a serving bug makes features effectively constant the model regresses toward the training mean, and predictions can average a healthy 38.0 while every one *is* 38.0 | Same 2 lines |
| Prediction range violations (outside 7.6–117.5, or ≤ 0) | `validate_raw` rejects non-positive targets at training time, so a non-positive prediction is definitionally broken rather than merely unusual. An integer counter, no statistics | Same 2 lines |
| Input feature distributions vs the training reference | Detects the upstream change that causes a silent quality drop, before any label exists. Covers out-of-range inputs: `mrt_distance` spans 23–6488 in training and the schema bounds it below at 0 but not above, so a request at 40 000 is schema-valid and still an extrapolation the model answers confidently | See §3 |

**Why MAE is the headline and RMSE rides alongside:** MAE is in target units — "off by 5.7 per unit
area" is directly actionable. RMSE is reported because the *gap* is the signal: the target's thin
upper tail (to 117.5 against a mean of 38.0) means a handful of expensive properties dominate
squared error. RMSE ≫ MAE (8.22 vs 5.71) says the model is failing specifically on the tail, which
is a different problem from failing everywhere.

## 2. Alerting strategy

**High priority — notify a human.** All three work today with no new instrumentation, which is the
property worth having: the paging tier does not depend on work that has not been done.

| Condition | Rationale |
|---|---|
| `/ready` failing on all replicas for > 2 min | Total outage. This is the one that must page |
| 5xx rate > 1% over 5 min | We are returning errors; a caller is broken by us |
| Restart loop: > 3 restarts in 10 min on a replica | Usually a bad release; couples directly to the rollback in §4 |

**Warning — ticket, reviewed in hours.**

| Condition | Rationale | Status |
|---|---|---|
| Holdout MAE regresses > 15% on a retrain | Retraining made things worse; do not promote | Live — CI has the number |
| Total 422 rate > 10% over 15 min | Usually a caller deploying a contract change | Live — `status` |
| Prediction mean shifts > 20%, or stddev drops > 50%, from training | Model behaviour changed without a deploy | Needs §1's 2 lines |
| Feature mean drifts > 3 standard errors from training | Upstream data change | Needs §3 stage 2 |

422 rate is live in aggregate but not per field. The offending field names are in `exc.errors()`,
which [app/main.py:71](../app/main.py#L71) returns to the caller and never logs — so the breakdown
that identifies *which* caller broke *what* needs one `log_event` in that handler.

**Deliberately not alerted:** individual prediction values. There is no per-request notion of a
"wrong" prediction without ground truth, and alerting on individual outputs generates noise that
trains people to ignore the channel.

## 3. Lightweight drift detection

Ground truth lags by weeks, so drift detection works on *inputs and outputs*, not accuracy. Staged
cheapest-first; each stage is useful without the next.

**Stage 1 — log what we already compute (~4 lines).** The predict handler holds the validated input
and the prediction and logs neither. Emit both on the existing request line. Property attributes,
not personal data — retention is a storage question, not a privacy one. Nothing downstream is
possible until this exists.

**Stage 2 — compare against the scaler we already ship (~10 lines).** `preprocessor.joblib` is a
fitted `StandardScaler`, so it carries `mean_` and `var_` per feature: a training reference already
sitting in the image. A scheduled job reads the last N days of logs and computes, per feature,
`(batch_mean − mean_[i]) / sqrt(var_[i] / n)` — how many standard errors the mean has moved. No new
artifact, no binning. Its limit: mean shifts only, so a feature that goes bimodal or doubles its
variance around a stable mean passes clean.

**Stage 3 — PSI, once shape changes matter (~50 lines).** `PSI = Σ (actual% − expected%) ×
ln(actual% / expected%)` per feature, read the conventional way: under 0.10 stable, 0.10–0.25 ticket
and review at the next retrain, over 0.25 investigate and consider retraining. This one does need a
training change — decile boundaries written per feature, since `metadata.json` holds no distribution
statistics today. Worth it after a mean-only check proves insufficient, not before.

At each stage, run the same comparison on the *prediction* distribution. Input drift with stable
predictions is usually benign; prediction drift without input drift means something changed in the
serving path, which is a bug, not drift.

The staging is a deliberate trade: the time budget bought a working release path, so drift is
specified rather than built.

## 4. Rollback

**Code and model roll back together.** Artifacts are baked into the image, so the image holds both
the service code at a commit and the model trained from it. Rollback is one action:

```bash
kubectl rollout undo deployment/house-price-api      # or: redeploy the previous tag
docker run house-price-api:<previous-short-sha>      # local equivalent
```

Verify with `GET /model/info`, minding the asymmetry: `make docker-build` tags with the **short** SHA
while the metadata records the **full** one, so the check is that `git_sha` *starts with* the tag you
rolled back to. CI compares the full value against `github.sha`, which is why gate 5 can use equality
where a human at a terminal cannot.

**Rolling back the model alone requires a rebuild** — the cost of immutability. It does not require
retraining: CI uploads `model-artifacts-<sha>` on every run with 14-day retention, so a previous
model can be retrieved and rebuilt into an image. Past that window, `make train` at the commit
reproduces it, since the split is seeded and the rule recorded. In exchange for the friction, "which
model is in production right now?" is never ambiguous. This stops being the right trade the moment
models ship on a different cadence than code — at that point artifacts move to a registry, the image
takes a version pin, and model rollback becomes a config change ([ARCHITECTURE.md](ARCHITECTURE.md)).

### Release gates

No image is promoted unless all of these pass in CI. All five are implemented in
[ci.yml](../.github/workflows/ci.yml):

1. `ruff check` and `ruff format --check` clean.
2. `make train` completes from a clean checkout — reproducibility is asserted, not assumed.
3. Full test suite green, including the data/schema validation tests.
4. Holdout MAE < 7.0 (`tests/test_predict.py::test_release_gate_holdout_mae`, part of gate 3).
5. The built image starts, answers `/health`, `/ready`, `/model/info` and `/predict`, and its baked
   `git_sha` equals the commit being built.

## 5. Incident response runbook

**Triage — first 5 minutes**

1. `GET /health` — process alive? Failing → down or restart-looping; check orchestrator events and
   container logs.
2. `GET /ready` — model loaded? 200 on `/health` with 503 here → **artifacts failed to load**. This
   is the designed signal: the process stays up so the logs are readable. Grep `artifact_load_failed`
   ([app/model.py:39](../app/model.py#L39)); the line carries the exact error and directory.
3. `GET /model/info` — is the *expected* model running? A `git_sha` or `model_version` mismatch means
   a bad or partial deploy → rollback (§4).

**Diagnose — next 10 minutes**

4. Error class. 4xx spike → caller-side; the field detail sits in the 422 body the caller received,
   not in our logs (§2), so ask them for it. 5xx spike → ours: the `unhandled_error` line carries the
   traceback and `path`, the middleware line carries `request_id` and `status`; correlate the two by
   timestamp, as the traceback line has no `request_id`.
5. Latency with restarts → resource limits. Latency alone → check whether artifacts are being
   reloaded per request instead of once per process.
6. Did a deploy or an upstream data change precede the symptom? Correlate with `trained_at` and
   `git_sha`.

**Act**

7. Deploy-correlated → roll back first, diagnose after (§4). MTTR beats root cause in the moment.
8. Service up but predictions look wrong → the blind spot until §3 stage 1 ships. Today: replay known
   inputs against `/predict` and compare with the holdout predictions at that commit. Depool if
   predictions are consumed automatically.
9. Caller contract break → contact the caller; do not loosen validation to make the error go away.
   The 422 is the system working.

**Close out**

10. Record the timeline, the signal that caught it, and what would have caught it sooner. If no alert
    fired, adding one is part of closing the incident.
