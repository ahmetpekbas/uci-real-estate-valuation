# House Price Prediction Service

A reproducible training pipeline and a REST inference service for the Taipei real-estate dataset,
with CI, container packaging, and operational documentation.

## Setup

```bash
make install
```

Requires Python 3.14 (any 3.12+ works), and Docker only for the container targets.
Creates .venv and installed pinned dependencies.

## Train

```bash
make train
```

Writes three artifacts to `artifacts/`:

| File | Contents |
|---|---|
| `model.joblib` | Fitted estimator |
| `preprocessor.joblib` | Fitted `StandardScaler`, reused verbatim at serving time |
| `metadata.json` | Version, features, metrics, CV results, timestamp, git SHA, split policy + resolved rule, seed |

Re-running produces **byte-identical** model and preprocessor artifacts. The split is a policy
(`HOLDOUT_MONTHS = 2` — hold out the most recent two transaction months) resolved against the data
rather than a seeded shuffle; both the policy and the resolved cutoff
(`X1 transaction date >= 2013.5`) are written to the metadata, and every remaining source of
randomness is seeded and recorded.

## Test

```bash
make test           # 36 tests
make lint           # ruff check + format check
```

Tests run against real artifacts, not mocks; if `artifacts/` is missing, the session fixture trains
once. Coverage maps onto the required strategy:

| File | Covers |
|---|---|
| `tests/test_features.py` | Preprocessing units — feature contract, column order, schema/contract agreement |
| `tests/test_predict.py` | Prediction units, artifact consistency, and the release gate (MAE < 7.0) |
| `tests/test_api.py` | API contract — all four endpoints, validation failures, unavailable-model behaviour |
| `tests/test_data.py` | Data/schema validation on the provided CSV |

## Run locally

```bash
make serve
```

```bash
curl localhost:8000/health
curl localhost:8000/ready
curl localhost:8000/model/info

curl -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"house_age":32,"mrt_distance":84.87882,"convenience_stores":10,
       "latitude":24.98298,"longitude":121.54024}'
```

Response example:

```json
{"prediction": 41.06, "model_version": "1.0.0", "request_id": "a1b2c3..."}
```

### Run in Docker

```bash
make docker-build   # trains in stage 1, bakes artifacts into the image, tags with the git SHA
make docker-run
```

## API

| Endpoint | Purpose | Codes |
|---|---|---|
| `POST /predict` | Single prediction | 200, 422 validation, 503 model unavailable |
| `GET /health` | **Liveness** — process is up; never inspects the model | 200 |
| `GET /ready` | **Readiness** — artifacts loaded, instance can serve | 200, 503 |
| `GET /model/info` | Training metadata: version, features, metrics, git SHA, split rule | 200, 503 |

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Split | Temporal: hold out the last 2 transaction months (resolves to `X1 >= 2013.5`) | 20 rows repeat a feature vector with a different target; a random split leaks them across the boundary. The temporal cut leaves 1 such vector straddling, and matches how the model is used. The cutoff is derived from a stated policy rather than hardcoded, and both policy and resolved value are recorded in metadata |
| `X1 transaction date` | Not a feature | Every future serving date is outside the training range; trees cannot extrapolate, so it would silently decay into a constant |
| Model | Ridge vs RandomForest, selected on 5-fold CV MAE | Ridge is the baseline to beat, RandomForest is what ships: CV MAE 4.80 vs 5.90. Both are recorded in metadata, so the baseline is on the record rather than asserted |
| Preprocessing | Fitted `StandardScaler` shipped as an artifact | Serving reuses the exact fitted object — the primary skew defence |
| Feature contract | `ml/features.py` imported by both `ml/` and `app/` | Raw columns are not Python identifiers; the mapping exists once, and a test asserts the API schema matches it |
| Packaging | Multi-stage image, artifacts baked in | Image tag = git SHA = model version: one immutable unit, rollback is redeploying the previous tag |
| Scope | No model registry, no bonus items | A single model does not need a registry; `metadata.json` plus an immutable tag answers "which model is running" |

## Release strategy

**Versioning.** `metadata.json` carries `model_version` and the `git_sha` of the commit that trained
the model. The container is tagged with that same SHA and has the artifacts baked in, so the service
and the model it serves are one immutable, identifiable unit. `GET /model/info` lets any running
instance prove which one it is.

**Minimum gates before release** — all enforced in CI:

1. `ruff check` and `ruff format --check` clean.
2. `make train` completes from a clean checkout (reproducibility asserted, not assumed).
3. Full test suite green, including data/schema validation.
4. Holdout MAE < 7.0 — an executable gate, not a convention.
5. The built image starts and answers `/health`, `/ready`, `/model/info` and `/predict`, and the
   baked `git_sha` matches the commit being built.
