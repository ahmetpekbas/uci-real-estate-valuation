"""API contract tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import bundle
from tests.conftest import SAMPLE_REQUEST


@pytest.fixture
def client(artifacts_dir):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(artifacts_dir):
    """A live process whose artifacts failed to load -- the case /ready exists for."""
    with TestClient(app) as test_client:
        saved = (bundle.model, bundle.preprocessor, bundle.metadata)
        bundle.model = bundle.preprocessor = bundle.metadata = None
        yield test_client
        bundle.model, bundle.preprocessor, bundle.metadata = saved


def test_health_is_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_loaded_model(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["model_version"]


def test_model_info_returns_metadata(client):
    body = client.get("/model/info").json()
    assert body["model_version"]
    assert body["features"]
    assert body["metrics"]["mae"] > 0
    assert "git_sha" in body
    assert body["split_rule"]


def test_predict_returns_prediction_and_version(client):
    response = client.post("/predict", json=SAMPLE_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert 5.0 < body["prediction"] < 130.0
    assert body["model_version"] == client.get("/model/info").json()["model_version"]
    assert body["request_id"] == response.headers["x-request-id"]


def test_predict_honours_supplied_request_id(client):
    response = client.post("/predict", json=SAMPLE_REQUEST, headers={"x-request-id": "trace-abc"})
    assert response.json()["request_id"] == "trace-abc"


@pytest.mark.parametrize(
    "payload,reason",
    [
        ({k: v for k, v in SAMPLE_REQUEST.items() if k != "latitude"}, "missing field"),
        ({**SAMPLE_REQUEST, "house_age": -1}, "below minimum"),
        ({**SAMPLE_REQUEST, "latitude": 91}, "above maximum"),
        ({**SAMPLE_REQUEST, "mrt_distance": "near"}, "wrong type"),
        ({**SAMPLE_REQUEST, "unexpected": 1}, "unknown field"),
        ({}, "empty body"),
    ],
)
def test_predict_rejects_invalid_payloads(client, payload, reason):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, reason
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["detail"]
    assert body["request_id"]


def test_health_stays_up_when_model_is_missing(client_without_model):
    """Liveness must not depend on the model, or the pod restart-loops."""
    assert client_without_model.get("/health").status_code == 200


def test_ready_returns_503_when_model_is_missing(client_without_model):
    response = client_without_model.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_predict_returns_503_when_model_is_missing(client_without_model):
    response = client_without_model.post("/predict", json=SAMPLE_REQUEST)
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


def test_model_info_returns_503_when_model_is_missing(client_without_model):
    assert client_without_model.get("/model/info").status_code == 503
