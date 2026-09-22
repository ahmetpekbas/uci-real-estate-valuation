"""REST service for house price prediction.

/health is liveness (process up, never touches the model) and /ready is readiness (model
actually loaded). Keeping them separate matters: if /health checked the model, a failed
artifact load would make an orchestrator restart-loop the pod instead of leaving it up and
depooled, where the logs are readable.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.model import bundle
from app.schemas import PredictRequest, PredictResponse
from ml.logging_setup import configure_logging, log_event

log = logging.getLogger("app")

MODEL_UNAVAILABLE = "model_unavailable"


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    bundle.load()
    yield


app = FastAPI(
    title="House Price Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)


def _error(status: int, code: str, detail, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": code,
            "detail": detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.middleware("http")
async def observability(request: Request, call_next):
    """Tag every request and emit one structured line with its outcome and latency."""
    request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    started = time.perf_counter()
    response = await call_next(request)
    log_event(
        log,
        "request",
        request_id=request.state.request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    response.headers["x-request-id"] = request.state.request_id
    return response


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    return _error(422, "validation_error", exc.errors(), request)


@app.exception_handler(HTTPException)
async def on_http_error(request: Request, exc: HTTPException):
    code = MODEL_UNAVAILABLE if exc.status_code == 503 else "http_error"
    return _error(exc.status_code, code, exc.detail, request)


@app.exception_handler(Exception)
async def on_unexpected_error(request: Request, exc: Exception):
    log.exception("unhandled_error", extra={"fields": {"path": request.url.path}})
    return _error(500, "internal_error", "an unexpected error occurred", request)


def _require_model() -> None:
    if not bundle.loaded:
        raise HTTPException(status_code=503, detail="model artifacts are not loaded")


@app.get("/health")
async def health() -> dict:
    """Liveness: the process is running. Deliberately does not inspect the model."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict:
    """Readiness: artifacts are loaded and this instance can serve traffic."""
    _require_model()
    return {"status": "ready", "model_version": bundle.model_version}


@app.get("/model/info")
async def model_info() -> dict:
    """The metadata written at training time -- the link between model and service version."""
    _require_model()
    return bundle.metadata


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    _require_model()
    prediction = bundle.predict(payload.model_dump())
    return PredictResponse(
        prediction=round(prediction, 2),
        model_version=bundle.model_version,
        request_id=request.state.request_id,
    )
