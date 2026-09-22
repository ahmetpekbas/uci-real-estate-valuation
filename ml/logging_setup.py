"""Structured JSON logging, shared by the training job and the service.

Both sides emit the same line shape so a single log pipeline can parse them. It lives in
ml/ because the dependency direction is app -> ml (app already imports ml.features); the
reverse would create a cycle.
"""

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # uvicorn installs its own plain-text handlers. Route them through ours so every line on
    # stdout is parseable JSON, and drop uvicorn.access outright -- the request middleware
    # already logs each request with more detail (request_id, latency_ms).
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True


def log_event(logger: logging.Logger, message: str, **fields) -> None:
    """Emit a structured line: log_event(log, "request", path="/predict", status=200)."""
    logger.info(message, extra={"fields": fields})
