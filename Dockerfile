# Multi-stage: stage 1 trains, stage 2 serves. The artifacts are baked in rather than
# mounted at runtime, so the image tagged <git-sha> carries both the service code and the
# model metadata produced by that commit -- one version to reason about, and rollback is
# "redeploy the previous tag".

FROM python:3.14-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The build context excludes .git, so the commit id arrives as a build arg.
ARG GIT_SHA=""
ENV GIT_SHA=${GIT_SHA}

COPY ml/ ml/
COPY data/ data/
RUN python -m ml.train


FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 1000 appuser

COPY app/ app/
COPY ml/ ml/
COPY --from=builder /build/artifacts/ artifacts/

USER appuser
EXPOSE 8000

# Liveness, not readiness: the orchestrator should restart on /health failing and depool on
# /ready failing. Checking /ready here would restart-loop a pod with unloadable artifacts.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
