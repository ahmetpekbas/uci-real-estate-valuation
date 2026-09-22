.PHONY: install train test lint format serve docker-build docker-run clean

# The venv layout differs between Windows and Linux; CI runs Linux, dev here is Windows.
ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

IMAGE := house-price-api
TAG := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
GIT_SHA := $(shell git rev-parse HEAD 2>/dev/null)

install:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

train:
	$(PY) -m ml.train

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .

serve:
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build --build-arg GIT_SHA=$(GIT_SHA) -t $(IMAGE):$(TAG) -t $(IMAGE):latest .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE):$(TAG)

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
