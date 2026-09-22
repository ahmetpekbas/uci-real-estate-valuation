"""Shared fixtures.

Tests run against real artifacts, not mocks. If they are absent the training job runs once
for the session (it takes under a second on 414 rows), so `make test` works on a fresh
clone and the release gate in test_predict.py can never be silently skipped.
"""

from pathlib import Path

import pytest

from app.model import ARTIFACTS_DIR, ModelBundle
from ml import train

SAMPLE_REQUEST = {
    "house_age": 32.0,
    "mrt_distance": 84.87882,
    "convenience_stores": 10,
    "latitude": 24.98298,
    "longitude": 121.54024,
}


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    if not (ARTIFACTS_DIR / "metadata.json").exists():
        train.main()
    return ARTIFACTS_DIR


@pytest.fixture(scope="session")
def bundle(artifacts_dir: Path) -> ModelBundle:
    loaded = ModelBundle()
    assert loaded.load(artifacts_dir), "artifacts failed to load"
    return loaded
