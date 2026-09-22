"""Unit tests for the prediction component and the release gate."""

import math

import pytest

from ml.features import FEATURE_ORDER
from tests.conftest import SAMPLE_REQUEST

# Release gate. Measured holdout MAE is ~5.71; this fails the build on a real regression
# while tolerating minor library-version drift. Recorded in README as a release criterion.
MAX_ACCEPTABLE_MAE = 7.0


def test_prediction_is_finite_and_plausible(bundle):
    prediction = bundle.predict(SAMPLE_REQUEST)
    assert math.isfinite(prediction)
    assert 5.0 < prediction < 130.0  # target range is 7.6-117.5


def test_prediction_is_close_to_known_target(bundle):
    """SAMPLE_REQUEST is row 1 of the dataset, actual price 37.9."""
    assert bundle.predict(SAMPLE_REQUEST) == pytest.approx(37.9, abs=15.0)


def test_prediction_is_deterministic(bundle):
    assert bundle.predict(SAMPLE_REQUEST) == bundle.predict(SAMPLE_REQUEST)


def test_further_from_mrt_predicts_lower_price(bundle):
    """Sanity check on the dominant feature rather than a hardcoded number."""
    near = bundle.predict({**SAMPLE_REQUEST, "mrt_distance": 100.0})
    far = bundle.predict({**SAMPLE_REQUEST, "mrt_distance": 5000.0})
    assert far < near


def test_preprocessor_is_actually_applied(bundle):
    from ml.features import to_frame

    frame = to_frame([SAMPLE_REQUEST])
    scaled = bundle.preprocessor.transform(frame)
    assert scaled.shape == (1, len(FEATURE_ORDER))
    assert not (scaled == frame.to_numpy()).all()


def test_unloaded_bundle_refuses_to_predict():
    from app.model import ModelBundle

    empty = ModelBundle()
    assert not empty.loaded
    with pytest.raises(RuntimeError, match="not loaded"):
        empty.predict(SAMPLE_REQUEST)


def test_load_rejects_feature_mismatch(bundle, tmp_path):
    """Artifacts trained on a different feature set must not be served."""
    from app.model import ModelBundle

    with pytest.raises(ValueError, match="feature mismatch"):
        ModelBundle._check_consistency(bundle.preprocessor, {"features": ["only_one"]})


def test_release_gate_holdout_mae(bundle):
    assert bundle.metadata["metrics"]["mae"] < MAX_ACCEPTABLE_MAE


def test_metadata_records_reproducibility_inputs(bundle):
    metadata = bundle.metadata
    assert metadata["features"] == FEATURE_ORDER
    assert metadata["seed"] == 42
    assert metadata["split_rule"].endswith("2013.5")
    assert metadata["n_train"] == 344
    assert metadata["n_test"] == 70
    for key in ("model_version", "model_type", "trained_at", "sklearn_version", "cv_results"):
        assert metadata[key], f"metadata missing {key}"
