"""Unit tests for preprocessing: the feature contract shared by training and serving."""

import pandas as pd
import pytest

from app.schemas import PredictRequest
from ml.features import (
    DATE_RAW,
    FEATURE_ORDER,
    RAW_TO_FIELD,
    TARGET_RAW,
    SchemaError,
    prepare,
    to_frame,
)

SAMPLE = {
    "house_age": 32.0,
    "mrt_distance": 84.87882,
    "convenience_stores": 10,
    "latitude": 24.98298,
    "longitude": 121.54024,
}


def test_api_schema_matches_feature_contract():
    """The skew tripwire: a renamed feature must break this, not production."""
    assert list(PredictRequest.model_fields) == FEATURE_ORDER


def test_to_frame_enforces_column_order():
    shuffled = {k: SAMPLE[k] for k in reversed(FEATURE_ORDER)}
    assert list(to_frame([shuffled]).columns) == FEATURE_ORDER


def test_to_frame_rejects_missing_feature():
    incomplete = {k: v for k, v in SAMPLE.items() if k != "latitude"}
    with pytest.raises(SchemaError, match="missing feature"):
        to_frame([incomplete])


def test_to_frame_rejects_empty_input():
    with pytest.raises(SchemaError):
        to_frame([])


def test_prepare_renames_features_and_keeps_split_key_and_target():
    raw_features = dict(zip(RAW_TO_FIELD, SAMPLE.values(), strict=True))
    raw = pd.DataFrame([{"No": 1, DATE_RAW: 2012.917, **raw_features, TARGET_RAW: 37.9}])
    prepared = prepare(raw)
    assert "No" not in prepared.columns
    assert set(FEATURE_ORDER) <= set(prepared.columns)
    assert DATE_RAW in prepared.columns
    assert TARGET_RAW in prepared.columns
    assert DATE_RAW not in FEATURE_ORDER  # the split key is not a feature (D3)
