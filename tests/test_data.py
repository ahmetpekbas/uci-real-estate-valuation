"""Data/schema validation against the provided dataset."""

import pandas as pd
import pytest

from ml.features import (
    DATE_RAW,
    EXPECTED_RAW_COLUMNS,
    TARGET_RAW,
    SchemaError,
    prepare,
    validate_raw,
)
from ml.train import DATA_PATH, HOLDOUT_MONTHS, split_cutoff, temporal_split


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def test_dataset_matches_expected_schema(raw):
    validate_raw(raw)
    assert list(raw.columns) == EXPECTED_RAW_COLUMNS
    assert len(raw) == 414
    assert raw.isna().sum().sum() == 0


def test_target_within_documented_range(raw):
    assert raw[TARGET_RAW].min() >= 7.6
    assert raw[TARGET_RAW].max() <= 117.5


def test_validate_raw_rejects_missing_column(raw):
    with pytest.raises(SchemaError, match="missing column"):
        validate_raw(raw.drop(columns=[TARGET_RAW]))


def test_validate_raw_rejects_nulls(raw):
    broken = raw.copy()
    broken.loc[0, TARGET_RAW] = None
    with pytest.raises(SchemaError, match="null values"):
        validate_raw(broken)


def test_temporal_split_is_deterministic_and_disjoint(raw):
    train, test, cutoff = temporal_split(prepare(raw))
    assert len(train) == 344
    assert len(test) == 70
    assert train[DATE_RAW].max() < cutoff <= test[DATE_RAW].min()


def test_split_policy_resolves_to_the_expected_cutoff(raw):
    """The policy must reproduce the documented split on the provided dataset."""
    assert HOLDOUT_MONTHS == 2
    assert split_cutoff(prepare(raw)) == 2013.5


def test_split_cutoff_rejects_policy_wider_than_history(raw):
    """Holding out more months than exist would leave nothing to train on."""
    df = prepare(raw)
    too_short = df[df[DATE_RAW] >= 2013.5]  # only 2 distinct dates remain
    with pytest.raises(ValueError, match="distinct transaction dates"):
        split_cutoff(too_short)
