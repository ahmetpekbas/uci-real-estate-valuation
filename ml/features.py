"""Single source of truth for the feature contract.

The raw CSV columns are not valid Python identifiers ("X2 house age"), so something has to
rename them. If training and serving each did it independently they would drift apart --
classic training-serving skew, and here it would fail silently: swap latitude and longitude
and you still get a plausible-looking price back. So the mapping and the column order live
here, and both ml/train.py and app/ import them.
"""

import pandas as pd

TARGET_RAW = "Y house price of unit area"
DATE_RAW = "X1 transaction date"
INDEX_RAW = "No"

# Raw CSV column -> API field name. Order defines the model's feature order.
RAW_TO_FIELD = {
    "X2 house age": "house_age",
    "X3 distance to the nearest MRT station": "mrt_distance",
    "X4 number of convenience stores": "convenience_stores",
    "X5 latitude": "latitude",
    "X6 longitude": "longitude",
}

FEATURE_ORDER = list(RAW_TO_FIELD.values())

# X1 is deliberately absent from FEATURE_ORDER: it is the split key, not a feature.
EXPECTED_RAW_COLUMNS = [INDEX_RAW, DATE_RAW, *RAW_TO_FIELD, TARGET_RAW]


class SchemaError(ValueError):
    """Raised when input data does not match the expected schema."""


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Build a model-ready frame from API-style records, in FEATURE_ORDER.

    Used at serving time. Raises SchemaError rather than letting pandas silently
    produce NaN columns for missing fields.
    """
    if not records:
        raise SchemaError("no records supplied")
    missing = set(FEATURE_ORDER) - set(records[0])
    if missing:
        raise SchemaError(f"missing feature(s): {sorted(missing)}")
    return pd.DataFrame(records)[FEATURE_ORDER]


def validate_raw(df: pd.DataFrame) -> None:
    """Validate the training CSV before anything downstream touches it."""
    missing = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing column(s): {missing}")

    non_numeric = [c for c in EXPECTED_RAW_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise SchemaError(f"non-numeric column(s): {non_numeric}")

    null_counts = df[EXPECTED_RAW_COLUMNS].isna().sum()
    if null_counts.any():
        raise SchemaError(f"null values in: {null_counts[null_counts > 0].to_dict()}")

    if df.empty:
        raise SchemaError("dataset is empty")

    if (df[TARGET_RAW] <= 0).any():
        raise SchemaError("target contains non-positive values")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the row index, rename features, keep the split key and target."""
    return df.drop(columns=[INDEX_RAW]).rename(columns=RAW_TO_FIELD)
