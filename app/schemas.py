"""Request/response contract.

Field names come from ml.features.FEATURE_ORDER; test_features.py asserts the two stay in
step, so a feature rename cannot reach production without the schema following it.
"""

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    # extra="forbid" rejects unknown fields instead of ignoring them -- a typo'd field name
    # would otherwise be silently dropped and the request served with a default-shaped frame.
    model_config = ConfigDict(extra="forbid")

    house_age: float = Field(ge=0, description="Age of the house in years")
    mrt_distance: float = Field(ge=0, description="Metres to nearest MRT station")
    convenience_stores: int = Field(ge=0, description="Convenience stores in walking range")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class PredictResponse(BaseModel):
    prediction: float = Field(description="Predicted house price of unit area")
    model_version: str
    request_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: object | None = None
    request_id: str | None = None
