"""Artifact loading and prediction.

Load failure is caught, never fatal: the process must stay up so /health keeps answering and
/ready can report 503. A crash at startup would make that distinction meaningless.
"""

import json
import logging
from pathlib import Path

import joblib

from ml.features import FEATURE_ORDER, to_frame

ARTIFACTS_DIR = Path("artifacts")

log = logging.getLogger("app.model")


class ModelBundle:
    def __init__(self) -> None:
        self.model = None
        self.preprocessor = None
        self.metadata: dict | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    @property
    def model_version(self) -> str:
        return self.metadata.get("model_version", "unknown") if self.metadata else "unknown"

    def load(self, artifacts_dir: Path = ARTIFACTS_DIR) -> bool:
        try:
            model = joblib.load(artifacts_dir / "model.joblib")
            preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")
            metadata = json.loads((artifacts_dir / "metadata.json").read_text())
            self._check_consistency(preprocessor, metadata)
        except Exception as exc:  # noqa: BLE001 - any failure means "not ready", not "crash"
            log.warning(
                "artifact_load_failed",
                extra={"fields": {"error": str(exc), "dir": str(artifacts_dir)}},
            )
            self.model = self.preprocessor = self.metadata = None
            return False

        self.model, self.preprocessor, self.metadata = model, preprocessor, metadata
        log.info(
            "artifacts_loaded",
            extra={"fields": {"model_version": metadata.get("model_version")}},
        )
        return True

    @staticmethod
    def _check_consistency(preprocessor, metadata: dict) -> None:
        """Refuse artifacts that disagree with the serving feature contract.

        This is the training-serving skew tripwire: a model trained on a different feature
        set must not be served by this code, and failing loudly at load time is far better
        than returning plausible-looking numbers computed from misaligned columns.
        """
        if metadata.get("features") != FEATURE_ORDER:
            raise ValueError(
                f"feature mismatch: metadata {metadata.get('features')} != code {FEATURE_ORDER}"
            )
        n_expected = getattr(preprocessor, "n_features_in_", len(FEATURE_ORDER))
        if n_expected != len(FEATURE_ORDER):
            raise ValueError(
                f"preprocessor expects {n_expected} features, contract has {len(FEATURE_ORDER)}"
            )

    def predict(self, record: dict) -> float:
        if not self.loaded:
            raise RuntimeError("model not loaded")
        frame = to_frame([record])
        return float(self.model.predict(self.preprocessor.transform(frame))[0])


bundle = ModelBundle()
