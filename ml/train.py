"""Training entrypoint. Reproducible with a single command: `make train`.

Loads and validates the CSV, splits it temporally, selects between two candidate models by
cross-validated MAE, scores the winner once on the holdout, and writes three artifacts:
the model, the fitted preprocessor, and the metadata that GET /model/info serves.
"""

import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from ml.features import DATE_RAW, FEATURE_ORDER, TARGET_RAW, prepare, validate_raw
from ml.logging_setup import configure_logging, log_event

DATA_PATH = Path("data/Real estate.csv")
ARTIFACTS_DIR = Path("artifacts")

SEED = 42
MODEL_VERSION = "1.0.0"
# Policy: hold out the most recent N transaction months. A temporal cut rather than a random one,
# because 20 rows share a feature vector with a different target and a random split would leak
# them across the boundary. Stated as a rule rather than a fixed date so the split stays
# meaningful if the dataset is extended; the resolved cutoff is recorded in the metadata. A month
# count rather than a target test fraction because the months are lumpy (23-58 rows), so an
# intended percentage cannot be hit without the split lurching between 17% and 31%.
HOLDOUT_MONTHS = 2

CANDIDATES = {
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(n_estimators=300, random_state=SEED),
}

log = logging.getLogger("ml.train")


def git_sha() -> str | None:
    """Best-effort commit id.

    Prefers $GIT_SHA because the Docker build context excludes .git -- the build passes the
    commit in as a build arg so the baked metadata still names the commit that produced it.
    Falls back to asking git, then to None.
    """
    if env_sha := os.environ.get("GIT_SHA"):
        return env_sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def split_cutoff(df: pd.DataFrame) -> float:
    """Resolve the temporal cutoff: the start of the Nth-from-last transaction month."""
    dates = sorted(df[DATE_RAW].unique())
    if len(dates) <= HOLDOUT_MONTHS:
        raise ValueError(
            f"need more than {HOLDOUT_MONTHS} distinct transaction dates to hold out "
            f"{HOLDOUT_MONTHS}; found {len(dates)}"
        )
    return float(dates[-HOLDOUT_MONTHS])


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Split on the resolved cutoff, returning it so the metadata can record the actual value."""
    cutoff = split_cutoff(df)
    train = df[df[DATE_RAW] < cutoff]
    test = df[df[DATE_RAW] >= cutoff]
    if train.empty or test.empty:
        raise ValueError(f"cutoff {cutoff} produced an empty side")
    return train, test, cutoff


def select_model(x_train, y_train) -> tuple[str, float, dict[str, float]]:
    """Pick the candidate with the lowest cross-validated MAE on the training rows only.

    Every candidate gets its own log line, not just the winner: the baseline's score is the
    evidence that the selected model earns its complexity, and a run's log should carry that
    without anyone having to open metadata.json.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for name, model in CANDIDATES.items():
        mae = -cross_val_score(
            model, x_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
        ).mean()
        scores[name] = mae
        log_event(log, "candidate_scored", model=name, cv_mae=round(mae, 4))
    best = min(scores, key=scores.__getitem__)
    return best, scores[best], scores


def main() -> None:
    configure_logging()

    raw = pd.read_csv(DATA_PATH)
    validate_raw(raw)
    df = prepare(raw)

    train_df, test_df, cutoff = temporal_split(df)
    x_train, y_train = train_df[FEATURE_ORDER], train_df[TARGET_RAW]
    x_test, y_test = test_df[FEATURE_ORDER], test_df[TARGET_RAW]

    # The preprocessor is fitted on training rows only and saved as an artifact, so serving
    # reuses this exact object instead of re-implementing the transform.
    preprocessor = StandardScaler().fit(x_train)
    x_train_s = preprocessor.transform(x_train)
    x_test_s = preprocessor.transform(x_test)

    best_name, best_cv_mae, cv_results = select_model(x_train_s, y_train)
    log_event(
        log,
        "model_selected",
        model=best_name,
        cv_mae=round(best_cv_mae, 4),
        cv_mae_by_model={k: round(v, 4) for k, v in cv_results.items()},
    )

    model = CANDIDATES[best_name].fit(x_train_s, y_train)
    predictions = model.predict(x_test_s)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": best_name,
        "features": FEATURE_ORDER,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "cv_results": {k: round(v, 4) for k, v in cv_results.items()},
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "git_sha": git_sha(),
        "split_policy": f"last {HOLDOUT_MONTHS} transaction months",
        "split_rule": f"{DATE_RAW} >= {cutoff}",
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # The holdout is scored once, by the winner only (D4), so the final line reports the
    # winner's holdout metrics alongside the CV comparison that chose it.
    log_event(
        log,
        "training_complete",
        **metadata["metrics"],
        model=best_name,
        cv_mae_by_model=metadata["cv_results"],
    )


if __name__ == "__main__":
    main()
