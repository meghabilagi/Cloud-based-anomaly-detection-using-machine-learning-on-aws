"""
Model training module.

Trains an Isolation Forest model on preprocessed metrics and
serializes it to S3.
"""
from __future__ import annotations

import io

import boto3
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from utils import (
    ConfigurationError,
    PipelineConfig,
    TrainingResult,
    get_logger,
    get_s3_key,
)

logger = get_logger(__name__)


def _validate_hyperparameters(config: PipelineConfig) -> None:
    """
    Guard: validate that all IsolationForest hyperparameters are present and valid.

    Raises ConfigurationError if any hyperparameter is missing or out of range.
    """
    # contamination must be a float strictly between 0 and 0.5
    if not isinstance(config.contamination, float):
        raise ConfigurationError(
            f"'contamination' must be a float, got {type(config.contamination).__name__!r} "
            f"(value: {config.contamination!r})."
        )
    if not (0.0 < config.contamination < 0.5):
        raise ConfigurationError(
            f"'contamination' must be in the open interval (0, 0.5), "
            f"got {config.contamination!r}."
        )

    # n_estimators must be a positive integer
    if not isinstance(config.n_estimators, int):
        raise ConfigurationError(
            f"'n_estimators' must be an int, got {type(config.n_estimators).__name__!r} "
            f"(value: {config.n_estimators!r})."
        )
    if config.n_estimators <= 0:
        raise ConfigurationError(
            f"'n_estimators' must be a positive integer, got {config.n_estimators!r}."
        )

    # random_state must be an integer
    if not isinstance(config.random_state, int):
        raise ConfigurationError(
            f"'random_state' must be an int, got {type(config.random_state).__name__!r} "
            f"(value: {config.random_state!r})."
        )


def train_model(processed_df: pd.DataFrame, config: PipelineConfig) -> TrainingResult:
    """
    Train an IsolationForest on *processed_df* and persist it to S3.

    Feature columns are all columns in *processed_df* except ``timestamp``.

    Args:
        processed_df: Preprocessed metrics DataFrame (output of the preprocessing stage).
        config: Immutable pipeline configuration.

    Returns:
        TrainingResult containing the S3 key and the fitted IsolationForest model.

    Raises:
        ConfigurationError: If any required hyperparameter is missing or invalid.
    """
    logger.info(
        "Model training started — contamination=%.4f, n_estimators=%d, random_state=%d",
        config.contamination,
        config.n_estimators,
        config.random_state,
    )

    # Guard: validate hyperparameters before doing any work
    _validate_hyperparameters(config)

    # Determine feature columns (everything except 'timestamp')
    feature_cols = [col for col in processed_df.columns if col != "timestamp"]
    X = processed_df[feature_cols].values

    # Instantiate and fit the model
    model = IsolationForest(
        contamination=config.contamination,
        n_estimators=config.n_estimators,
        random_state=config.random_state,
    )
    model.fit(X)

    # Serialize model to an in-memory buffer
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_bytes = buffer.getvalue()

    # Build the S3 key and upload
    s3_key = get_s3_key(
        prefix="models",
        name="isolation_forest",
        instance_id=config.instance_id,
        timestamp=config.run_timestamp,
        extension="joblib",
    )

    s3_client = boto3.client("s3", region_name=config.aws_region)
    s3_client.put_object(Bucket=config.s3_bucket, Key=s3_key, Body=model_bytes)

    logger.info(
        "Model training complete — model saved to s3://%s/%s",
        config.s3_bucket,
        s3_key,
    )

    return TrainingResult(s3_key=s3_key, model=model)
