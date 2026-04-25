"""
Anomaly detection module.

Loads a trained Isolation Forest model from S3 and predicts
anomaly labels on preprocessed metric data.
"""
from __future__ import annotations

import io
from typing import Optional

import boto3
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from utils import (
    DetectionResult,
    PipelineConfig,
    get_logger,
    get_s3_key,
)

logger = get_logger(__name__)


def load_latest_model(config: PipelineConfig) -> Optional[IsolationForest]:
    """
    Load the most recently modified IsolationForest model from S3.

    Lists all objects under the ``models/`` prefix in the configured S3 bucket
    and returns the model deserialized from the object with the most recent
    ``LastModified`` timestamp.

    Args:
        config: Immutable pipeline configuration.

    Returns:
        The fitted ``IsolationForest`` model, or ``None`` if no model objects
        exist under the ``models/`` prefix.
    """
    s3_client = boto3.client("s3", region_name=config.aws_region)

    response = s3_client.list_objects_v2(
        Bucket=config.s3_bucket,
        Prefix="models/",
    )

    contents = response.get("Contents")
    if not contents:
        logger.info("No model objects found under models/ prefix — will train a new model.")
        return None

    # Find the object with the most recent LastModified timestamp
    latest_obj = max(contents, key=lambda obj: obj["LastModified"])
    latest_key = latest_obj["Key"]

    logger.info(
        "Loading latest model from s3://%s/%s (LastModified: %s)",
        config.s3_bucket,
        latest_key,
        latest_obj["LastModified"],
    )

    obj_response = s3_client.get_object(Bucket=config.s3_bucket, Key=latest_key)
    body = obj_response["Body"].read()
    model: IsolationForest = joblib.load(io.BytesIO(body))

    return model


def detect_anomalies(
    processed_df: pd.DataFrame,
    model: IsolationForest,
    config: PipelineConfig,
) -> DetectionResult:
    """
    Apply the fitted model to *processed_df* and write results to S3.

    Feature columns are all columns in *processed_df* except ``timestamp``.
    Appends an ``anomaly_label`` column to a copy of the DataFrame where
    ``-1`` indicates an anomaly and ``1`` indicates a normal observation.

    Args:
        processed_df: Preprocessed metrics DataFrame (output of the preprocessing stage).
        model: A fitted ``IsolationForest`` instance.
        config: Immutable pipeline configuration.

    Returns:
        ``DetectionResult`` containing the S3 key, the results DataFrame
        (with ``anomaly_label`` column), and the count of anomalous rows.
    """
    logger.info("Anomaly detection started.")

    # Extract feature columns (all except 'timestamp')
    feature_cols = [col for col in processed_df.columns if col != "timestamp"]
    features = processed_df[feature_cols].values

    # Predict: -1 = anomaly, 1 = normal
    labels = model.predict(features)

    # Append anomaly_label to a copy of the DataFrame
    result_df = processed_df.copy()
    result_df["anomaly_label"] = labels

    # Serialize results to CSV bytes
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")

    # Build the S3 key and upload
    s3_key = get_s3_key(
        prefix="results",
        name="results",
        instance_id=config.instance_id,
        timestamp=config.run_timestamp,
        extension="csv",
    )

    s3_client = boto3.client("s3", region_name=config.aws_region)
    s3_client.put_object(Bucket=config.s3_bucket, Key=s3_key, Body=csv_bytes)

    anomaly_count = int((labels == -1).sum())

    logger.info(
        "Anomaly detection complete — %d anomalies found, results saved to s3://%s/%s",
        anomaly_count,
        config.s3_bucket,
        s3_key,
    )

    return DetectionResult(
        s3_key=s3_key,
        dataframe=result_df,
        anomaly_count=anomaly_count,
    )
