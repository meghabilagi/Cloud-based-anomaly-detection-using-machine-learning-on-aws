"""
Preprocessing module.

Cleans, imputes, and normalizes raw metric data before model
training and inference.
"""
from __future__ import annotations

import boto3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import (
    InsufficientDataError,
    PipelineConfig,
    PreprocessResult,
    get_logger,
    get_s3_key,
)

logger = get_logger(__name__)


def preprocess(raw_df: pd.DataFrame, config: PipelineConfig) -> PreprocessResult:
    """
    Imputes missing values with column means, applies min-max scaling to metric
    columns, writes the processed CSV to S3, and returns a PreprocessResult.

    Args:
        raw_df: DataFrame containing a ``timestamp`` column and one or more
                metric columns (all columns except ``timestamp``).
        config: Immutable pipeline configuration.

    Returns:
        PreprocessResult with the S3 key and the processed DataFrame.

    Raises:
        InsufficientDataError: If fewer than 2 rows remain after imputation.
    """
    logger.info(
        "Preprocessing started: input shape=%s, instance_id=%s",
        raw_df.shape,
        config.instance_id,
    )

    # Identify metric columns (everything except 'timestamp')
    metric_cols = [col for col in raw_df.columns if col != "timestamp"]

    # Work on a copy so we don't mutate the caller's DataFrame
    df = raw_df.copy()

    # Impute NaN values in metric columns using column means.
    # If a column is entirely NaN, its mean is NaN too, so fall back to 0.
    col_means = df[metric_cols].mean()
    df[metric_cols] = df[metric_cols].fillna(col_means).fillna(0.0)

    # Raise if fewer than 2 rows remain after imputation
    if len(df) < 2:
        raise InsufficientDataError(
            f"Insufficient data for preprocessing: only {len(df)} row(s) remain "
            "after imputation. At least 2 rows are required."
        )

    # Apply min-max scaling to metric columns only (fit on current batch; not persisted)
    scaler = MinMaxScaler()
    df[metric_cols] = scaler.fit_transform(df[metric_cols])

    # Preserve the timestamp column unchanged (already untouched above)

    # Generate S3 key and write processed CSV
    s3_key = get_s3_key(
        prefix="processed",
        name="processed",
        instance_id=config.instance_id,
        timestamp=config.run_timestamp,
        extension="csv",
    )

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    s3_client = boto3.client("s3", region_name=config.aws_region)
    s3_client.put_object(Bucket=config.s3_bucket, Key=s3_key, Body=csv_bytes)

    logger.info(
        "Preprocessing complete: output shape=%s, s3_key=%s",
        df.shape,
        s3_key,
    )

    return PreprocessResult(s3_key=s3_key, dataframe=df)
