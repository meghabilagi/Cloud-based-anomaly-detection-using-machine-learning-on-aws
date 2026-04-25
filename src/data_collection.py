"""
Data collection module.

Fetches EC2 performance metrics from Amazon CloudWatch and writes
raw CSV artifacts to S3.
"""
from __future__ import annotations

from datetime import timedelta
from typing import List

import boto3
import pandas as pd

from utils import (
    CollectionResult,
    PipelineConfig,
    get_logger,
    get_s3_key,
    retry_with_backoff,
)

logger = get_logger(__name__)

# The five EC2 metrics collected on every pipeline run.
_METRICS: List[str] = [
    "CPUUtilization",
    "NetworkIn",
    "NetworkOut",
    "DiskReadBytes",
    "DiskWriteBytes",
]


def _get_metric(
    cw_client,
    instance_id: str,
    metric_name: str,
    start_time,
    end_time,
) -> pd.Series:
    """
    Fetches a single CloudWatch metric for the given EC2 instance.

    Wraps the ``GetMetricStatistics`` API call in ``retry_with_backoff``.

    Returns a ``pd.Series`` indexed by UTC timestamp with metric values as
    values.  Returns an empty ``pd.Series`` when CloudWatch returns no data
    points for the requested window.
    """
    def _call():
        return cw_client.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName=metric_name,
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Statistics=["Average"],
        )

    response = retry_with_backoff(_call)
    datapoints = response.get("Datapoints", [])

    if not datapoints:
        return pd.Series(dtype=float)

    timestamps = [dp["Timestamp"] for dp in datapoints]
    values = [dp["Average"] for dp in datapoints]
    series = pd.Series(data=values, index=timestamps, name=metric_name)
    series.sort_index(inplace=True)
    return series


def collect_metrics(config: PipelineConfig) -> CollectionResult:
    """
    Collects all five EC2 metrics from CloudWatch for the last 60 minutes.

    The query window is ``[run_timestamp − 60 minutes, run_timestamp]``.
    Results are assembled into a DataFrame with columns:
    ``timestamp``, ``CPUUtilization``, ``NetworkIn``, ``NetworkOut``,
    ``DiskReadBytes``, ``DiskWriteBytes``.

    Writes the raw CSV to S3 under the ``raw/`` prefix and returns a
    ``CollectionResult`` containing the S3 key and the DataFrame.
    """
    logger.info(
        "Starting metric collection for instance %s at %s",
        config.instance_id,
        config.run_timestamp.isoformat(),
    )

    end_time = config.run_timestamp
    start_time = end_time - timedelta(minutes=60)

    cw_client = boto3.client("cloudwatch", region_name=config.aws_region)
    s3_client = boto3.client("s3", region_name=config.aws_region)

    # Fetch each metric as a Series indexed by timestamp.
    series_map = {}
    for metric_name in _METRICS:
        series_map[metric_name] = _get_metric(
            cw_client,
            config.instance_id,
            metric_name,
            start_time,
            end_time,
        )

    # Assemble all series into a single DataFrame by joining on the timestamp
    # index.  Metrics with no data produce NaN columns.
    df = pd.DataFrame(series_map)
    df.index.name = "timestamp"
    df.reset_index(inplace=True)

    # Ensure column order matches the schema.
    columns = ["timestamp"] + _METRICS
    df = df.reindex(columns=columns)

    # Serialise to CSV bytes and upload to S3.
    key = get_s3_key(
        prefix="raw",
        name="metrics",
        instance_id=config.instance_id,
        timestamp=config.run_timestamp,
        extension="csv",
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    s3_client.put_object(Bucket=config.s3_bucket, Key=key, Body=csv_bytes)

    logger.info(
        "Metric collection complete. Wrote %d rows to s3://%s/%s",
        len(df),
        config.s3_bucket,
        key,
    )

    return CollectionResult(s3_key=key, dataframe=df)
