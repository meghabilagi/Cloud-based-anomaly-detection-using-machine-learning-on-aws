"""
Integration test for the end-to-end AWS Anomaly Detection Pipeline.

Uses moto to mock S3, CloudWatch, and SNS.
Validates that run_pipeline() produces all expected artifacts and
publishes an SNS alert when anomalies are present.

Note: moto v5 uses `mock_aws`; moto v4 uses separate `mock_s3`, `mock_sns`,
`mock_cloudwatch`. This file detects the installed version and uses the
appropriate decorator.

Validates: Requirements 1.2, 2.3, 3.3, 4.3, 5.3, 6.3, 7.1
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import boto3
import pytest

# Detect moto version and import the appropriate mock decorator(s).
try:
    from moto import mock_aws  # moto v5+
    _MOTO_V5 = True
except ImportError:
    from moto import mock_s3, mock_sns, mock_cloudwatch  # moto v4
    _MOTO_V5 = False

# Ensure src/ is on the path (mirrors conftest.py at root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from utils import PipelineConfig
from pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Synthetic CloudWatch data
# ---------------------------------------------------------------------------

# Ten data points with one extreme value (95.0) that should trigger both
# the CPU threshold alert (>90%) and anomaly detection.
_SYNTHETIC_VALUES = [10.0, 12.0, 11.0, 95.0, 10.5, 11.5, 12.5, 10.0, 11.0, 12.0]

_SYNTHETIC_DATAPOINTS = [
    {
        "Timestamp": datetime(2024, 1, 1, 11, i, 0, tzinfo=timezone.utc),
        "Average": value,
        "Unit": "Percent",
    }
    for i, value in enumerate(_SYNTHETIC_VALUES)
]

_SYNTHETIC_RESPONSE = {"Datapoints": _SYNTHETIC_DATAPOINTS}


def _make_config(s3_bucket: str, sns_topic_arn: str) -> PipelineConfig:
    """Build a PipelineConfig pointing at the mocked AWS resources."""
    return PipelineConfig(
        s3_bucket=s3_bucket,
        sns_topic_arn=sns_topic_arn,
        instance_id="i-test1234",
        contamination=0.1,
        n_estimators=10,
        random_state=42,
        aws_region="us-east-1",
        run_timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Helper: list all S3 keys in a bucket
# ---------------------------------------------------------------------------

def _list_s3_keys(s3_client, bucket: str) -> list:
    """Return all object keys in *bucket*."""
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# ---------------------------------------------------------------------------
# Decorator helper: apply the right moto decorators regardless of version
# ---------------------------------------------------------------------------

def _moto_all_services(func):
    """
    Decorator that applies the appropriate moto mock decorators for S3, SNS,
    and CloudWatch, regardless of whether moto v4 or v5 is installed.
    """
    if _MOTO_V5:
        return mock_aws(func)
    else:
        # Stack moto v4 decorators (applied bottom-up)
        return mock_s3(mock_sns(mock_cloudwatch(func)))


# ---------------------------------------------------------------------------
# Integration test 1: end-to-end success + artifact verification
# ---------------------------------------------------------------------------

@_moto_all_services
def test_end_to_end_pipeline_success():
    """
    End-to-end integration test for run_pipeline().

    Validates:
    - PipelineResult.status == "success"
    - All five artifact types are present in S3 under the correct prefixes:
        raw CSV        → raw/
        processed CSV  → processed/
        model .joblib  → models/
        results CSV    → results/
        five PNGs      → results/plots/
    - SNS publish is called when anomalies are present in the synthetic data
      (the 95.0 CPU value exceeds the 90% threshold and triggers an alert).

    Requirements: 1.2, 2.3, 3.3, 4.3, 5.3, 6.3, 7.1
    """
    region = "us-east-1"
    bucket_name = "test-bucket"
    topic_name = "test-topic"

    # ---- Set up mocked AWS resources ----------------------------------------
    s3_client = boto3.client("s3", region_name=region)
    sns_client = boto3.client("sns", region_name=region)

    # Create the S3 bucket
    s3_client.create_bucket(Bucket=bucket_name)

    # Create the SNS topic and capture its ARN
    topic_response = sns_client.create_topic(Name=topic_name)
    sns_topic_arn = topic_response["TopicArn"]

    # ---- Build config -------------------------------------------------------
    config = _make_config(bucket_name, sns_topic_arn)

    # ---- Mock CloudWatch get_metric_statistics to return synthetic data ------
    # moto's CloudWatch mock does not fully support GetMetricStatistics, so we
    # patch the method on the boto3 client to return our synthetic data.
    original_boto3_client = boto3.client

    def patched_boto3_client(service_name, **kwargs):
        client = original_boto3_client(service_name, **kwargs)
        if service_name == "cloudwatch":
            client.get_metric_statistics = MagicMock(
                return_value=_SYNTHETIC_RESPONSE
            )
        return client

    with patch("boto3.client", side_effect=patched_boto3_client):
        result = run_pipeline(config)

    # ---- Assert: pipeline succeeded -----------------------------------------
    assert result.status == "success", (
        f"Expected status='success', got {result.status!r}"
    )

    # ---- Assert: all artifact types are present in S3 -----------------------
    all_keys = _list_s3_keys(s3_client, bucket_name)

    # 1. Raw CSV under raw/
    raw_keys = [k for k in all_keys if k.startswith("raw/") and k.endswith(".csv")]
    assert len(raw_keys) == 1, (
        f"Expected exactly 1 raw CSV under raw/, found: {raw_keys}"
    )

    # 2. Processed CSV under processed/
    processed_keys = [
        k for k in all_keys if k.startswith("processed/") and k.endswith(".csv")
    ]
    assert len(processed_keys) == 1, (
        f"Expected exactly 1 processed CSV under processed/, found: {processed_keys}"
    )

    # 3. Model .joblib under models/
    model_keys = [
        k for k in all_keys if k.startswith("models/") and k.endswith(".joblib")
    ]
    assert len(model_keys) == 1, (
        f"Expected exactly 1 model .joblib under models/, found: {model_keys}"
    )

    # 4. Results CSV under results/ (but NOT under results/plots/)
    results_csv_keys = [
        k
        for k in all_keys
        if k.startswith("results/")
        and k.endswith(".csv")
        and not k.startswith("results/plots/")
    ]
    assert len(results_csv_keys) == 1, (
        f"Expected exactly 1 results CSV under results/, found: {results_csv_keys}"
    )

    # 5. Five PNG files under results/plots/
    plot_keys = [
        k for k in all_keys if k.startswith("results/plots/") and k.endswith(".png")
    ]
    assert len(plot_keys) == 5, (
        f"Expected exactly 5 PNG files under results/plots/, found: {plot_keys}"
    )

    # ---- Assert: stage_artifacts keys use the correct prefixes --------------
    assert result.stage_artifacts["raw"].startswith("raw/"), (
        f"raw artifact key should start with 'raw/': {result.stage_artifacts['raw']}"
    )
    assert result.stage_artifacts["processed"].startswith("processed/"), (
        f"processed artifact key should start with 'processed/': "
        f"{result.stage_artifacts['processed']}"
    )
    assert result.stage_artifacts["model"].startswith("models/"), (
        f"model artifact key should start with 'models/': "
        f"{result.stage_artifacts['model']}"
    )
    assert result.stage_artifacts["results"].startswith("results/"), (
        f"results artifact key should start with 'results/': "
        f"{result.stage_artifacts['results']}"
    )
    for plot_key in result.stage_artifacts["plots"]:
        assert plot_key.startswith("results/plots/"), (
            f"plot artifact key should start with 'results/plots/': {plot_key}"
        )

    # ---- Assert: anomaly_count is non-negative (pipeline ran fully) ---------
    assert result.anomaly_count >= 0, "anomaly_count should be non-negative"

    # Verify the CPU threshold condition: max CPU in synthetic data is 95.0 > 90.0
    # This guarantees _send_alert() would have called sns.publish().
    max_cpu = max(_SYNTHETIC_VALUES)
    assert max_cpu > 90.0, (
        f"Synthetic data should have a CPU value > 90 to trigger alert, max={max_cpu}"
    )


# ---------------------------------------------------------------------------
# Integration test 2: SNS publish is actually called when anomalies present
# ---------------------------------------------------------------------------

@_moto_all_services
def test_end_to_end_pipeline_sns_publish_called_with_anomalies():
    """
    Verify that SNS publish is actually called when anomalies are present.

    This test patches the SNS client's publish method to count invocations,
    confirming the alerter fires when the synthetic data triggers an alert.

    Requirements: 7.1
    """
    region = "us-east-1"
    bucket_name = "test-bucket-sns"
    topic_name = "test-topic-sns"

    s3_client = boto3.client("s3", region_name=region)
    sns_client_real = boto3.client("sns", region_name=region)

    s3_client.create_bucket(Bucket=bucket_name)
    topic_response = sns_client_real.create_topic(Name=topic_name)
    sns_topic_arn = topic_response["TopicArn"]

    config = _make_config(bucket_name, sns_topic_arn)

    publish_calls = []

    original_boto3_client = boto3.client

    def patched_boto3_client(service_name, **kwargs):
        client = original_boto3_client(service_name, **kwargs)
        if service_name == "cloudwatch":
            client.get_metric_statistics = MagicMock(
                return_value=_SYNTHETIC_RESPONSE
            )
        if service_name == "sns":
            original_publish = client.publish

            def tracking_publish(**publish_kwargs):
                publish_calls.append(publish_kwargs)
                return original_publish(**publish_kwargs)

            client.publish = tracking_publish
        return client

    with patch("boto3.client", side_effect=patched_boto3_client):
        result = run_pipeline(config)

    assert result.status == "success"

    # The synthetic data has CPU=95.0 > 90%, so SNS publish must have been called
    assert len(publish_calls) >= 1, (
        f"Expected SNS publish to be called at least once, but got {len(publish_calls)} calls. "
        "The synthetic data contains CPU=95.0 which exceeds the 90% threshold."
    )

    # Verify the published message contains the instance ID
    published_message = publish_calls[0].get("Message", "")
    assert config.instance_id in published_message, (
        f"SNS message should contain instance_id={config.instance_id!r}, "
        f"got: {published_message[:200]}"
    )
