"""
Unit tests for data_collection.py.

Requirements: 2.1, 2.3, 2.4, 2.6
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.data_collection import collect_metrics
from src.utils import PipelineConfig

# CollectionResult is imported from the bare 'utils' module (as data_collection.py
# does) so that isinstance checks work against the same class object.
import utils as _utils
CollectionResult = _utils.CollectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> PipelineConfig:
    """Return a PipelineConfig populated with safe test defaults."""
    defaults = dict(
        s3_bucket="test-bucket",
        sns_topic_arn="arn:aws:sns:us-east-1:123456789:test-topic",
        instance_id="i-1234567890abcdef0",
        contamination=0.1,
        n_estimators=100,
        random_state=42,
        aws_region="us-east-1",
        run_timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_datapoint(ts: datetime, value: float) -> dict:
    """Build a CloudWatch Datapoints entry."""
    return {"Timestamp": ts, "Average": value, "Unit": "Percent"}


def _full_cw_response(value: float = 50.0) -> dict:
    """CloudWatch response with a single datapoint."""
    return {
        "Datapoints": [
            _make_datapoint(datetime(2024, 1, 1, 11, 0, 0), value)
        ]
    }


def _empty_cw_response() -> dict:
    """CloudWatch response with no datapoints."""
    return {"Datapoints": []}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_boto3_client():
    """
    Patches boto3.client so that no real AWS calls are made.

    Returns a tuple (mock_cw_client, mock_s3_client) for per-test configuration.
    """
    mock_cw = MagicMock()
    mock_s3 = MagicMock()

    def _client_factory(service, **kwargs):
        if service == "cloudwatch":
            return mock_cw
        if service == "s3":
            return mock_s3
        raise ValueError(f"Unexpected boto3 service: {service}")

    with patch("data_collection.boto3.client", side_effect=_client_factory):
        yield mock_cw, mock_s3


# ---------------------------------------------------------------------------
# Test: NaN columns when CloudWatch returns empty data
# ---------------------------------------------------------------------------

class TestEmptyDatapointsProduceNaNColumns:
    """
    Requirement 2.4 — when CloudWatch returns no Datapoints for a metric,
    the corresponding column in the returned DataFrame must be all-NaN.
    """

    def test_all_metrics_empty_produces_all_nan_columns(self, mock_boto3_client):
        """All five metric columns are NaN when every CloudWatch response is empty."""
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        config = _make_config()
        result = collect_metrics(config)

        metric_cols = [
            "CPUUtilization",
            "NetworkIn",
            "NetworkOut",
            "DiskReadBytes",
            "DiskWriteBytes",
        ]
        for col in metric_cols:
            assert col in result.dataframe.columns, f"Column '{col}' missing from DataFrame"
            assert result.dataframe[col].isna().all(), (
                f"Expected all-NaN column for '{col}' but got: {result.dataframe[col].tolist()}"
            )

    def test_single_metric_empty_produces_nan_column(self, mock_boto3_client):
        """
        When only one metric (NetworkIn) returns empty Datapoints, only that
        column should be NaN; the others should have values.
        """
        mock_cw, mock_s3 = mock_boto3_client

        metrics_order = [
            "CPUUtilization",
            "NetworkIn",
            "NetworkOut",
            "DiskReadBytes",
            "DiskWriteBytes",
        ]

        def _side_effect(**kwargs):
            metric_name = kwargs["MetricName"]
            if metric_name == "NetworkIn":
                return _empty_cw_response()
            return _full_cw_response(value=10.0)

        mock_cw.get_metric_statistics.side_effect = _side_effect

        config = _make_config()
        result = collect_metrics(config)

        assert result.dataframe["NetworkIn"].isna().all(), (
            "NetworkIn should be all-NaN when CloudWatch returns empty Datapoints"
        )
        for col in ["CPUUtilization", "NetworkOut", "DiskReadBytes", "DiskWriteBytes"]:
            assert not result.dataframe[col].isna().all(), (
                f"Column '{col}' should not be all-NaN when CloudWatch returns data"
            )

    def test_result_is_collection_result_instance(self, mock_boto3_client):
        """collect_metrics always returns a CollectionResult."""
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        result = collect_metrics(_make_config())
        assert isinstance(result, CollectionResult)

    def test_dataframe_has_timestamp_column(self, mock_boto3_client):
        """The returned DataFrame always contains a 'timestamp' column."""
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        result = collect_metrics(_make_config())
        assert "timestamp" in result.dataframe.columns


# ---------------------------------------------------------------------------
# Test: S3 write uses the raw/ prefix
# ---------------------------------------------------------------------------

class TestS3WriteUsesRawPrefix:
    """
    Requirement 2.3 — the S3 key used in put_object must start with 'raw/'.
    """

    def test_s3_key_starts_with_raw_prefix(self, mock_boto3_client):
        """The S3 key passed to put_object starts with 'raw/'."""
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        collect_metrics(_make_config())

        mock_s3.put_object.assert_called_once()
        _, kwargs = mock_s3.put_object.call_args
        assert kwargs["Key"].startswith("raw/"), (
            f"Expected S3 key to start with 'raw/' but got: {kwargs['Key']}"
        )

    def test_s3_key_in_collection_result_starts_with_raw_prefix(self, mock_boto3_client):
        """The s3_key field of the returned CollectionResult also starts with 'raw/'."""
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        result = collect_metrics(_make_config())
        assert result.s3_key.startswith("raw/"), (
            f"CollectionResult.s3_key should start with 'raw/' but got: {result.s3_key}"
        )

    def test_s3_bucket_matches_config(self, mock_boto3_client):
        """The Bucket argument to put_object matches config.s3_bucket."""
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        config = _make_config(s3_bucket="my-special-bucket")
        collect_metrics(config)

        _, kwargs = mock_s3.put_object.call_args
        assert kwargs["Bucket"] == "my-special-bucket"

    def test_s3_put_object_called_exactly_once(self, mock_boto3_client):
        """put_object is called exactly once per collect_metrics invocation."""
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        collect_metrics(_make_config())
        assert mock_s3.put_object.call_count == 1

    def test_s3_key_contains_csv_extension(self, mock_boto3_client):
        """The S3 key ends with '.csv'."""
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        collect_metrics(_make_config())

        _, kwargs = mock_s3.put_object.call_args
        assert kwargs["Key"].endswith(".csv"), (
            f"Expected S3 key to end with '.csv' but got: {kwargs['Key']}"
        )


# ---------------------------------------------------------------------------
# Test: INSTANCE_ID env var is used as the instance identifier
# ---------------------------------------------------------------------------

class TestInstanceIdUsedInCloudWatchCalls:
    """
    Requirement 2.1 / 2.6 — the instance_id from PipelineConfig (sourced from
    the INSTANCE_ID env var) must be passed to every CloudWatch API call.
    """

    def test_instance_id_passed_to_cloudwatch_dimensions(self, mock_boto3_client):
        """
        The InstanceId dimension in every get_metric_statistics call matches
        config.instance_id.
        """
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        instance_id = "i-1234567890abcdef0"
        config = _make_config(instance_id=instance_id)
        collect_metrics(config)

        for call_args in mock_cw.get_metric_statistics.call_args_list:
            _, kwargs = call_args
            dimensions = kwargs["Dimensions"]
            instance_dim = next(
                (d for d in dimensions if d["Name"] == "InstanceId"), None
            )
            assert instance_dim is not None, "InstanceId dimension missing from CloudWatch call"
            assert instance_dim["Value"] == instance_id, (
                f"Expected InstanceId '{instance_id}' but got '{instance_dim['Value']}'"
            )

    def test_different_instance_id_is_propagated(self, mock_boto3_client):
        """A different instance_id in config is correctly forwarded to CloudWatch."""
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        other_id = "i-deadbeef00000000"
        config = _make_config(instance_id=other_id)
        collect_metrics(config)

        for call_args in mock_cw.get_metric_statistics.call_args_list:
            _, kwargs = call_args
            dimensions = kwargs["Dimensions"]
            instance_dim = next(d for d in dimensions if d["Name"] == "InstanceId")
            assert instance_dim["Value"] == other_id

    def test_instance_id_in_s3_key(self, mock_boto3_client):
        """The S3 key contains the instance_id so artifacts are traceable."""
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        instance_id = "i-1234567890abcdef0"
        config = _make_config(instance_id=instance_id)
        result = collect_metrics(config)

        assert instance_id in result.s3_key, (
            f"Expected instance_id '{instance_id}' in S3 key '{result.s3_key}'"
        )

    def test_cloudwatch_called_for_all_five_metrics(self, mock_boto3_client):
        """
        Requirement 2.1 — get_metric_statistics is called exactly five times,
        once for each of the five EC2 metrics.
        """
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        collect_metrics(_make_config())

        assert mock_cw.get_metric_statistics.call_count == 5

    def test_all_five_metric_names_requested(self, mock_boto3_client):
        """All five metric names are requested from CloudWatch."""
        mock_cw, mock_s3 = mock_boto3_client
        mock_cw.get_metric_statistics.return_value = _empty_cw_response()

        collect_metrics(_make_config())

        requested_metrics = {
            call_args[1]["MetricName"]
            for call_args in mock_cw.get_metric_statistics.call_args_list
        }
        expected_metrics = {
            "CPUUtilization",
            "NetworkIn",
            "NetworkOut",
            "DiskReadBytes",
            "DiskWriteBytes",
        }
        assert requested_metrics == expected_metrics
