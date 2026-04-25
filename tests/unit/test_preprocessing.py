"""
Unit tests for preprocessing.py.

Requirements: 3.3, 3.4
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.preprocessing import preprocess
from src.utils import PipelineConfig

import utils as _utils
InsufficientDataError = _utils.InsufficientDataError
PreprocessResult = _utils.PreprocessResult


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


def _make_valid_df() -> pd.DataFrame:
    """Return a valid DataFrame with at least 2 rows and metric columns."""
    return pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 11, 0, 0),
                datetime(2024, 1, 1, 11, 5, 0),
                datetime(2024, 1, 1, 11, 10, 0),
            ],
            "CPUUtilization": [10.0, 50.0, 90.0],
            "NetworkIn": [100.0, 200.0, 300.0],
            "NetworkOut": [50.0, 75.0, 100.0],
            "DiskReadBytes": [1024.0, 2048.0, 4096.0],
            "DiskWriteBytes": [512.0, 1024.0, 2048.0],
        }
    )


def _make_zero_row_df() -> pd.DataFrame:
    """Return a DataFrame with 0 rows but the expected metric columns."""
    return pd.DataFrame(
        columns=[
            "timestamp",
            "CPUUtilization",
            "NetworkIn",
            "NetworkOut",
            "DiskReadBytes",
            "DiskWriteBytes",
        ]
    )


def _make_one_row_df() -> pd.DataFrame:
    """Return a DataFrame with exactly 1 row and metric columns."""
    return pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 11, 0, 0)],
            "CPUUtilization": [42.0],
            "NetworkIn": [100.0],
            "NetworkOut": [50.0],
            "DiskReadBytes": [1024.0],
            "DiskWriteBytes": [512.0],
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_boto3_client():
    """
    Patches boto3.client inside preprocessing so no real AWS calls are made.

    Yields a mock S3 client for per-test inspection.
    """
    mock_s3 = MagicMock()

    def _client_factory(service, **kwargs):
        if service == "s3":
            return mock_s3
        raise ValueError(f"Unexpected boto3 service: {service}")

    with patch("preprocessing.boto3.client", side_effect=_client_factory):
        yield mock_s3


# ---------------------------------------------------------------------------
# Test: InsufficientDataError for 0-row DataFrames
# ---------------------------------------------------------------------------

class TestInsufficientDataErrorZeroRows:
    """
    Requirement 3.3 — preprocess must raise InsufficientDataError when the
    input DataFrame has 0 rows.
    """

    def test_raises_for_zero_row_dataframe(self, mock_boto3_client):
        """preprocess raises InsufficientDataError when given an empty DataFrame."""
        config = _make_config()
        df = _make_zero_row_df()

        with pytest.raises(InsufficientDataError):
            preprocess(df, config)

    def test_error_message_mentions_row_count(self, mock_boto3_client):
        """The InsufficientDataError message includes the actual row count."""
        config = _make_config()
        df = _make_zero_row_df()

        with pytest.raises(InsufficientDataError, match="0"):
            preprocess(df, config)

    def test_s3_not_called_for_zero_row_dataframe(self, mock_boto3_client):
        """S3 put_object is never called when InsufficientDataError is raised."""
        config = _make_config()
        df = _make_zero_row_df()

        with pytest.raises(InsufficientDataError):
            preprocess(df, config)

        mock_boto3_client.put_object.assert_not_called()


# ---------------------------------------------------------------------------
# Test: InsufficientDataError for 1-row DataFrames
# ---------------------------------------------------------------------------

class TestInsufficientDataErrorOneRow:
    """
    Requirement 3.4 — preprocess must raise InsufficientDataError when the
    input DataFrame has exactly 1 row.
    """

    def test_raises_for_one_row_dataframe(self, mock_boto3_client):
        """preprocess raises InsufficientDataError when given a 1-row DataFrame."""
        config = _make_config()
        df = _make_one_row_df()

        with pytest.raises(InsufficientDataError):
            preprocess(df, config)

    def test_error_message_mentions_row_count(self, mock_boto3_client):
        """The InsufficientDataError message includes the actual row count (1)."""
        config = _make_config()
        df = _make_one_row_df()

        with pytest.raises(InsufficientDataError, match="1"):
            preprocess(df, config)

    def test_s3_not_called_for_one_row_dataframe(self, mock_boto3_client):
        """S3 put_object is never called when InsufficientDataError is raised."""
        config = _make_config()
        df = _make_one_row_df()

        with pytest.raises(InsufficientDataError):
            preprocess(df, config)

        mock_boto3_client.put_object.assert_not_called()

    def test_two_rows_does_not_raise(self, mock_boto3_client):
        """A DataFrame with exactly 2 rows does not raise InsufficientDataError."""
        config = _make_config()
        df = _make_valid_df().head(2)

        # Should not raise
        result = preprocess(df, config)
        assert isinstance(result, PreprocessResult)


# ---------------------------------------------------------------------------
# Test: S3 write uses the processed/ prefix
# ---------------------------------------------------------------------------

class TestS3WriteUsesProcessedPrefix:
    """
    Requirement 3.3 — the S3 key used in put_object must start with 'processed/'.
    """

    def test_s3_key_starts_with_processed_prefix(self, mock_boto3_client):
        """The S3 key passed to put_object starts with 'processed/'."""
        config = _make_config()
        df = _make_valid_df()

        preprocess(df, config)

        mock_boto3_client.put_object.assert_called_once()
        _, kwargs = mock_boto3_client.put_object.call_args
        assert kwargs["Key"].startswith("processed/"), (
            f"Expected S3 key to start with 'processed/' but got: {kwargs['Key']}"
        )

    def test_s3_key_in_preprocess_result_starts_with_processed_prefix(
        self, mock_boto3_client
    ):
        """The s3_key field of the returned PreprocessResult also starts with 'processed/'."""
        config = _make_config()
        df = _make_valid_df()

        result = preprocess(df, config)
        assert result.s3_key.startswith("processed/"), (
            f"PreprocessResult.s3_key should start with 'processed/' but got: {result.s3_key}"
        )

    def test_s3_key_ends_with_csv_extension(self, mock_boto3_client):
        """The S3 key ends with '.csv'."""
        config = _make_config()
        df = _make_valid_df()

        preprocess(df, config)

        _, kwargs = mock_boto3_client.put_object.call_args
        assert kwargs["Key"].endswith(".csv"), (
            f"Expected S3 key to end with '.csv' but got: {kwargs['Key']}"
        )

    def test_s3_bucket_matches_config(self, mock_boto3_client):
        """The Bucket argument to put_object matches config.s3_bucket."""
        config = _make_config(s3_bucket="my-special-bucket")
        df = _make_valid_df()

        preprocess(df, config)

        _, kwargs = mock_boto3_client.put_object.call_args
        assert kwargs["Bucket"] == "my-special-bucket"

    def test_s3_put_object_called_exactly_once(self, mock_boto3_client):
        """put_object is called exactly once per preprocess invocation."""
        config = _make_config()
        df = _make_valid_df()

        preprocess(df, config)
        assert mock_boto3_client.put_object.call_count == 1

    def test_s3_key_contains_instance_id(self, mock_boto3_client):
        """The S3 key contains the instance_id for artifact traceability."""
        instance_id = "i-1234567890abcdef0"
        config = _make_config(instance_id=instance_id)
        df = _make_valid_df()

        result = preprocess(df, config)
        assert instance_id in result.s3_key, (
            f"Expected instance_id '{instance_id}' in S3 key '{result.s3_key}'"
        )

    def test_returns_preprocess_result_instance(self, mock_boto3_client):
        """preprocess returns a PreprocessResult on success."""
        config = _make_config()
        df = _make_valid_df()

        result = preprocess(df, config)
        assert isinstance(result, PreprocessResult)
