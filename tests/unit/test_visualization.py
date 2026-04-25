"""
Unit tests for visualization.py.

Requirements: 6.1, 6.3, 6.5
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.visualization import generate_plots
from src.utils import PipelineConfig

# visualization.py imports VisualizationResult from the bare 'utils' module
# (no 'src.' prefix), so we must use the same import path to get the same
# class object for isinstance() checks.
import utils as _utils
VisualizationResult = _utils.VisualizationResult


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


def _make_results_df_with_anomalies() -> pd.DataFrame:
    """
    Return a results DataFrame that contains some anomaly rows
    (anomaly_label == -1) alongside normal rows (anomaly_label == 1).
    """
    return pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 11, 0, 0),
                datetime(2024, 1, 1, 11, 5, 0),
                datetime(2024, 1, 1, 11, 10, 0),
                datetime(2024, 1, 1, 11, 15, 0),
                datetime(2024, 1, 1, 11, 20, 0),
            ],
            "CPUUtilization": [10.0, 95.0, 12.0, 88.0, 11.0],
            "NetworkIn": [100.0, 5000.0, 110.0, 4800.0, 105.0],
            "NetworkOut": [50.0, 2500.0, 55.0, 2400.0, 52.0],
            "DiskReadBytes": [1024.0, 50000.0, 1100.0, 48000.0, 1050.0],
            "DiskWriteBytes": [512.0, 25000.0, 550.0, 24000.0, 530.0],
            "anomaly_label": [1, -1, 1, -1, 1],
        }
    )


def _make_results_df_no_anomalies() -> pd.DataFrame:
    """
    Return a results DataFrame where ALL rows are normal (anomaly_label == 1).
    No anomaly rows exist.
    """
    return pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 11, 0, 0),
                datetime(2024, 1, 1, 11, 5, 0),
                datetime(2024, 1, 1, 11, 10, 0),
                datetime(2024, 1, 1, 11, 15, 0),
                datetime(2024, 1, 1, 11, 20, 0),
            ],
            "CPUUtilization": [10.0, 12.0, 11.0, 13.0, 10.5],
            "NetworkIn": [100.0, 110.0, 105.0, 108.0, 102.0],
            "NetworkOut": [50.0, 55.0, 52.0, 54.0, 51.0],
            "DiskReadBytes": [1024.0, 1100.0, 1050.0, 1080.0, 1030.0],
            "DiskWriteBytes": [512.0, 550.0, 530.0, 540.0, 515.0],
            "anomaly_label": [1, 1, 1, 1, 1],
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_s3_client():
    """
    Patches boto3.client inside visualization so no real AWS calls are made.

    Yields a mock S3 client for per-test inspection.
    """
    mock_s3 = MagicMock()

    def _client_factory(service, **kwargs):
        if service == "s3":
            return mock_s3
        raise ValueError(f"Unexpected boto3 service: {service}")

    with patch("visualization.boto3.client", side_effect=_client_factory):
        yield mock_s3


# ---------------------------------------------------------------------------
# Test: generate_plots writes exactly five PNG files to S3  (Req 6.1)
# ---------------------------------------------------------------------------

class TestGeneratePlotsWritesFivePNGs:
    """
    Requirement 6.1 — generate_plots must produce exactly five PNG files,
    one per metric (CPUUtilization, NetworkIn, NetworkOut, DiskReadBytes,
    DiskWriteBytes).
    """

    def test_put_object_called_exactly_five_times_with_anomalies(self, mock_s3_client):
        """put_object is called exactly five times when anomalies are present."""
        config = _make_config()
        df = _make_results_df_with_anomalies()

        generate_plots(df, config)

        assert mock_s3_client.put_object.call_count == 5, (
            f"Expected 5 put_object calls, got {mock_s3_client.put_object.call_count}"
        )

    def test_put_object_called_exactly_five_times_no_anomalies(self, mock_s3_client):
        """put_object is called exactly five times even when no anomalies exist."""
        config = _make_config()
        df = _make_results_df_no_anomalies()

        generate_plots(df, config)

        assert mock_s3_client.put_object.call_count == 5, (
            f"Expected 5 put_object calls, got {mock_s3_client.put_object.call_count}"
        )

    def test_returns_visualization_result_with_five_keys(self, mock_s3_client):
        """generate_plots returns a VisualizationResult containing exactly five S3 keys."""
        config = _make_config()
        df = _make_results_df_with_anomalies()

        result = generate_plots(df, config)

        assert isinstance(result, VisualizationResult)
        assert len(result.s3_keys) == 5, (
            f"Expected 5 S3 keys in VisualizationResult, got {len(result.s3_keys)}"
        )

    def test_returns_visualization_result_instance(self, mock_s3_client):
        """generate_plots returns a VisualizationResult instance."""
        config = _make_config()
        df = _make_results_df_with_anomalies()

        result = generate_plots(df, config)

        assert isinstance(result, VisualizationResult)

    def test_each_put_object_call_has_body(self, mock_s3_client):
        """Each put_object call includes a non-empty Body (PNG bytes)."""
        config = _make_config()
        df = _make_results_df_with_anomalies()

        generate_plots(df, config)

        for i, call_args in enumerate(mock_s3_client.put_object.call_args_list):
            _, kwargs = call_args
            assert "Body" in kwargs, f"Call {i} missing 'Body' argument"
            assert len(kwargs["Body"]) > 0, f"Call {i} has empty Body"

    def test_each_put_object_call_uses_correct_bucket(self, mock_s3_client):
        """Each put_object call uses the bucket from config."""
        config = _make_config(s3_bucket="my-viz-bucket")
        df = _make_results_df_with_anomalies()

        generate_plots(df, config)

        for i, call_args in enumerate(mock_s3_client.put_object.call_args_list):
            _, kwargs = call_args
            assert kwargs["Bucket"] == "my-viz-bucket", (
                f"Call {i}: expected Bucket='my-viz-bucket', got '{kwargs['Bucket']}'"
            )


# ---------------------------------------------------------------------------
# Test: all five S3 keys use the results/plots/ prefix  (Req 6.3)
# ---------------------------------------------------------------------------

class TestGeneratePlotsS3KeyPrefix:
    """
    Requirement 6.3 — all S3 keys produced by generate_plots must start
    with the 'results/plots/' prefix.
    """

    def test_all_put_object_keys_start_with_results_plots_prefix(self, mock_s3_client):
        """Every Key argument in put_object calls starts with 'results/plots/'."""
        config = _make_config()
        df = _make_results_df_with_anomalies()

        generate_plots(df, config)

        for i, call_args in enumerate(mock_s3_client.put_object.call_args_list):
            _, kwargs = call_args
            assert kwargs["Key"].startswith("results/plots/"), (
                f"Call {i}: expected key to start with 'results/plots/', "
                f"got '{kwargs['Key']}'"
            )

    def test_all_returned_s3_keys_start_with_results_plots_prefix(self, mock_s3_client):
        """Every key in VisualizationResult.s3_keys starts with 'results/plots/'."""
        config = _make_config()
        df = _make_results_df_with_anomalies()

        result = generate_plots(df, config)

        for key in result.s3_keys:
            assert key.startswith("results/plots/"), (
                f"Expected key to start with 'results/plots/', got '{key}'"
            )

    def test_all_keys_end_with_png_extension(self, mock_s3_client):
        """Every S3 key ends with '.png'."""
        config = _make_config()
        df = _make_results_df_with_anomalies()

        result = generate_plots(df, config)

        for key in result.s3_keys:
            assert key.endswith(".png"), (
                f"Expected key to end with '.png', got '{key}'"
            )

    def test_keys_contain_instance_id(self, mock_s3_client):
        """Every S3 key contains the instance_id for artifact traceability."""
        instance_id = "i-1234567890abcdef0"
        config = _make_config(instance_id=instance_id)
        df = _make_results_df_with_anomalies()

        result = generate_plots(df, config)

        for key in result.s3_keys:
            assert instance_id in key, (
                f"Expected instance_id '{instance_id}' in key '{key}'"
            )

    def test_keys_are_unique(self, mock_s3_client):
        """All five S3 keys are distinct (one per metric)."""
        config = _make_config()
        df = _make_results_df_with_anomalies()

        result = generate_plots(df, config)

        assert len(set(result.s3_keys)) == 5, (
            f"Expected 5 unique S3 keys, got {len(set(result.s3_keys))}: {result.s3_keys}"
        )

    def test_keys_contain_metric_names(self, mock_s3_client):
        """Each S3 key contains the corresponding metric name."""
        config = _make_config()
        df = _make_results_df_with_anomalies()
        expected_metrics = [
            "CPUUtilization",
            "NetworkIn",
            "NetworkOut",
            "DiskReadBytes",
            "DiskWriteBytes",
        ]

        result = generate_plots(df, config)

        for metric in expected_metrics:
            assert any(metric in key for key in result.s3_keys), (
                f"No S3 key contains metric name '{metric}'. Keys: {result.s3_keys}"
            )


# ---------------------------------------------------------------------------
# Test: generate_plots runs without error when no anomaly rows exist  (Req 6.5)
# ---------------------------------------------------------------------------

class TestGeneratePlotsZeroAnomalies:
    """
    Requirement 6.5 — generate_plots must complete successfully and still
    produce five PNG files even when no rows have anomaly_label == -1.
    """

    def test_no_exception_raised_when_no_anomalies(self, mock_s3_client):
        """generate_plots does not raise when all anomaly_label values are 1."""
        config = _make_config()
        df = _make_results_df_no_anomalies()

        # Should not raise
        generate_plots(df, config)

    def test_still_writes_five_files_when_no_anomalies(self, mock_s3_client):
        """Five PNG files are still uploaded to S3 when there are no anomalies."""
        config = _make_config()
        df = _make_results_df_no_anomalies()

        result = generate_plots(df, config)

        assert mock_s3_client.put_object.call_count == 5
        assert len(result.s3_keys) == 5

    def test_all_keys_use_results_plots_prefix_when_no_anomalies(self, mock_s3_client):
        """All S3 keys still use 'results/plots/' prefix when no anomalies exist."""
        config = _make_config()
        df = _make_results_df_no_anomalies()

        result = generate_plots(df, config)

        for key in result.s3_keys:
            assert key.startswith("results/plots/"), (
                f"Expected key to start with 'results/plots/', got '{key}'"
            )

    def test_returns_visualization_result_when_no_anomalies(self, mock_s3_client):
        """generate_plots returns a VisualizationResult even with zero anomalies."""
        config = _make_config()
        df = _make_results_df_no_anomalies()

        result = generate_plots(df, config)

        assert isinstance(result, VisualizationResult)
        assert len(result.s3_keys) == 5
