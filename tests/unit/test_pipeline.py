"""
Unit tests for pipeline.py.

Requirements: 1.2, 5.5, 7.3, 10.3
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.utils import (
    CollectionResult,
    DetectionResult,
    PipelineConfig,
    PipelineResult,
    PreprocessResult,
    TrainingResult,
    VisualizationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> PipelineConfig:
    """Return a PipelineConfig populated with safe test defaults."""
    defaults = dict(
        s3_bucket="test-bucket",
        sns_topic_arn="arn:aws:sns:us-east-1:123456789012:test-topic",
        instance_id="i-1234567890abcdef0",
        contamination=0.1,
        n_estimators=100,
        random_state=42,
        aws_region="us-east-1",
        run_timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_results_df(anomaly_count: int = 0, cpu_max: float = 50.0) -> pd.DataFrame:
    """
    Return a results DataFrame with an anomaly_label column.

    Args:
        anomaly_count: Number of rows to mark as anomalies (-1).
        cpu_max: Maximum CPUUtilization value to include.
    """
    n_rows = max(5, anomaly_count)
    labels = [-1] * anomaly_count + [1] * (n_rows - anomaly_count)
    cpu_values = [cpu_max] + [10.0] * (n_rows - 1)
    return pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 11, i, 0) for i in range(n_rows)],
            "CPUUtilization": cpu_values,
            "NetworkIn": [0.2] * n_rows,
            "NetworkOut": [0.1] * n_rows,
            "DiskReadBytes": [0.0] * n_rows,
            "DiskWriteBytes": [0.1] * n_rows,
            "anomaly_label": labels,
        }
    )


def _make_pipeline_result(anomaly_count: int = 0) -> PipelineResult:
    """Return a PipelineResult with the given anomaly count."""
    return PipelineResult(
        status="success",
        anomaly_count=anomaly_count,
        stage_artifacts={
            "raw": "raw/metrics_i-1234_20240101T120000.csv",
            "processed": "processed/metrics_i-1234_20240101T120000.csv",
            "model": "models/model_i-1234_20240101T120000.joblib",
            "results": "results/results_i-1234_20240101T120000.csv",
            "plots": ["results/plots/CPUUtilization_i-1234_20240101T120000.png"],
        },
    )


# ---------------------------------------------------------------------------
# Test: lambda_handler calls run_pipeline and returns a status dict  (Req 1.2)
# ---------------------------------------------------------------------------

class TestLambdaHandler:
    """
    Requirement 1.2 — lambda_handler must call run_pipeline and return a
    status dict with 'status', 'anomaly_count', and 'stage_artifacts' keys.
    """

    def test_lambda_handler_returns_dict_with_required_keys(self, monkeypatch):
        """lambda_handler returns a dict containing status, anomaly_count, stage_artifacts."""
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:test-topic")
        monkeypatch.setenv("INSTANCE_ID", "i-1234567890abcdef0")
        monkeypatch.setenv("CONTAMINATION", "0.1")
        monkeypatch.setenv("N_ESTIMATORS", "100")
        monkeypatch.setenv("RANDOM_STATE", "42")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        expected_result = _make_pipeline_result(anomaly_count=2)

        import src.pipeline as _pipeline_mod
        with patch.object(_pipeline_mod, "run_pipeline", return_value=expected_result):
            result = _pipeline_mod.lambda_handler({}, None)

        assert isinstance(result, dict), "lambda_handler must return a dict"
        assert "status" in result, "Return dict must contain 'status' key"
        assert "anomaly_count" in result, "Return dict must contain 'anomaly_count' key"
        assert "stage_artifacts" in result, "Return dict must contain 'stage_artifacts' key"

    def test_lambda_handler_calls_run_pipeline_once(self, monkeypatch):
        """lambda_handler calls run_pipeline exactly once."""
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:test-topic")
        monkeypatch.setenv("INSTANCE_ID", "i-1234567890abcdef0")
        monkeypatch.setenv("CONTAMINATION", "0.1")
        monkeypatch.setenv("N_ESTIMATORS", "100")
        monkeypatch.setenv("RANDOM_STATE", "42")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        expected_result = _make_pipeline_result()

        import src.pipeline as _pipeline_mod
        with patch.object(_pipeline_mod, "run_pipeline", return_value=expected_result) as mock_run:
            _pipeline_mod.lambda_handler({}, None)

        mock_run.assert_called_once()

    def test_lambda_handler_status_value_matches_pipeline_result(self, monkeypatch):
        """The 'status' value in the returned dict matches PipelineResult.status."""
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:test-topic")
        monkeypatch.setenv("INSTANCE_ID", "i-1234567890abcdef0")
        monkeypatch.setenv("CONTAMINATION", "0.1")
        monkeypatch.setenv("N_ESTIMATORS", "100")
        monkeypatch.setenv("RANDOM_STATE", "42")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        expected_result = _make_pipeline_result(anomaly_count=3)

        import src.pipeline as _pipeline_mod
        with patch.object(_pipeline_mod, "run_pipeline", return_value=expected_result):
            result = _pipeline_mod.lambda_handler({}, None)

        assert result["status"] == "success"
        assert result["anomaly_count"] == 3

    def test_lambda_handler_passes_pipeline_config_to_run_pipeline(self, monkeypatch):
        """lambda_handler passes a PipelineConfig instance to run_pipeline."""
        monkeypatch.setenv("S3_BUCKET", "my-bucket")
        monkeypatch.setenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:my-topic")
        monkeypatch.setenv("INSTANCE_ID", "i-abcdef1234567890")
        monkeypatch.setenv("CONTAMINATION", "0.05")
        monkeypatch.setenv("N_ESTIMATORS", "50")
        monkeypatch.setenv("RANDOM_STATE", "0")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")

        expected_result = _make_pipeline_result()

        import src.pipeline as _pipeline_mod
        with patch.object(_pipeline_mod, "run_pipeline", return_value=expected_result) as mock_run:
            _pipeline_mod.lambda_handler({}, None)

        config_arg = mock_run.call_args[0][0]
        assert config_arg.__class__.__name__ == "PipelineConfig"
        assert config_arg.s3_bucket == "my-bucket"
        assert config_arg.instance_id == "i-abcdef1234567890"
        assert config_arg.contamination == 0.05

    def test_lambda_handler_propagates_exception_from_run_pipeline(self, monkeypatch):
        """lambda_handler does not suppress exceptions raised by run_pipeline."""
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:test-topic")
        monkeypatch.setenv("INSTANCE_ID", "i-1234567890abcdef0")
        monkeypatch.setenv("CONTAMINATION", "0.1")
        monkeypatch.setenv("N_ESTIMATORS", "100")
        monkeypatch.setenv("RANDOM_STATE", "42")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        import src.pipeline as _pipeline_mod
        with patch.object(_pipeline_mod, "run_pipeline", side_effect=RuntimeError("stage failed")):
            with pytest.raises(RuntimeError, match="stage failed"):
                _pipeline_mod.lambda_handler({}, None)


# ---------------------------------------------------------------------------
# Test: run_pipeline calls train_model when load_latest_model returns None  (Req 5.5)
# ---------------------------------------------------------------------------

class TestRunPipelineTrainsWhenNoModelExists:
    """
    Requirement 5.5 — when load_latest_model returns None, run_pipeline must
    call train_model before detect_anomalies.
    """

    def _make_stage_mocks(self, load_model_return=None):
        """
        Build a dict of mock stage functions for patching pipeline module.

        Args:
            load_model_return: Return value for load_latest_model mock.
                               None means no existing model (triggers training).
        """
        config = _make_config()
        results_df = _make_results_df(anomaly_count=0)

        mock_collect = MagicMock(
            return_value=CollectionResult(
                s3_key="raw/metrics_i-1234_20240101T120000.csv",
                dataframe=pd.DataFrame({"timestamp": [], "CPUUtilization": []}),
            )
        )
        mock_preprocess = MagicMock(
            return_value=PreprocessResult(
                s3_key="processed/metrics_i-1234_20240101T120000.csv",
                dataframe=pd.DataFrame({"timestamp": [], "CPUUtilization": []}),
            )
        )
        mock_load_model = MagicMock(return_value=load_model_return)

        mock_model_obj = MagicMock()
        mock_train = MagicMock(
            return_value=TrainingResult(
                s3_key="models/model_i-1234_20240101T120000.joblib",
                model=mock_model_obj,
            )
        )
        mock_detect = MagicMock(
            return_value=DetectionResult(
                s3_key="results/results_i-1234_20240101T120000.csv",
                dataframe=results_df,
                anomaly_count=0,
            )
        )
        mock_plots = MagicMock(
            return_value=VisualizationResult(
                s3_keys=["results/plots/CPUUtilization_i-1234_20240101T120000.png"]
            )
        )
        mock_send_alert = MagicMock()

        return {
            "collect": mock_collect,
            "preprocess": mock_preprocess,
            "load_model": mock_load_model,
            "train": mock_train,
            "detect": mock_detect,
            "plots": mock_plots,
            "send_alert": mock_send_alert,
        }

    def test_train_model_called_when_load_returns_none(self):
        """train_model is called when load_latest_model returns None."""
        import src.pipeline as _pipeline_mod
        mocks = self._make_stage_mocks(load_model_return=None)

        with (
            patch.object(_pipeline_mod, "collect_metrics", mocks["collect"]),
            patch.object(_pipeline_mod, "preprocess", mocks["preprocess"]),
            patch.object(_pipeline_mod, "load_latest_model", mocks["load_model"]),
            patch.object(_pipeline_mod, "train_model", mocks["train"]),
            patch.object(_pipeline_mod, "detect_anomalies", mocks["detect"]),
            patch.object(_pipeline_mod, "generate_plots", mocks["plots"]),
            patch.object(_pipeline_mod, "_send_alert", mocks["send_alert"]),
        ):
            _pipeline_mod.run_pipeline(_make_config())

        mocks["train"].assert_called_once()

    def test_train_model_called_before_detect_anomalies(self):
        """train_model is called before detect_anomalies when no model exists."""
        import src.pipeline as _pipeline_mod
        call_order = []

        mocks = self._make_stage_mocks(load_model_return=None)

        def _train_side_effect(*args, **kwargs):
            call_order.append("train_model")
            return mocks["train"].return_value

        def _detect_side_effect(*args, **kwargs):
            call_order.append("detect_anomalies")
            return mocks["detect"].return_value

        mocks["train"].side_effect = _train_side_effect
        mocks["detect"].side_effect = _detect_side_effect

        with (
            patch.object(_pipeline_mod, "collect_metrics", mocks["collect"]),
            patch.object(_pipeline_mod, "preprocess", mocks["preprocess"]),
            patch.object(_pipeline_mod, "load_latest_model", mocks["load_model"]),
            patch.object(_pipeline_mod, "train_model", mocks["train"]),
            patch.object(_pipeline_mod, "detect_anomalies", mocks["detect"]),
            patch.object(_pipeline_mod, "generate_plots", mocks["plots"]),
            patch.object(_pipeline_mod, "_send_alert", mocks["send_alert"]),
        ):
            _pipeline_mod.run_pipeline(_make_config())

        assert call_order.index("train_model") < call_order.index("detect_anomalies"), (
            f"Expected train_model before detect_anomalies, got order: {call_order}"
        )

    def test_train_model_not_called_when_model_exists(self):
        """train_model is NOT called when load_latest_model returns an existing model."""
        import src.pipeline as _pipeline_mod
        existing_model = MagicMock()
        mocks = self._make_stage_mocks(load_model_return=existing_model)

        with (
            patch.object(_pipeline_mod, "collect_metrics", mocks["collect"]),
            patch.object(_pipeline_mod, "preprocess", mocks["preprocess"]),
            patch.object(_pipeline_mod, "load_latest_model", mocks["load_model"]),
            patch.object(_pipeline_mod, "train_model", mocks["train"]),
            patch.object(_pipeline_mod, "detect_anomalies", mocks["detect"]),
            patch.object(_pipeline_mod, "generate_plots", mocks["plots"]),
            patch.object(_pipeline_mod, "_send_alert", mocks["send_alert"]),
        ):
            _pipeline_mod.run_pipeline(_make_config())

        mocks["train"].assert_not_called()

    def test_detect_anomalies_uses_trained_model_when_no_existing_model(self):
        """detect_anomalies receives the model returned by train_model."""
        import src.pipeline as _pipeline_mod
        mocks = self._make_stage_mocks(load_model_return=None)
        trained_model = mocks["train"].return_value.model

        with (
            patch.object(_pipeline_mod, "collect_metrics", mocks["collect"]),
            patch.object(_pipeline_mod, "preprocess", mocks["preprocess"]),
            patch.object(_pipeline_mod, "load_latest_model", mocks["load_model"]),
            patch.object(_pipeline_mod, "train_model", mocks["train"]),
            patch.object(_pipeline_mod, "detect_anomalies", mocks["detect"]),
            patch.object(_pipeline_mod, "generate_plots", mocks["plots"]),
            patch.object(_pipeline_mod, "_send_alert", mocks["send_alert"]),
        ):
            _pipeline_mod.run_pipeline(_make_config())

        detect_call_args = mocks["detect"].call_args
        # Second positional arg to detect_anomalies is the model
        assert detect_call_args[0][1] is trained_model, (
            "detect_anomalies should receive the model returned by train_model"
        )

    def test_run_pipeline_returns_success_status(self):
        """run_pipeline returns PipelineResult with status='success'."""
        import src.pipeline as _pipeline_mod
        mocks = self._make_stage_mocks(load_model_return=None)

        with (
            patch.object(_pipeline_mod, "collect_metrics", mocks["collect"]),
            patch.object(_pipeline_mod, "preprocess", mocks["preprocess"]),
            patch.object(_pipeline_mod, "load_latest_model", mocks["load_model"]),
            patch.object(_pipeline_mod, "train_model", mocks["train"]),
            patch.object(_pipeline_mod, "detect_anomalies", mocks["detect"]),
            patch.object(_pipeline_mod, "generate_plots", mocks["plots"]),
            patch.object(_pipeline_mod, "_send_alert", mocks["send_alert"]),
        ):
            result = _pipeline_mod.run_pipeline(_make_config())

        assert result.__class__.__name__ == "PipelineResult"
        assert result.status == "success"


# ---------------------------------------------------------------------------
# Test: SNS_TOPIC_ARN env var value is used in the SNS publish call  (Req 7.3)
# ---------------------------------------------------------------------------

class TestSnsTopicArnUsedInPublish:
    """
    Requirement 7.3 — _send_alert must publish to the SNS topic ARN from config.
    """

    def test_sns_publish_uses_topic_arn_from_config(self):
        """SNS publish is called with the topic ARN from config.sns_topic_arn."""
        import src.pipeline as _pipeline_mod
        topic_arn = "arn:aws:sns:us-east-1:123456789012:my-alert-topic"
        config = _make_config(sns_topic_arn=topic_arn)
        results_df = _make_results_df(anomaly_count=2, cpu_max=50.0)
        detection_result = DetectionResult(
            s3_key="results/results.csv",
            dataframe=results_df,
            anomaly_count=2,
        )

        mock_sns = MagicMock()

        with patch.object(_pipeline_mod.boto3, "client", return_value=mock_sns):
            _pipeline_mod._send_alert(results_df, detection_result, config)

        mock_sns.publish.assert_called_once()
        _, kwargs = mock_sns.publish.call_args
        assert kwargs["TopicArn"] == topic_arn, (
            f"Expected TopicArn='{topic_arn}', got '{kwargs.get('TopicArn')}'"
        )

    def test_sns_publish_uses_different_topic_arn(self):
        """SNS publish uses the correct ARN when a different topic ARN is configured."""
        import src.pipeline as _pipeline_mod
        topic_arn = "arn:aws:sns:eu-west-1:999888777666:another-topic"
        config = _make_config(sns_topic_arn=topic_arn)
        results_df = _make_results_df(anomaly_count=1, cpu_max=50.0)
        detection_result = DetectionResult(
            s3_key="results/results.csv",
            dataframe=results_df,
            anomaly_count=1,
        )

        mock_sns = MagicMock()

        with patch.object(_pipeline_mod.boto3, "client", return_value=mock_sns):
            _pipeline_mod._send_alert(results_df, detection_result, config)

        _, kwargs = mock_sns.publish.call_args
        assert kwargs["TopicArn"] == topic_arn

    def test_sns_not_published_when_no_anomalies_and_cpu_below_threshold(self):
        """SNS publish is NOT called when anomaly_count == 0 and CPU <= 90%."""
        import src.pipeline as _pipeline_mod
        config = _make_config()
        results_df = _make_results_df(anomaly_count=0, cpu_max=50.0)
        detection_result = DetectionResult(
            s3_key="results/results.csv",
            dataframe=results_df,
            anomaly_count=0,
        )

        mock_sns = MagicMock()

        with patch.object(_pipeline_mod.boto3, "client", return_value=mock_sns):
            _pipeline_mod._send_alert(results_df, detection_result, config)

        mock_sns.publish.assert_not_called()

    def test_sns_published_when_cpu_exceeds_threshold_even_with_no_anomalies(self):
        """SNS publish IS called when CPUUtilization > 90 even if anomaly_count == 0."""
        import src.pipeline as _pipeline_mod
        topic_arn = "arn:aws:sns:us-east-1:123456789012:cpu-alert-topic"
        config = _make_config(sns_topic_arn=topic_arn)
        results_df = _make_results_df(anomaly_count=0, cpu_max=95.0)
        detection_result = DetectionResult(
            s3_key="results/results.csv",
            dataframe=results_df,
            anomaly_count=0,
        )

        mock_sns = MagicMock()

        with patch.object(_pipeline_mod.boto3, "client", return_value=mock_sns):
            _pipeline_mod._send_alert(results_df, detection_result, config)

        mock_sns.publish.assert_called_once()
        _, kwargs = mock_sns.publish.call_args
        assert kwargs["TopicArn"] == topic_arn

    def test_sns_message_contains_instance_id(self):
        """The SNS message body contains the instance ID."""
        import src.pipeline as _pipeline_mod
        instance_id = "i-1234567890abcdef0"
        config = _make_config(instance_id=instance_id)
        results_df = _make_results_df(anomaly_count=1, cpu_max=50.0)
        detection_result = DetectionResult(
            s3_key="results/results.csv",
            dataframe=results_df,
            anomaly_count=1,
        )

        mock_sns = MagicMock()

        with patch.object(_pipeline_mod.boto3, "client", return_value=mock_sns):
            _pipeline_mod._send_alert(results_df, detection_result, config)

        _, kwargs = mock_sns.publish.call_args
        assert instance_id in kwargs["Message"], (
            f"Expected instance_id '{instance_id}' in SNS message"
        )


# ---------------------------------------------------------------------------
# Test: loggers use StreamHandler(sys.stdout)  (Req 10.3)
# ---------------------------------------------------------------------------

class TestLoggerUsesStreamHandlerStdout:
    """
    Requirement 10.3 — loggers must write to sys.stdout via StreamHandler.
    """

    def test_pipeline_logger_has_stream_handler(self):
        """The pipeline module logger has at least one StreamHandler."""
        import src.pipeline as _pipeline_mod

        logger = _pipeline_mod.logger
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) >= 1, (
            f"Expected at least one StreamHandler on pipeline logger, "
            f"got handlers: {logger.handlers}"
        )

    def test_pipeline_logger_stream_handler_points_to_stdout(self):
        """The pipeline module logger's StreamHandler writes to sys.stdout."""
        import src.pipeline as _pipeline_mod

        logger = _pipeline_mod.logger
        stdout_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
        ]
        assert len(stdout_handlers) >= 1, (
            f"Expected StreamHandler pointing to sys.stdout on pipeline logger, "
            f"got handlers: {[(type(h).__name__, getattr(h, 'stream', None)) for h in logger.handlers]}"
        )

    def test_get_logger_returns_logger_with_stdout_handler(self):
        """get_logger() returns a logger whose StreamHandler points to sys.stdout."""
        from src.utils import get_logger

        test_logger = get_logger("test_pipeline_logger_stdout")
        stdout_handlers = [
            h for h in test_logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
        ]
        assert len(stdout_handlers) >= 1, (
            f"Expected StreamHandler pointing to sys.stdout, "
            f"got handlers: {[(type(h).__name__, getattr(h, 'stream', None)) for h in test_logger.handlers]}"
        )

    def test_pipeline_logger_level_is_info(self):
        """The pipeline module logger is set to INFO level."""
        import src.pipeline as _pipeline_mod

        logger = _pipeline_mod.logger
        assert logger.level == logging.INFO, (
            f"Expected logger level INFO ({logging.INFO}), got {logger.level}"
        )
