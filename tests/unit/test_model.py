"""
Unit tests for model.py.

Requirements: 4.3, 4.4, 4.5
"""
from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.model import train_model, _validate_hyperparameters
from src.utils import PipelineConfig

import utils as _utils
ConfigurationError = _utils.ConfigurationError
TrainingResult = _utils.TrainingResult


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


def _make_processed_df() -> pd.DataFrame:
    """
    Return a valid preprocessed DataFrame with a timestamp column and
    five metric columns (at least 5 rows).

    The timestamp column should be excluded from features by train_model.
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
            "CPUUtilization": [0.1, 0.5, 0.9, 0.3, 0.7],
            "NetworkIn": [0.2, 0.4, 0.6, 0.8, 1.0],
            "NetworkOut": [0.1, 0.3, 0.5, 0.7, 0.9],
            "DiskReadBytes": [0.0, 0.25, 0.5, 0.75, 1.0],
            "DiskWriteBytes": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_boto3_client():
    """
    Patches boto3.client inside model so no real AWS calls are made.

    Yields a mock S3 client for per-test inspection.
    """
    mock_s3 = MagicMock()

    def _client_factory(service, **kwargs):
        if service == "s3":
            return mock_s3
        raise ValueError(f"Unexpected boto3 service: {service}")

    with patch("model.boto3.client", side_effect=_client_factory):
        yield mock_s3


# ---------------------------------------------------------------------------
# Test: S3 write uses the models/ prefix  (Requirement 4.3)
# ---------------------------------------------------------------------------

class TestS3WriteUsesModelsPrefix:
    """
    Requirement 4.3 — the S3 key used in put_object must start with 'models/'.
    """

    def test_s3_key_starts_with_models_prefix(self, mock_boto3_client):
        """The S3 key passed to put_object starts with 'models/'."""
        config = _make_config()
        df = _make_processed_df()

        train_model(df, config)

        mock_boto3_client.put_object.assert_called_once()
        _, kwargs = mock_boto3_client.put_object.call_args
        assert kwargs["Key"].startswith("models/"), (
            f"Expected S3 key to start with 'models/' but got: {kwargs['Key']}"
        )

    def test_s3_key_in_training_result_starts_with_models_prefix(self, mock_boto3_client):
        """The s3_key field of the returned TrainingResult also starts with 'models/'."""
        config = _make_config()
        df = _make_processed_df()

        result = train_model(df, config)
        assert result.s3_key.startswith("models/"), (
            f"TrainingResult.s3_key should start with 'models/' but got: {result.s3_key}"
        )

    def test_s3_key_ends_with_joblib_extension(self, mock_boto3_client):
        """The S3 key ends with '.joblib'."""
        config = _make_config()
        df = _make_processed_df()

        train_model(df, config)

        _, kwargs = mock_boto3_client.put_object.call_args
        assert kwargs["Key"].endswith(".joblib"), (
            f"Expected S3 key to end with '.joblib' but got: {kwargs['Key']}"
        )

    def test_s3_bucket_matches_config(self, mock_boto3_client):
        """The Bucket argument to put_object matches config.s3_bucket."""
        config = _make_config(s3_bucket="my-model-bucket")
        df = _make_processed_df()

        train_model(df, config)

        _, kwargs = mock_boto3_client.put_object.call_args
        assert kwargs["Bucket"] == "my-model-bucket"

    def test_s3_put_object_called_exactly_once(self, mock_boto3_client):
        """put_object is called exactly once per train_model invocation."""
        config = _make_config()
        df = _make_processed_df()

        train_model(df, config)
        assert mock_boto3_client.put_object.call_count == 1

    def test_s3_key_contains_instance_id(self, mock_boto3_client):
        """The S3 key contains the instance_id for artifact traceability."""
        instance_id = "i-1234567890abcdef0"
        config = _make_config(instance_id=instance_id)
        df = _make_processed_df()

        result = train_model(df, config)
        assert instance_id in result.s3_key, (
            f"Expected instance_id '{instance_id}' in S3 key '{result.s3_key}'"
        )

    def test_returns_training_result_instance(self, mock_boto3_client):
        """train_model returns a TrainingResult on success."""
        config = _make_config()
        df = _make_processed_df()

        result = train_model(df, config)
        assert isinstance(result, TrainingResult)

    def test_timestamp_column_excluded_from_features(self, mock_boto3_client):
        """
        The timestamp column must not be used as a feature.
        We verify this indirectly: train_model succeeds even when the
        timestamp column contains datetime objects (not numeric).
        """
        config = _make_config()
        df = _make_processed_df()

        # Should not raise even though 'timestamp' is non-numeric
        result = train_model(df, config)
        assert result.model is not None


# ---------------------------------------------------------------------------
# Test: ConfigurationError raised for invalid hyperparameters  (Requirement 4.4)
# ---------------------------------------------------------------------------

class TestConfigurationErrorOnInvalidHyperparameters:
    """
    Requirement 4.4 — train_model (via _validate_hyperparameters) must raise
    ConfigurationError when hyperparameter values are missing or invalid.
    """

    def test_raises_for_contamination_zero(self):
        """ConfigurationError is raised when contamination == 0.0 (out of range)."""
        config = _make_config()
        object.__setattr__(config, "contamination", 0.0)

        with pytest.raises(ConfigurationError, match="contamination"):
            _validate_hyperparameters(config)

    def test_raises_for_contamination_at_upper_bound(self):
        """ConfigurationError is raised when contamination == 0.5 (out of range)."""
        config = _make_config()
        object.__setattr__(config, "contamination", 0.5)

        with pytest.raises(ConfigurationError, match="contamination"):
            _validate_hyperparameters(config)

    def test_raises_for_contamination_above_upper_bound(self):
        """ConfigurationError is raised when contamination > 0.5."""
        config = _make_config()
        object.__setattr__(config, "contamination", 0.9)

        with pytest.raises(ConfigurationError, match="contamination"):
            _validate_hyperparameters(config)

    def test_raises_for_contamination_negative(self):
        """ConfigurationError is raised when contamination is negative."""
        config = _make_config()
        object.__setattr__(config, "contamination", -0.1)

        with pytest.raises(ConfigurationError, match="contamination"):
            _validate_hyperparameters(config)

    def test_raises_for_contamination_wrong_type(self):
        """ConfigurationError is raised when contamination is not a float."""
        config = _make_config()
        object.__setattr__(config, "contamination", "0.1")

        with pytest.raises(ConfigurationError, match="contamination"):
            _validate_hyperparameters(config)

    def test_raises_for_n_estimators_zero(self):
        """ConfigurationError is raised when n_estimators == 0."""
        config = _make_config()
        object.__setattr__(config, "n_estimators", 0)

        with pytest.raises(ConfigurationError, match="n_estimators"):
            _validate_hyperparameters(config)

    def test_raises_for_n_estimators_negative(self):
        """ConfigurationError is raised when n_estimators is negative."""
        config = _make_config()
        object.__setattr__(config, "n_estimators", -10)

        with pytest.raises(ConfigurationError, match="n_estimators"):
            _validate_hyperparameters(config)

    def test_raises_for_n_estimators_wrong_type(self):
        """ConfigurationError is raised when n_estimators is not an int."""
        config = _make_config()
        object.__setattr__(config, "n_estimators", 100.0)

        with pytest.raises(ConfigurationError, match="n_estimators"):
            _validate_hyperparameters(config)

    def test_raises_for_random_state_wrong_type(self):
        """ConfigurationError is raised when random_state is not an int."""
        config = _make_config()
        object.__setattr__(config, "random_state", "42")

        with pytest.raises(ConfigurationError, match="random_state"):
            _validate_hyperparameters(config)

    def test_valid_hyperparameters_do_not_raise(self):
        """No exception is raised for valid hyperparameter values."""
        config = _make_config()
        # Should not raise
        _validate_hyperparameters(config)

    def test_train_model_raises_configuration_error_for_invalid_contamination(
        self, mock_boto3_client
    ):
        """
        train_model itself raises ConfigurationError (not just _validate_hyperparameters)
        when contamination is invalid, before any S3 call is made.
        """
        config = _make_config()
        object.__setattr__(config, "contamination", 0.0)
        df = _make_processed_df()

        with pytest.raises(ConfigurationError):
            train_model(df, config)

        # S3 should never be called when validation fails
        mock_boto3_client.put_object.assert_not_called()

    def test_train_model_raises_configuration_error_for_invalid_n_estimators(
        self, mock_boto3_client
    ):
        """
        train_model raises ConfigurationError when n_estimators is invalid,
        before any S3 call is made.
        """
        config = _make_config()
        object.__setattr__(config, "n_estimators", -1)
        df = _make_processed_df()

        with pytest.raises(ConfigurationError):
            train_model(df, config)

        mock_boto3_client.put_object.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Training parameters logged at INFO level  (Requirement 4.5)
# ---------------------------------------------------------------------------

class TestTrainingParametersLoggedAtInfo:
    """
    Requirement 4.5 — train_model must log training parameters at INFO level.
    """

    def test_contamination_logged_at_info(self, mock_boto3_client, caplog):
        """contamination value is present in an INFO-level log message."""
        config = _make_config(contamination=0.1)
        df = _make_processed_df()

        with caplog.at_level(logging.INFO):
            train_model(df, config)

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("0.1" in msg or "0.1000" in msg for msg in info_messages), (
            f"Expected contamination value in INFO logs, got: {info_messages}"
        )

    def test_n_estimators_logged_at_info(self, mock_boto3_client, caplog):
        """n_estimators value is present in an INFO-level log message."""
        config = _make_config(n_estimators=100)
        df = _make_processed_df()

        with caplog.at_level(logging.INFO):
            train_model(df, config)

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("100" in msg for msg in info_messages), (
            f"Expected n_estimators value in INFO logs, got: {info_messages}"
        )

    def test_random_state_logged_at_info(self, mock_boto3_client, caplog):
        """random_state value is present in an INFO-level log message."""
        config = _make_config(random_state=42)
        df = _make_processed_df()

        with caplog.at_level(logging.INFO):
            train_model(df, config)

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("42" in msg for msg in info_messages), (
            f"Expected random_state value in INFO logs, got: {info_messages}"
        )

    def test_s3_key_logged_at_info(self, mock_boto3_client, caplog):
        """The resulting S3 key is logged at INFO level after training completes."""
        config = _make_config()
        df = _make_processed_df()

        with caplog.at_level(logging.INFO):
            result = train_model(df, config)

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any(result.s3_key in msg for msg in info_messages), (
            f"Expected S3 key '{result.s3_key}' in INFO logs, got: {info_messages}"
        )

    def test_at_least_two_info_messages_emitted(self, mock_boto3_client, caplog):
        """
        train_model emits at least two INFO messages: one at start and one
        at completion (as specified in the design).
        """
        config = _make_config()
        df = _make_processed_df()

        with caplog.at_level(logging.INFO):
            train_model(df, config)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 2, (
            f"Expected at least 2 INFO log messages, got {len(info_records)}: "
            f"{[r.message for r in info_records]}"
        )
