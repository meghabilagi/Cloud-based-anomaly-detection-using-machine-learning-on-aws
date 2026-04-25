"""
Unit tests for utils.load_env_var and utils.get_s3_key.

Requirements: 4.4, 9.3
"""
from __future__ import annotations

import os
from datetime import datetime

import pytest

from src.utils import ConfigurationError, load_env_var, get_s3_key


# ---------------------------------------------------------------------------
# load_env_var — missing variable
# ---------------------------------------------------------------------------

class TestLoadEnvVarMissing:
    """load_env_var raises ConfigurationError when the env var is not set."""

    def test_missing_s3_bucket(self, monkeypatch):
        monkeypatch.delenv("S3_BUCKET", raising=False)
        with pytest.raises(ConfigurationError, match="S3_BUCKET"):
            load_env_var("S3_BUCKET")

    def test_missing_sns_topic_arn(self, monkeypatch):
        monkeypatch.delenv("SNS_TOPIC_ARN", raising=False)
        with pytest.raises(ConfigurationError, match="SNS_TOPIC_ARN"):
            load_env_var("SNS_TOPIC_ARN")

    def test_missing_instance_id(self, monkeypatch):
        monkeypatch.delenv("INSTANCE_ID", raising=False)
        with pytest.raises(ConfigurationError, match="INSTANCE_ID"):
            load_env_var("INSTANCE_ID")

    def test_missing_contamination(self, monkeypatch):
        monkeypatch.delenv("CONTAMINATION", raising=False)
        with pytest.raises(ConfigurationError, match="CONTAMINATION"):
            load_env_var("CONTAMINATION", cast=float)

    def test_missing_n_estimators(self, monkeypatch):
        monkeypatch.delenv("N_ESTIMATORS", raising=False)
        with pytest.raises(ConfigurationError, match="N_ESTIMATORS"):
            load_env_var("N_ESTIMATORS", cast=int)

    def test_missing_random_state(self, monkeypatch):
        monkeypatch.delenv("RANDOM_STATE", raising=False)
        with pytest.raises(ConfigurationError, match="RANDOM_STATE"):
            load_env_var("RANDOM_STATE", cast=int)

    def test_missing_aws_region(self, monkeypatch):
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        with pytest.raises(ConfigurationError, match="AWS_DEFAULT_REGION"):
            load_env_var("AWS_DEFAULT_REGION")

    def test_error_message_mentions_variable_name(self, monkeypatch):
        monkeypatch.delenv("MY_CUSTOM_VAR", raising=False)
        with pytest.raises(ConfigurationError) as exc_info:
            load_env_var("MY_CUSTOM_VAR")
        assert "MY_CUSTOM_VAR" in str(exc_info.value)


# ---------------------------------------------------------------------------
# load_env_var — invalid cast
# ---------------------------------------------------------------------------

class TestLoadEnvVarInvalidCast:
    """load_env_var raises ConfigurationError when the value cannot be cast."""

    def test_non_float_contamination(self, monkeypatch):
        monkeypatch.setenv("CONTAMINATION", "not_a_float")
        with pytest.raises(ConfigurationError, match="CONTAMINATION"):
            load_env_var("CONTAMINATION", cast=float)

    def test_non_int_n_estimators(self, monkeypatch):
        monkeypatch.setenv("N_ESTIMATORS", "abc")
        with pytest.raises(ConfigurationError, match="N_ESTIMATORS"):
            load_env_var("N_ESTIMATORS", cast=int)

    def test_non_int_random_state(self, monkeypatch):
        monkeypatch.setenv("RANDOM_STATE", "3.14")
        with pytest.raises(ConfigurationError, match="RANDOM_STATE"):
            load_env_var("RANDOM_STATE", cast=int)

    def test_empty_string_for_float(self, monkeypatch):
        monkeypatch.setenv("CONTAMINATION", "")
        with pytest.raises(ConfigurationError, match="CONTAMINATION"):
            load_env_var("CONTAMINATION", cast=float)

    def test_empty_string_for_int(self, monkeypatch):
        monkeypatch.setenv("N_ESTIMATORS", "")
        with pytest.raises(ConfigurationError, match="N_ESTIMATORS"):
            load_env_var("N_ESTIMATORS", cast=int)

    def test_error_message_mentions_invalid_value(self, monkeypatch):
        monkeypatch.setenv("CONTAMINATION", "bad_value")
        with pytest.raises(ConfigurationError) as exc_info:
            load_env_var("CONTAMINATION", cast=float)
        assert "bad_value" in str(exc_info.value)

    def test_error_message_mentions_expected_type(self, monkeypatch):
        monkeypatch.setenv("CONTAMINATION", "bad_value")
        with pytest.raises(ConfigurationError) as exc_info:
            load_env_var("CONTAMINATION", cast=float)
        assert "float" in str(exc_info.value)


# ---------------------------------------------------------------------------
# load_env_var — successful reads
# ---------------------------------------------------------------------------

class TestLoadEnvVarSuccess:
    """load_env_var returns the correctly cast value when the env var is valid."""

    def test_string_default_cast(self, monkeypatch):
        monkeypatch.setenv("S3_BUCKET", "my-bucket")
        assert load_env_var("S3_BUCKET") == "my-bucket"

    def test_float_cast(self, monkeypatch):
        monkeypatch.setenv("CONTAMINATION", "0.1")
        result = load_env_var("CONTAMINATION", cast=float)
        assert result == pytest.approx(0.1)

    def test_int_cast(self, monkeypatch):
        monkeypatch.setenv("N_ESTIMATORS", "100")
        result = load_env_var("N_ESTIMATORS", cast=int)
        assert result == 100

    def test_int_cast_zero(self, monkeypatch):
        monkeypatch.setenv("RANDOM_STATE", "0")
        result = load_env_var("RANDOM_STATE", cast=int)
        assert result == 0


# ---------------------------------------------------------------------------
# get_s3_key — key format
# ---------------------------------------------------------------------------

class TestGetS3Key:
    """get_s3_key returns the expected S3 object key string."""

    def test_basic_key_format(self):
        dt = datetime(2024, 1, 1, 12, 0, 0)
        key = get_s3_key("raw", "metrics", "i-1234", dt, "csv")
        assert key == "raw/metrics_i-1234_20240101T120000.csv"

    def test_prefix_is_first_segment(self):
        dt = datetime(2024, 6, 15, 8, 30, 45)
        key = get_s3_key("processed", "data", "i-abcd", dt, "csv")
        assert key.startswith("processed/")

    def test_extension_is_last_segment(self):
        dt = datetime(2024, 3, 20, 0, 0, 0)
        key = get_s3_key("models", "model", "i-5678", dt, "joblib")
        assert key.endswith(".joblib")

    def test_timestamp_format_yyyymmddthhmmss(self):
        dt = datetime(2023, 12, 31, 23, 59, 59)
        key = get_s3_key("results", "output", "i-0001", dt, "csv")
        assert "20231231T235959" in key

    def test_instance_id_in_key(self):
        dt = datetime(2024, 1, 1, 0, 0, 0)
        instance_id = "i-0a1b2c3d4e5f"
        key = get_s3_key("raw", "metrics", instance_id, dt, "csv")
        assert instance_id in key

    def test_name_in_key(self):
        dt = datetime(2024, 1, 1, 0, 0, 0)
        key = get_s3_key("raw", "cpu_metrics", "i-1234", dt, "csv")
        assert "cpu_metrics" in key

    def test_results_plots_prefix(self):
        dt = datetime(2024, 5, 10, 14, 22, 0)
        key = get_s3_key("results/plots", "CPUUtilization", "i-9999", dt, "png")
        assert key.startswith("results/plots/")
        assert key.endswith(".png")

    def test_full_key_structure(self):
        dt = datetime(2024, 1, 1, 12, 0, 0)
        key = get_s3_key("raw", "metrics", "i-1234", dt, "csv")
        # Verify exact format: {prefix}/{name}_{instance_id}_{ts}.{ext}
        prefix, rest = key.split("/", 1)
        assert prefix == "raw"
        name_part, instance_part, ts_ext = rest.split("_", 2)
        assert name_part == "metrics"
        assert instance_part == "i-1234"
        ts, ext = ts_ext.rsplit(".", 1)
        assert ts == "20240101T120000"
        assert ext == "csv"

    def test_different_timestamps_produce_different_keys(self):
        dt1 = datetime(2024, 1, 1, 12, 0, 0)
        dt2 = datetime(2024, 1, 1, 13, 0, 0)
        key1 = get_s3_key("raw", "metrics", "i-1234", dt1, "csv")
        key2 = get_s3_key("raw", "metrics", "i-1234", dt2, "csv")
        assert key1 != key2

    def test_midnight_timestamp(self):
        dt = datetime(2024, 2, 29, 0, 0, 0)  # leap year
        key = get_s3_key("raw", "metrics", "i-1234", dt, "csv")
        assert "20240229T000000" in key


# ---------------------------------------------------------------------------
# retry_with_backoff — unit tests
# ---------------------------------------------------------------------------
# Requirements: 2.5, 7.4, 10.4, 10.5

from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from src.utils import retry_with_backoff


def _make_client_error(code: str) -> ClientError:
    """Helper: build a botocore ClientError with the given error code."""
    return ClientError(
        error_response={"Error": {"Code": code, "Message": f"Simulated {code}"}},
        operation_name="TestOperation",
    )


class TestRetryWithBackoffSuccess:
    """retry_with_backoff returns immediately when the function succeeds."""

    def test_success_on_first_attempt_called_once(self):
        """A function that succeeds immediately is called exactly once."""
        func = MagicMock(return_value="ok")
        with patch("time.sleep") as mock_sleep:
            result = retry_with_backoff(func, max_attempts=3)
        assert result == "ok"
        assert func.call_count == 1
        mock_sleep.assert_not_called()

    def test_return_value_is_propagated(self):
        """The return value from the wrapped function is returned unchanged."""
        func = MagicMock(return_value=42)
        with patch("time.sleep"):
            result = retry_with_backoff(func, max_attempts=3)
        assert result == 42


class TestRetryWithBackoffAlwaysFails:
    """retry_with_backoff exhausts all attempts when the function always raises."""

    def test_always_failing_called_max_attempts_times(self):
        """A function that always raises ThrottlingException is called max_attempts times."""
        func = MagicMock(side_effect=_make_client_error("ThrottlingException"))
        with patch("time.sleep"):
            with pytest.raises(ClientError):
                retry_with_backoff(func, max_attempts=3)
        assert func.call_count == 3

    def test_always_failing_raises_last_exception(self):
        """The exception raised after exhausting retries is the last one from the function."""
        exc = _make_client_error("ThrottlingException")
        func = MagicMock(side_effect=exc)
        with patch("time.sleep"):
            with pytest.raises(ClientError) as exc_info:
                retry_with_backoff(func, max_attempts=2)
        assert exc_info.value is exc

    def test_max_attempts_one_calls_once_then_raises(self):
        """With max_attempts=1, the function is called once and the exception propagates."""
        func = MagicMock(side_effect=_make_client_error("ServiceUnavailable"))
        with patch("time.sleep"):
            with pytest.raises(ClientError):
                retry_with_backoff(func, max_attempts=1)
        assert func.call_count == 1

    def test_max_attempts_five_calls_five_times(self):
        """With max_attempts=5, a always-failing function is called exactly 5 times."""
        func = MagicMock(side_effect=_make_client_error("ThrottlingException"))
        with patch("time.sleep"):
            with pytest.raises(ClientError):
                retry_with_backoff(func, max_attempts=5)
        assert func.call_count == 5

    def test_sleep_is_called_between_retries(self):
        """time.sleep is called (max_attempts - 1) times between retry attempts."""
        func = MagicMock(side_effect=_make_client_error("ThrottlingException"))
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ClientError):
                retry_with_backoff(func, max_attempts=3, base_delay=1.0)
        # sleep is called after each failed attempt except the last
        assert mock_sleep.call_count == 2


class TestRetryWithBackoffRetryableCodes:
    """retry_with_backoff retries on ThrottlingException and ServiceUnavailable."""

    def test_throttling_exception_triggers_retry(self):
        """ThrottlingException is retryable — function is retried."""
        func = MagicMock(side_effect=_make_client_error("ThrottlingException"))
        with patch("time.sleep"):
            with pytest.raises(ClientError):
                retry_with_backoff(func, max_attempts=3)
        assert func.call_count == 3

    def test_service_unavailable_triggers_retry(self):
        """ServiceUnavailable is retryable — function is retried."""
        func = MagicMock(side_effect=_make_client_error("ServiceUnavailable"))
        with patch("time.sleep"):
            with pytest.raises(ClientError):
                retry_with_backoff(func, max_attempts=3)
        assert func.call_count == 3

    def test_succeeds_after_transient_throttling(self):
        """Function succeeds on the third attempt after two ThrottlingExceptions."""
        responses = [
            _make_client_error("ThrottlingException"),
            _make_client_error("ThrottlingException"),
            "success",
        ]
        func = MagicMock(side_effect=responses)
        with patch("time.sleep"):
            result = retry_with_backoff(func, max_attempts=3)
        assert result == "success"
        assert func.call_count == 3


class TestRetryWithBackoffNonRetryableCodes:
    """retry_with_backoff raises immediately for non-retryable error codes."""

    def test_access_denied_raises_immediately(self):
        """AccessDenied is not retryable — raises on first attempt without retry."""
        func = MagicMock(side_effect=_make_client_error("AccessDenied"))
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ClientError) as exc_info:
                retry_with_backoff(func, max_attempts=3)
        assert func.call_count == 1
        mock_sleep.assert_not_called()
        assert exc_info.value.response["Error"]["Code"] == "AccessDenied"

    def test_no_such_bucket_raises_immediately(self):
        """NoSuchBucket is not retryable — raises on first attempt without retry."""
        func = MagicMock(side_effect=_make_client_error("NoSuchBucket"))
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ClientError) as exc_info:
                retry_with_backoff(func, max_attempts=3)
        assert func.call_count == 1
        mock_sleep.assert_not_called()
        assert exc_info.value.response["Error"]["Code"] == "NoSuchBucket"

    def test_invalid_parameter_raises_immediately(self):
        """An arbitrary non-retryable code raises immediately without retry."""
        func = MagicMock(side_effect=_make_client_error("InvalidParameterValue"))
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ClientError):
                retry_with_backoff(func, max_attempts=3)
        assert func.call_count == 1
        mock_sleep.assert_not_called()

    def test_non_retryable_does_not_sleep(self):
        """No sleep occurs when a non-retryable error is raised."""
        func = MagicMock(side_effect=_make_client_error("AccessDenied"))
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ClientError):
                retry_with_backoff(func, max_attempts=5)
        mock_sleep.assert_not_called()
