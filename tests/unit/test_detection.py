"""
Unit tests for detection.py.

Requirements: 5.1, 5.3, 5.5
"""
from __future__ import annotations

import io
from datetime import datetime
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from src.detection import detect_anomalies, load_latest_model
from src.utils import PipelineConfig

import utils as _utils
DetectionResult = _utils.DetectionResult


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
    """Return a valid preprocessed DataFrame with timestamp and five metric columns."""
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


def _make_fitted_model() -> IsolationForest:
    """Return a real fitted IsolationForest for use in tests."""
    df = _make_processed_df()
    feature_cols = [c for c in df.columns if c != "timestamp"]
    model = IsolationForest(n_estimators=10, random_state=42)
    model.fit(df[feature_cols].values)
    return model


def _serialize_model(model: IsolationForest) -> bytes:
    """Serialize a model to bytes using joblib."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_s3_client():
    """
    Patches boto3.client inside detection so no real AWS calls are made.

    Yields a mock S3 client for per-test inspection.
    """
    mock_s3 = MagicMock()

    def _client_factory(service, **kwargs):
        if service == "s3":
            return mock_s3
        raise ValueError(f"Unexpected boto3 service: {service}")

    with patch("detection.boto3.client", side_effect=_client_factory):
        yield mock_s3


# ---------------------------------------------------------------------------
# Test: load_latest_model returns None when no model objects exist  (Req 5.1)
# ---------------------------------------------------------------------------

class TestLoadLatestModelReturnsNoneWhenEmpty:
    """
    Requirement 5.1 — load_latest_model must return None when no objects
    exist under the models/ prefix in S3.
    """

    def test_returns_none_when_contents_key_absent(self, mock_s3_client):
        """Returns None when list_objects_v2 response has no 'Contents' key."""
        mock_s3_client.list_objects_v2.return_value = {}

        result = load_latest_model(_make_config())

        assert result is None

    def test_returns_none_when_contents_is_empty_list(self, mock_s3_client):
        """Returns None when list_objects_v2 returns an empty Contents list."""
        mock_s3_client.list_objects_v2.return_value = {"Contents": []}

        result = load_latest_model(_make_config())

        assert result is None

    def test_list_objects_called_with_models_prefix(self, mock_s3_client):
        """list_objects_v2 is called with the 'models/' prefix."""
        mock_s3_client.list_objects_v2.return_value = {}

        load_latest_model(_make_config())

        mock_s3_client.list_objects_v2.assert_called_once()
        _, kwargs = mock_s3_client.list_objects_v2.call_args
        assert kwargs["Prefix"] == "models/"

    def test_list_objects_called_with_correct_bucket(self, mock_s3_client):
        """list_objects_v2 is called with the bucket from config."""
        mock_s3_client.list_objects_v2.return_value = {}

        load_latest_model(_make_config(s3_bucket="my-bucket"))

        _, kwargs = mock_s3_client.list_objects_v2.call_args
        assert kwargs["Bucket"] == "my-bucket"

    def test_get_object_not_called_when_no_models(self, mock_s3_client):
        """get_object is never called when there are no model objects."""
        mock_s3_client.list_objects_v2.return_value = {}

        load_latest_model(_make_config())

        mock_s3_client.get_object.assert_not_called()


# ---------------------------------------------------------------------------
# Test: load_latest_model returns the most recently modified model  (Req 5.1)
# ---------------------------------------------------------------------------

class TestLoadLatestModelReturnsNewest:
    """
    Requirement 5.1 — load_latest_model must return the IsolationForest
    deserialized from the object with the most recent LastModified timestamp.
    """

    def _make_list_response(self, keys_and_times: list[tuple[str, datetime]]) -> dict:
        """Build a list_objects_v2 response from (key, LastModified) pairs."""
        return {
            "Contents": [
                {"Key": key, "LastModified": ts}
                for key, ts in keys_and_times
            ]
        }

    def test_returns_model_from_most_recent_object(self, mock_s3_client):
        """
        When multiple model objects exist, the model from the object with
        the most recent LastModified is returned.
        """
        old_model = IsolationForest(n_estimators=5, random_state=1)
        new_model = IsolationForest(n_estimators=20, random_state=99)

        df = _make_processed_df()
        feature_cols = [c for c in df.columns if c != "timestamp"]
        old_model.fit(df[feature_cols].values)
        new_model.fit(df[feature_cols].values)

        old_bytes = _serialize_model(old_model)
        new_bytes = _serialize_model(new_model)

        mock_s3_client.list_objects_v2.return_value = self._make_list_response(
            [
                ("models/old_model.joblib", datetime(2024, 1, 1, 10, 0, 0)),
                ("models/new_model.joblib", datetime(2024, 1, 1, 12, 0, 0)),
                ("models/mid_model.joblib", datetime(2024, 1, 1, 11, 0, 0)),
            ]
        )

        def _get_object(Bucket, Key):
            if Key == "models/new_model.joblib":
                return {"Body": io.BytesIO(new_bytes)}
            return {"Body": io.BytesIO(old_bytes)}

        mock_s3_client.get_object.side_effect = _get_object

        result = load_latest_model(_make_config())

        # Verify get_object was called with the newest key
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="models/new_model.joblib"
        )
        assert isinstance(result, IsolationForest)

    def test_returns_model_when_only_one_object_exists(self, mock_s3_client):
        """When exactly one model object exists, it is returned."""
        model = _make_fitted_model()
        model_bytes = _serialize_model(model)

        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "models/only_model.joblib", "LastModified": datetime(2024, 1, 1, 12, 0, 0)}
            ]
        }
        mock_s3_client.get_object.return_value = {"Body": io.BytesIO(model_bytes)}

        result = load_latest_model(_make_config())

        assert isinstance(result, IsolationForest)
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="models/only_model.joblib"
        )

    def test_latest_model_is_deserialized_correctly(self, mock_s3_client):
        """The returned model is a properly deserialized IsolationForest."""
        model = _make_fitted_model()
        model_bytes = _serialize_model(model)

        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "models/model.joblib", "LastModified": datetime(2024, 1, 1, 12, 0, 0)}
            ]
        }
        mock_s3_client.get_object.return_value = {"Body": io.BytesIO(model_bytes)}

        result = load_latest_model(_make_config())

        assert isinstance(result, IsolationForest)
        # Verify the model can make predictions (i.e., it is properly fitted)
        df = _make_processed_df()
        feature_cols = [c for c in df.columns if c != "timestamp"]
        predictions = result.predict(df[feature_cols].values)
        assert set(predictions).issubset({-1, 1})

    def test_selects_latest_when_timestamps_differ_by_seconds(self, mock_s3_client):
        """Correctly selects the latest model even when timestamps differ by seconds."""
        model = _make_fitted_model()
        model_bytes = _serialize_model(model)

        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "models/model_a.joblib", "LastModified": datetime(2024, 1, 1, 12, 0, 0)},
                {"Key": "models/model_b.joblib", "LastModified": datetime(2024, 1, 1, 12, 0, 1)},
            ]
        }
        mock_s3_client.get_object.return_value = {"Body": io.BytesIO(model_bytes)}

        load_latest_model(_make_config())

        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="models/model_b.joblib"
        )


# ---------------------------------------------------------------------------
# Test: detect_anomalies writes results CSV with results/ prefix  (Req 5.3, 5.5)
# ---------------------------------------------------------------------------

class TestDetectAnomaliesResultsPrefix:
    """
    Requirement 5.3 — the S3 key used in put_object must start with 'results/'.
    Requirement 5.5 — detect_anomalies returns a DetectionResult.
    """

    def test_s3_key_starts_with_results_prefix(self, mock_s3_client):
        """The S3 key passed to put_object starts with 'results/'."""
        config = _make_config()
        df = _make_processed_df()
        model = _make_fitted_model()

        detect_anomalies(df, model, config)

        mock_s3_client.put_object.assert_called_once()
        _, kwargs = mock_s3_client.put_object.call_args
        assert kwargs["Key"].startswith("results/"), (
            f"Expected S3 key to start with 'results/' but got: {kwargs['Key']}"
        )

    def test_s3_key_in_detection_result_starts_with_results_prefix(self, mock_s3_client):
        """The s3_key field of the returned DetectionResult starts with 'results/'."""
        config = _make_config()
        df = _make_processed_df()
        model = _make_fitted_model()

        result = detect_anomalies(df, model, config)

        assert result.s3_key.startswith("results/"), (
            f"DetectionResult.s3_key should start with 'results/' but got: {result.s3_key}"
        )

    def test_s3_key_ends_with_csv_extension(self, mock_s3_client):
        """The S3 key ends with '.csv'."""
        config = _make_config()
        df = _make_processed_df()
        model = _make_fitted_model()

        detect_anomalies(df, model, config)

        _, kwargs = mock_s3_client.put_object.call_args
        assert kwargs["Key"].endswith(".csv"), (
            f"Expected S3 key to end with '.csv' but got: {kwargs['Key']}"
        )

    def test_s3_bucket_matches_config(self, mock_s3_client):
        """The Bucket argument to put_object matches config.s3_bucket."""
        config = _make_config(s3_bucket="my-results-bucket")
        df = _make_processed_df()
        model = _make_fitted_model()

        detect_anomalies(df, model, config)

        _, kwargs = mock_s3_client.put_object.call_args
        assert kwargs["Bucket"] == "my-results-bucket"

    def test_put_object_called_exactly_once(self, mock_s3_client):
        """put_object is called exactly once per detect_anomalies invocation."""
        config = _make_config()
        df = _make_processed_df()
        model = _make_fitted_model()

        detect_anomalies(df, model, config)

        assert mock_s3_client.put_object.call_count == 1

    def test_returns_detection_result_instance(self, mock_s3_client):
        """detect_anomalies returns a DetectionResult."""
        config = _make_config()
        df = _make_processed_df()
        model = _make_fitted_model()

        result = detect_anomalies(df, model, config)

        assert isinstance(result, DetectionResult)

    def test_result_dataframe_has_anomaly_label_column(self, mock_s3_client):
        """The result DataFrame contains an 'anomaly_label' column."""
        config = _make_config()
        df = _make_processed_df()
        model = _make_fitted_model()

        result = detect_anomalies(df, model, config)

        assert "anomaly_label" in result.dataframe.columns

    def test_anomaly_count_matches_label_column(self, mock_s3_client):
        """anomaly_count equals the number of -1 labels in the result DataFrame."""
        import numpy as np

        config = _make_config()
        df = _make_processed_df()

        # Use a mock model that returns known labels as a numpy array
        # (detection.py does `(labels == -1).sum()` which requires numpy array)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-1, 1, 1, -1, 1])

        result = detect_anomalies(df, mock_model, config)

        expected_count = 2
        assert result.anomaly_count == expected_count, (
            f"Expected anomaly_count={expected_count}, got {result.anomaly_count}"
        )

    def test_s3_key_contains_instance_id(self, mock_s3_client):
        """The S3 key contains the instance_id for artifact traceability."""
        instance_id = "i-1234567890abcdef0"
        config = _make_config(instance_id=instance_id)
        df = _make_processed_df()
        model = _make_fitted_model()

        result = detect_anomalies(df, model, config)

        assert instance_id in result.s3_key, (
            f"Expected instance_id '{instance_id}' in S3 key '{result.s3_key}'"
        )

    def test_original_dataframe_not_mutated(self, mock_s3_client):
        """detect_anomalies does not mutate the input DataFrame."""
        config = _make_config()
        df = _make_processed_df()
        original_columns = list(df.columns)
        model = _make_fitted_model()

        detect_anomalies(df, model, config)

        assert list(df.columns) == original_columns, (
            "Input DataFrame should not be mutated by detect_anomalies"
        )
