"""
Shared utilities, configuration dataclasses, and custom exceptions
for the AWS Anomaly Detection Pipeline.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd


# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Base exception for all pipeline errors."""


class ConfigurationError(PipelineError):
    """Raised when a required environment variable is missing or has an invalid value."""


class InsufficientDataError(PipelineError):
    """Raised when there are too few data points to proceed with training or inference."""


class ModelNotFoundError(PipelineError):
    """Raised when no trained model can be found in the S3 bucket."""


class RetryExhaustedError(PipelineError):
    """Raised when all retry attempts for an AWS API call have been exhausted."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger that writes to sys.stdout at INFO level.

    Uses a StreamHandler directed at sys.stdout so that AWS Lambda captures
    log output in Amazon CloudWatch Logs.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Environment variable loading
# ---------------------------------------------------------------------------

def load_env_var(name: str, cast: Type = str) -> Any:
    """
    Reads an environment variable and casts it to the given type.

    Raises ConfigurationError with a descriptive message if the variable is
    missing or if the value cannot be cast to the requested type.
    """
    raw = os.environ.get(name)
    if raw is None:
        raise ConfigurationError(
            f"Required environment variable '{name}' is not set. "
            f"Please set '{name}' before running the pipeline."
        )
    try:
        return cast(raw)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(
            f"Environment variable '{name}' has an invalid value '{raw}'. "
            f"Expected a value castable to {cast.__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# S3 key generation
# ---------------------------------------------------------------------------

def get_s3_key(
    prefix: str,
    name: str,
    instance_id: str,
    timestamp: datetime,
    extension: str,
) -> str:
    """
    Generates a consistent S3 object key.

    Format: ``{prefix}/{name}_{instance_id}_{ts}.{extension}``

    Example:
        get_s3_key("raw", "metrics", "i-1234", dt, "csv")
        -> "raw/metrics_i-1234_20240101T120000.csv"
    """
    ts = timestamp.strftime("%Y%m%dT%H%M%S")
    return f"{prefix}/{name}_{instance_id}_{ts}.{extension}"


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------

# AWS ClientError codes that indicate a transient, retryable condition.
_RETRYABLE_ERROR_CODES = frozenset({"ThrottlingException", "ServiceUnavailable"})

# Exception types that represent network-level timeouts and are retryable.
_RETRYABLE_EXCEPTION_TYPES = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def _is_retryable(exc: Exception) -> bool:
    """Return True if *exc* is a retryable AWS or network error."""
    # Check for botocore/boto3 ClientError with a retryable error code.
    try:
        from botocore.exceptions import ClientError, EndpointResolutionError  # type: ignore
        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "")
            return code in _RETRYABLE_ERROR_CODES
        if isinstance(exc, EndpointResolutionError):
            return True
    except ImportError:
        pass

    # Check for network-level timeouts.
    if isinstance(exc, _RETRYABLE_EXCEPTION_TYPES):
        return True

    return False


def retry_with_backoff(
    func: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    multiplier: float = 2.0,
) -> Any:
    """
    Calls *func()*, retrying on retryable AWS exceptions with exponential backoff.

    Retryable conditions:
    - ``botocore.exceptions.ClientError`` with error code ``ThrottlingException``
      or ``ServiceUnavailable``
    - ``botocore.exceptions.EndpointResolutionError`` (network-level failures)
    - Network timeout exceptions (``ConnectionError``, ``TimeoutError``, ``OSError``)

    Non-retryable conditions (raised immediately without retry):
    - ``ClientError`` with codes such as ``AccessDenied``, ``NoSuchBucket``, etc.

    Args:
        func: A zero-argument callable to invoke.
        max_attempts: Maximum number of total attempts (default 3).
        base_delay: Initial delay in seconds before the first retry (default 1.0).
        multiplier: Factor by which the delay grows on each retry (default 2.0).

    Returns:
        The return value of *func* on success.

    Raises:
        The last exception raised by *func* after all attempts are exhausted.
    """
    last_exc: Optional[Exception] = None
    delay = base_delay

    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if not _is_retryable(exc):
                raise
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(delay)
                delay *= multiplier

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """
    Immutable configuration object populated from Lambda environment variables.

    All fields are read at pipeline startup via ``load_env_var``.
    """
    s3_bucket: str
    sns_topic_arn: str
    instance_id: str
    contamination: float
    n_estimators: int
    random_state: int
    aws_region: str
    run_timestamp: datetime


@dataclass
class CollectionResult:
    """Result produced by the data collection stage."""
    s3_key: str
    dataframe: pd.DataFrame


@dataclass
class PreprocessResult:
    """Result produced by the preprocessing stage."""
    s3_key: str
    dataframe: pd.DataFrame


@dataclass
class TrainingResult:
    """Result produced by the model training stage."""
    s3_key: str
    model: Any  # sklearn.ensemble.IsolationForest — typed as Any to avoid hard import


@dataclass
class DetectionResult:
    """Result produced by the anomaly detection stage."""
    s3_key: str
    dataframe: pd.DataFrame
    anomaly_count: int


@dataclass
class VisualizationResult:
    """Result produced by the visualization stage."""
    s3_keys: List[str]


@dataclass
class PipelineResult:
    """Summary result returned by the full pipeline run."""
    status: str                      # "success" | "failure"
    anomaly_count: int
    stage_artifacts: Dict[str, str]  # stage_name -> s3_key
