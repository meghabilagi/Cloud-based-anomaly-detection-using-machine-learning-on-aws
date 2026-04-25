"""
Pipeline orchestrator and AWS Lambda entry point.

Coordinates all pipeline stages: data collection, preprocessing,
model training, anomaly detection, visualization, and alerting.
"""
from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any, Dict

import boto3

from data_collection import collect_metrics
from detection import detect_anomalies, load_latest_model
from model import train_model
from preprocessing import preprocess
from utils import (
    PipelineConfig,
    PipelineResult,
    get_logger,
    load_env_var,
    retry_with_backoff,
)
from visualization import generate_plots

logger = get_logger(__name__)

# CPU threshold above which an alert is sent regardless of anomaly labels
_CPU_ALERT_THRESHOLD = 90.0


def _send_alert(results_df, detection_result, config: PipelineConfig) -> None:
    """
    Publish an SNS alert if anomalies were detected OR if any CPUUtilization
    value exceeds the threshold.

    Args:
        results_df: DataFrame with anomaly_label and metric columns.
        detection_result: DetectionResult from the detection stage.
        config: Immutable pipeline configuration.
    """
    anomaly_count = detection_result.anomaly_count
    cpu_exceeded = bool(results_df["CPUUtilization"].max() > _CPU_ALERT_THRESHOLD)

    if anomaly_count == 0 and not cpu_exceeded:
        logger.info(
            "Alerting: no anomalies detected and CPU threshold not exceeded — "
            "skipping SNS publish."
        )
        return

    logger.info(
        "Alerting started: anomaly_count=%d, cpu_threshold_exceeded=%s",
        anomaly_count,
        cpu_exceeded,
    )

    # Build the alert message
    anomaly_rows = results_df[results_df["anomaly_label"] == -1]
    anomaly_timestamps = anomaly_rows["timestamp"].tolist()
    anomaly_metrics = anomaly_rows.drop(columns=["anomaly_label"]).to_dict(orient="records")

    message_lines = [
        f"AWS Anomaly Detection Alert",
        f"",
        f"Instance ID: {config.instance_id}",
        f"Total anomalies detected: {anomaly_count}",
        f"CPU threshold (>{_CPU_ALERT_THRESHOLD}%) exceeded: {cpu_exceeded}",
        f"",
        f"Anomalous timestamps:",
    ]
    for ts in anomaly_timestamps:
        message_lines.append(f"  - {ts}")

    message_lines.append("")
    message_lines.append("Metric values at anomalous rows:")
    for row in anomaly_metrics:
        message_lines.append(f"  {row}")

    message = "\n".join(message_lines)

    sns_client = boto3.client("sns", region_name=config.aws_region)

    def _publish():
        return sns_client.publish(
            TopicArn=config.sns_topic_arn,
            Subject=f"Anomaly Alert — Instance {config.instance_id}",
            Message=message,
        )

    retry_with_backoff(_publish)

    logger.info(
        "Alerting complete: SNS message published to %s",
        config.sns_topic_arn,
    )


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """
    Execute all pipeline stages in order and return a PipelineResult.

    Stages:
        1. collect_metrics
        2. preprocess
        3. load_latest_model (train_model if no model found)
        4. detect_anomalies
        5. generate_plots
        6. _send_alert

    Args:
        config: Immutable pipeline configuration.

    Returns:
        PipelineResult with status="success", anomaly_count, and stage_artifacts.

    Raises:
        Any exception raised by a stage after logging it at ERROR level.
    """
    stage_artifacts: Dict[str, Any] = {}

    try:
        # Stage 1: Data collection
        collection_result = collect_metrics(config)
        stage_artifacts["raw"] = collection_result.s3_key

        # Stage 2: Preprocessing
        preprocess_result = preprocess(collection_result.dataframe, config)
        stage_artifacts["processed"] = preprocess_result.s3_key

        # Stage 3: Model — load existing or train a new one
        model = load_latest_model(config)
        if model is None:
            training_result = train_model(preprocess_result.dataframe, config)
            stage_artifacts["model"] = training_result.s3_key
            model = training_result.model
        # (If a model was loaded from S3 there is no new model artifact key to record)

        # Stage 4: Anomaly detection
        detection_result = detect_anomalies(preprocess_result.dataframe, model, config)
        stage_artifacts["results"] = detection_result.s3_key

        # Stage 5: Visualization
        viz_result = generate_plots(detection_result.dataframe, config)
        stage_artifacts["plots"] = viz_result.s3_keys

        # Stage 6: Alerting
        _send_alert(detection_result.dataframe, detection_result, config)

    except Exception:
        logger.error(
            "Pipeline stage failed:\n%s",
            traceback.format_exc(),
        )
        raise

    return PipelineResult(
        status="success",
        anomaly_count=detection_result.anomaly_count,
        stage_artifacts=stage_artifacts,
    )


def lambda_handler(event: dict, context: object) -> dict:
    """
    AWS Lambda entry point.

    Loads configuration from environment variables, runs the pipeline,
    and returns a status dict.  Any exception raised by run_pipeline
    propagates out of this function unchanged.

    Args:
        event: The Lambda event payload (not used by the pipeline).
        context: The Lambda context object (not used by the pipeline).

    Returns:
        A dict with ``status``, ``anomaly_count``, and ``stage_artifacts``.
    """
    config = PipelineConfig(
        s3_bucket=load_env_var("S3_BUCKET", str),
        sns_topic_arn=load_env_var("SNS_TOPIC_ARN", str),
        instance_id=load_env_var("INSTANCE_ID", str),
        contamination=load_env_var("CONTAMINATION", float),
        n_estimators=load_env_var("N_ESTIMATORS", int),
        random_state=load_env_var("RANDOM_STATE", int),
        aws_region=load_env_var("AWS_DEFAULT_REGION", str),
        run_timestamp=datetime.utcnow(),
    )

    result = run_pipeline(config)

    return {
        "status": result.status,
        "anomaly_count": result.anomaly_count,
        "stage_artifacts": result.stage_artifacts,
    }
