"""
Visualization module.

Generates time-series plots with anomaly highlights and saves
them as PNG files to S3.
"""
import matplotlib
matplotlib.use('Agg')  # Must be set before any other matplotlib import for Lambda compatibility

import io
from typing import List

import boto3
import matplotlib.pyplot as plt
import pandas as pd

from utils import PipelineConfig, VisualizationResult, get_logger, get_s3_key

logger = get_logger(__name__)

# The five EC2 metrics to plot
_METRICS = [
    "CPUUtilization",
    "NetworkIn",
    "NetworkOut",
    "DiskReadBytes",
    "DiskWriteBytes",
]


def generate_plots(results_df: pd.DataFrame, config: PipelineConfig) -> VisualizationResult:
    """
    Generates one time-series PNG per metric.

    For each metric:
    - Plots all data points as a blue line.
    - Overlays anomaly points (anomaly_label == -1) as red scatter markers.
    - Labels the plot with the metric name, instance ID, and time range.
    - Saves the figure to an in-memory buffer and uploads PNG bytes to S3
      under the ``results/plots/`` prefix.
    - Closes the figure after saving to avoid memory leaks in Lambda.

    Args:
        results_df: DataFrame with columns ``timestamp``, the five metric columns,
                    and ``anomaly_label`` (-1 = anomaly, 1 = normal).
        config: Pipeline configuration (provides S3 bucket, instance ID, timestamp).

    Returns:
        VisualizationResult containing the list of five S3 keys.
    """
    logger.info(
        "Visualization stage started: generating plots for instance %s",
        config.instance_id,
    )

    s3_client = boto3.client("s3", region_name=config.aws_region)
    s3_keys: List[str] = []

    # Determine time range for plot labels
    timestamps = pd.to_datetime(results_df["timestamp"])
    time_start = timestamps.min()
    time_end = timestamps.max()
    time_range_label = f"{time_start} — {time_end}"

    anomaly_mask = results_df["anomaly_label"] == -1

    for metric in _METRICS:
        fig, ax = plt.subplots()

        try:
            # Plot all points as a blue line
            ax.plot(
                timestamps,
                results_df[metric],
                color="blue",
                linewidth=1,
                label=metric,
            )

            # Overlay anomaly points as red scatter markers
            if anomaly_mask.any():
                ax.scatter(
                    timestamps[anomaly_mask],
                    results_df[metric][anomaly_mask],
                    color="red",
                    zorder=5,
                    label="Anomaly",
                )

            # Labels and title
            ax.set_title(
                f"{metric} — Instance: {config.instance_id}\n{time_range_label}"
            )
            ax.set_xlabel("Timestamp")
            ax.set_ylabel(metric)
            ax.legend()

            # Save to in-memory buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            png_bytes = buf.read()

        finally:
            # Always close the figure to avoid memory leaks in Lambda
            plt.close(fig)

        # Build S3 key and upload
        key = get_s3_key(
            prefix="results/plots",
            name=metric,
            instance_id=config.instance_id,
            timestamp=config.run_timestamp,
            extension="png",
        )
        s3_client.put_object(
            Bucket=config.s3_bucket,
            Key=key,
            Body=png_bytes,
        )
        s3_keys.append(key)

    logger.info(
        "Visualization stage complete: %d plots uploaded for instance %s",
        len(s3_keys),
        config.instance_id,
    )

    return VisualizationResult(s3_keys=s3_keys)
