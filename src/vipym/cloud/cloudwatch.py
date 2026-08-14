"""CloudWatch structured logging and metric reporting."""

from typing import Dict, List, Optional
import boto3
from vipym.core.logger import get_logger

logger = get_logger(__name__)


class CloudWatchTelemetryEmitter:
    """Streams live metrics and status alarms to AWS CloudWatch."""

    def __init__(self, namespace: str = "ViPym/CompressionBenchmarks", region: str = "us-east-1") -> None:
        self.namespace = namespace
        try:
            self.cw_client = boto3.client("cloudwatch", region_name=region)
        except Exception:
            self.cw_client = None

    def emit_metric(self, metric_name: str, value: float, unit: str = "Count", dimensions: Optional[Dict[str, str]] = None) -> None:
        if self.cw_client is None:
            return
        dims = [{"Name": k, "Value": v} for k, v in (dimensions or {}).items()]
        try:
            self.cw_client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Value": value,
                        "Unit": unit,
                        "Dimensions": dims,
                    }
                ],
            )
        except Exception as e:
            logger.warning(f"Failed to put metric to CloudWatch: {e}")
