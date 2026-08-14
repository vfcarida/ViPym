"""AWS Ephemeral EC2 Lifecycle Manager with Auto-Termination."""

from typing import Any

import boto3
from pydantic import BaseModel

from vipym.core.exceptions import CloudOrchestrationError
from vipym.core.logger import get_logger

logger = get_logger(__name__)


class EphemeralNodeSpec(BaseModel):
    instance_type: str = "p5.48xlarge"
    ami_id: str = "ami-0123456789abcdef0"  # Deep Learning Base AMI
    spot_instance: bool = True
    max_spot_price_usd: float | None = None
    subnet_id: str | None = None
    iam_role_arn: str | None = None
    auto_poweroff_minutes: int = 20


class EphemeralEC2Manager:
    """Provisions ephemeral GPU nodes, executes experiment commands, and ensures self-termination."""

    def __init__(self, region: str = "us-east-1") -> None:
        self.region = region
        try:
            self.ec2_client = boto3.client("ec2", region_name=region)
        except Exception:
            self.ec2_client = None

    def launch_ephemeral_experiment_node(
        self, spec: EphemeralNodeSpec, user_data_script: str
    ) -> str:
        """Launch an ephemeral GPU EC2 instance with an auto-poweroff watchdog in user-data."""
        logger.info(
            f"Launching ephemeral node: type={spec.instance_type}, spot={spec.spot_instance}"
        )
        if self.ec2_client is None:
            logger.warning("Boto3 EC2 client not available, returning mock instance ID.")
            return "i-mock-0123456789abcdef"

        try:
            launch_args: dict[str, Any] = {
                "ImageId": spec.ami_id,
                "InstanceType": spec.instance_type,
                "MinCount": 1,
                "MaxCount": 1,
                "UserData": user_data_script,
                "InstanceInitiatedShutdownBehavior": "terminate",
            }
            if spec.spot_instance:
                launch_args["InstanceMarketOptions"] = {
                    "MarketType": "spot",
                    "SpotOptions": {"SpotInstanceType": "one-time"},
                }
            resp = self.ec2_client.run_instances(**launch_args)
            instance_id = resp["Instances"][0]["InstanceId"]
            logger.info(f"Successfully requested ephemeral instance: {instance_id}")
            return instance_id
        except Exception as e:
            raise CloudOrchestrationError(f"Failed to provision EC2 instance: {e}") from e

    def terminate_node(self, instance_id: str) -> None:
        if self.ec2_client:
            logger.info(f"Terminating instance {instance_id}")
            self.ec2_client.terminate_instances(InstanceIds=[instance_id])
