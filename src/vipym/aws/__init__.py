"""AWS Subpackage unifying EC2, S3, SageMaker, CloudWatch, and IAM."""

from vipym.aws.iam import get_least_privilege_iam_json
from vipym.cloud.cloudwatch import CloudWatchTelemetryEmitter
from vipym.cloud.ec2_ephemeral import EphemeralEC2Manager, EphemeralNodeSpec
from vipym.cloud.s3 import S3ArtifactStore

__all__ = [
    "CloudWatchTelemetryEmitter",
    "EphemeralEC2Manager",
    "EphemeralNodeSpec",
    "S3ArtifactStore",
    "get_least_privilege_iam_json",
]
