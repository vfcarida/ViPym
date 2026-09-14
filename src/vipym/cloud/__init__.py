"""Cloud orchestration and infrastructure package."""

from vipym.cloud.cloudwatch import CloudWatchTelemetryEmitter
from vipym.cloud.ec2_ephemeral import EphemeralEC2Manager, EphemeralNodeSpec
from vipym.cloud.iam import VIPYM_LEAST_PRIVILEGE_POLICY, get_least_privilege_iam_json
from vipym.cloud.s3 import S3ArtifactStore

__all__ = [
    "CloudWatchTelemetryEmitter",
    "EphemeralEC2Manager",
    "EphemeralNodeSpec",
    "S3ArtifactStore",
    "VIPYM_LEAST_PRIVILEGE_POLICY",
    "get_least_privilege_iam_json",
]
