"""Unit tests verifying least-privilege cloud IAM policies and backward compatibility."""

import json

from vipym.aws.iam import (
    VIPYM_LEAST_PRIVILEGE_POLICY as AWS_IAM_POLICY,
)
from vipym.aws.iam import (
    get_least_privilege_iam_json as aws_get_iam_json,
)
from vipym.cloud.iam import (
    VIPYM_LEAST_PRIVILEGE_POLICY,
    get_least_privilege_iam_json,
)


def test_backward_compatibility_reexports():
    """Verify that aws.iam and cloud.iam re-export identical policy and function objects."""
    assert VIPYM_LEAST_PRIVILEGE_POLICY is AWS_IAM_POLICY
    assert get_least_privilege_iam_json is aws_get_iam_json


def test_policy_json_validity_and_schema():
    """Verify generated IAM JSON string parses into valid IAM document."""
    json_str = get_least_privilege_iam_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)

    assert parsed["Version"] == "2012-10-17"
    assert "Statement" in parsed
    assert len(parsed["Statement"]) >= 3


def test_least_privilege_security_boundaries():
    """Ensure least-privilege statements enforce scoped access."""
    statements = {s["Sid"]: s for s in VIPYM_LEAST_PRIVILEGE_POLICY["Statement"]}

    # 1. S3 Artifact access must be restricted to vipym-* buckets
    s3_stmt = statements["ViPymS3ArtifactAccess"]
    assert s3_stmt["Effect"] == "Allow"
    assert "arn:aws:s3:::vipym-*" in s3_stmt["Resource"]
    assert "arn:aws:s3:::vipym-*/*" in s3_stmt["Resource"]
    assert "s3:GetObject" in s3_stmt["Action"]
    assert "s3:PutObject" in s3_stmt["Action"]

    # 2. CloudWatch metrics and logs
    cw_stmt = statements["ViPymCloudWatchMetrics"]
    assert cw_stmt["Effect"] == "Allow"
    assert "cloudwatch:PutMetricData" in cw_stmt["Action"]
    assert "logs:PutLogEvents" in cw_stmt["Action"]

    # 3. EC2 Self-termination allows terminating ephemeral compute nodes
    ec2_stmt = statements["ViPymEC2SelfTermination"]
    assert ec2_stmt["Effect"] == "Allow"
    assert "ec2:TerminateInstances" in ec2_stmt["Action"]
