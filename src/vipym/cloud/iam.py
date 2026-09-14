"""Least-privilege IAM and cloud security policies for ViPym deployments."""

from __future__ import annotations

import json
from typing import Any

VIPYM_LEAST_PRIVILEGE_POLICY: dict[str, Any] = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ViPymS3ArtifactAccess",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
            "Resource": ["arn:aws:s3:::vipym-*", "arn:aws:s3:::vipym-*/*"],
        },
        {
            "Sid": "ViPymCloudWatchMetrics",
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Resource": "*",
        },
        {
            "Sid": "ViPymEC2SelfTermination",
            "Effect": "Allow",
            "Action": ["ec2:TerminateInstances", "ec2:DescribeInstances"],
            "Resource": "*",
        },
    ],
}


def get_least_privilege_iam_json() -> str:
    """Return formatted JSON string of the least-privilege IAM policy."""
    return json.dumps(VIPYM_LEAST_PRIVILEGE_POLICY, indent=2)
