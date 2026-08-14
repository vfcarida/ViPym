"""Least-privilege IAM policies and security policies for ViPym AWS deployments."""

import json

VIPYM_LEAST_PRIVILEGE_POLICY = {
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
    """Return JSON string of least-privilege IAM policy."""
    return json.dumps(VIPYM_LEAST_PRIVILEGE_POLICY, indent=2)
