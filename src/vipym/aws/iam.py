"""Least-privilege IAM policies for ViPym AWS deployments (backward-compatible facade)."""

from vipym.cloud.iam import VIPYM_LEAST_PRIVILEGE_POLICY, get_least_privilege_iam_json

__all__ = ["VIPYM_LEAST_PRIVILEGE_POLICY", "get_least_privilege_iam_json"]
