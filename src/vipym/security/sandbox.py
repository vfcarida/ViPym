"""Sandboxed Untrusted Code Execution Engine."""

from vipym.core.exceptions import SandboxUnavailableError
from vipym.evaluation.sandbox.docker_sandbox import (
    ExecutionResult,
    SandboxedCodeRunner,
    is_docker_available,
)
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig
from vipym.security.sanitizer import sanitize_execution_environment
from vipym.security.threat_model import get_threat_model_summary

__all__ = [
    "ExecutionResult",
    "SandboxSecurityConfig",
    "SandboxUnavailableError",
    "SandboxedCodeRunner",
    "get_threat_model_summary",
    "is_docker_available",
    "sanitize_execution_environment",
]
