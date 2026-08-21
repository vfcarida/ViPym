"""Unit tests for sandboxed execution, Docker isolation, and environment sanitization."""

import logging
from unittest.mock import MagicMock

import pytest

from vipym.core.exceptions import SandboxUnavailableError
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig
from vipym.security.sandbox import SandboxedCodeRunner
from vipym.security.sanitizer import sanitize_execution_environment


def test_environment_sanitization():
    clean_env = sanitize_execution_environment()
    assert "AWS_ACCESS_KEY_ID" not in clean_env
    assert "HF_TOKEN" not in clean_env
    assert clean_env["PYTHONPATH"] == ""


def test_sandbox_ast_validation():
    runner = SandboxedCodeRunner(check_connectivity=False)
    assert runner.validate_ast("def foo(): return 42") is True
    assert runner.validate_ast("def foo() invalid syntax !!!") is False


def test_docker_unavailable_raises_sandbox_unavailable_error_on_init(monkeypatch):
    monkeypatch.setattr(
        "vipym.evaluation.sandbox.docker_sandbox.is_docker_available", lambda timeout_sec=3.0: False
    )
    monkeypatch.delenv("VIPYM_ALLOW_UNSAFE", raising=False)

    with pytest.raises(SandboxUnavailableError) as excinfo:
        SandboxedCodeRunner()
    assert "Docker daemon is not running or unreachable" in str(excinfo.value)


def test_docker_unavailable_raises_error_on_execute(monkeypatch):
    monkeypatch.setattr(
        "vipym.evaluation.sandbox.docker_sandbox.is_docker_available", lambda timeout_sec=3.0: False
    )
    monkeypatch.delenv("VIPYM_ALLOW_UNSAFE", raising=False)

    runner = SandboxedCodeRunner(check_connectivity=False)
    with pytest.raises(SandboxUnavailableError):
        runner.execute_in_sandbox("print('hello')")


def test_double_opt_in_missing_env_var_raises_error(monkeypatch):
    monkeypatch.setattr(
        "vipym.evaluation.sandbox.docker_sandbox.is_docker_available", lambda timeout_sec=3.0: False
    )
    monkeypatch.delenv("VIPYM_ALLOW_UNSAFE", raising=False)

    config = SandboxSecurityConfig(allow_unsafe_execution=True)
    with pytest.raises(SandboxUnavailableError):
        SandboxedCodeRunner(config=config)


def test_double_opt_in_missing_config_flag_raises_error(monkeypatch):
    monkeypatch.setattr(
        "vipym.evaluation.sandbox.docker_sandbox.is_docker_available", lambda timeout_sec=3.0: False
    )
    monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")

    config = SandboxSecurityConfig(allow_unsafe_execution=False)
    with pytest.raises(SandboxUnavailableError):
        SandboxedCodeRunner(config=config)


def test_degraded_mode_works_with_both_flags_and_logs_warning(monkeypatch, caplog):
    monkeypatch.setattr(
        "vipym.evaluation.sandbox.docker_sandbox.is_docker_available", lambda timeout_sec=3.0: False
    )
    monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")
    config = SandboxSecurityConfig(allow_unsafe_execution=True)

    with caplog.at_level(logging.WARNING):
        runner = SandboxedCodeRunner(config=config)
        code = """
def test_calc():
    assert 2 + 2 == 4
test_calc()
"""
        res = runner.execute_in_sandbox(code, timeout_sec=5)

    assert res.passed is True
    assert res.compile_success is True
    assert any("DEGRADED MODE" in record.message for record in caplog.records)


def test_sandbox_timeout_containment_in_degraded_mode(monkeypatch):
    monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")
    config = SandboxSecurityConfig(allow_unsafe_execution=True)
    runner = SandboxedCodeRunner(config=config)

    infinite_loop_code = """
import time
while True:
    time.sleep(0.1)
"""
    res = runner.execute_in_sandbox(infinite_loop_code, timeout_sec=1)
    assert res.passed is False
    assert res.timed_out is True


def test_docker_execution_when_available(monkeypatch):
    monkeypatch.setattr(
        "vipym.evaluation.sandbox.docker_sandbox.is_docker_available", lambda timeout_sec=3.0: True
    )

    mock_completed_process = MagicMock()
    mock_completed_process.returncode = 0
    mock_completed_process.stdout = "Docker Test Passed\n"
    mock_completed_process.stderr = ""

    called_cmd = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        called_cmd.extend(cmd)
        return mock_completed_process

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    runner = SandboxedCodeRunner(check_connectivity=True)
    res = runner.execute_in_sandbox("assert True", timeout_sec=5)

    assert res.passed is True
    assert res.compile_success is True
    assert res.stdout == "Docker Test Passed\n"
    assert "docker" in called_cmd[0]
    assert "--network" in called_cmd
    assert "none" in called_cmd
    assert "--memory" in called_cmd
    assert "--cpus" in called_cmd
    assert "--read-only" in called_cmd
