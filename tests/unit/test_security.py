"""Unit tests for sandboxed execution and environment sanitization."""

from vipym.security.sandbox import SandboxedCodeRunner
from vipym.security.sanitizer import sanitize_execution_environment


def test_environment_sanitization():
    clean_env = sanitize_execution_environment()
    assert "AWS_ACCESS_KEY_ID" not in clean_env
    assert "HF_TOKEN" not in clean_env
    assert clean_env["PYTHONPATH"] == ""


def test_sandbox_ast_validation():
    runner = SandboxedCodeRunner()
    assert runner.validate_ast("def foo(): return 42") is True
    assert runner.validate_ast("def foo() invalid syntax !!!") is False


def test_sandbox_safe_execution():
    runner = SandboxedCodeRunner()
    code = """
def test_calc():
    assert 2 + 2 == 4
test_calc()
"""
    res = runner.execute_in_sandbox(code, timeout_sec=5)
    assert res.passed is True
    assert res.compile_success is True


def test_sandbox_timeout_containment():
    runner = SandboxedCodeRunner()
    infinite_loop_code = """
import time
while True:
    time.sleep(0.1)
"""
    res = runner.execute_in_sandbox(infinite_loop_code, timeout_sec=1)
    assert res.passed is False
    assert res.timed_out is True
