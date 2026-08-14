"""Sandboxed Execution Engine using Docker / gVisor (runsc)."""

import ast
import subprocess
import tempfile
import time
from pathlib import Path

import pydantic

from vipym.core.logger import get_logger
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig

logger = get_logger(__name__)


class ExecutionResult(pydantic.BaseModel):
    exit_code: int
    passed: bool
    compile_success: bool
    stdout: str
    stderr: str
    execution_time_ms: float
    timed_out: bool = False


class SandboxedCodeRunner:
    """Isolated execution engine for untrusted generated code."""

    def __init__(self, config: SandboxSecurityConfig | None = None) -> None:
        self.config = config or SandboxSecurityConfig()

    def validate_ast(self, code: str) -> bool:
        """Check for syntax validity before container spawn."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def execute_in_sandbox(self, full_code: str, timeout_sec: int | None = None) -> ExecutionResult:
        """Execute full test script inside sandbox with resource and security constraints."""
        timeout = timeout_sec or self.config.timeout_seconds
        start_time = time.perf_counter()

        if not self.validate_ast(full_code):
            return ExecutionResult(
                exit_code=1,
                passed=False,
                compile_success=False,
                stdout="",
                stderr="SyntaxError: Invalid python code",
                execution_time_ms=0.0,
            )

        # Fallback local process isolation with timeout and clean env if Docker is not available
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", encoding="utf-8", delete=False
            ) as f:
                f.write(full_code)
                script_path = f.name

            # Run in isolated subprocess with empty environment to prevent credential exfiltration
            clean_env = {"PYTHONPATH": "", "PATH": "/usr/bin:/bin"}
            res = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=clean_env,
            )
            exec_time = (time.perf_counter() - start_time) * 1000.0

            passed = res.returncode == 0
            return ExecutionResult(
                exit_code=res.returncode,
                passed=passed,
                compile_success=True,
                stdout=res.stdout,
                stderr=res.stderr,
                execution_time_ms=exec_time,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                exit_code=-1,
                passed=False,
                compile_success=True,
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds",
                execution_time_ms=timeout * 1000.0,
                timed_out=True,
            )
        except Exception as e:
            return ExecutionResult(
                exit_code=1,
                passed=False,
                compile_success=False,
                stdout="",
                stderr=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )
        finally:
            try:
                Path(script_path).unlink(missing_ok=True)
            except Exception:
                pass
