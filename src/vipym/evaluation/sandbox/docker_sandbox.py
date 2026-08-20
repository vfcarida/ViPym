"""Sandboxed Execution Engine using Docker / gVisor (runsc)."""

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pydantic

from vipym.core.exceptions import SandboxUnavailableError
from vipym.core.logger import get_logger
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig

logger = get_logger(__name__)


def is_docker_available(timeout_sec: float = 3.0) -> bool:
    """Check if the Docker daemon is installed and actively responding."""
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False
    try:
        res = subprocess.run(
            [docker_bin, "info"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return res.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


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

    def __init__(
        self,
        config: SandboxSecurityConfig | None = None,
        check_connectivity: bool = True,
    ) -> None:
        self.config = config or SandboxSecurityConfig()
        if check_connectivity:
            self.verify_environment()

    def _is_unsafe_allowed(self) -> bool:
        """Check whether explicit double opt-in for degraded bare subprocess execution is met.

        Requires BOTH:
        1. config.allow_unsafe_execution is True
        2. Environment variable VIPYM_ALLOW_UNSAFE == '1'
        """
        env_flag = os.environ.get("VIPYM_ALLOW_UNSAFE", "").strip() == "1"
        return bool(self.config.allow_unsafe_execution and env_flag)

    def check_docker_connectivity(self) -> bool:
        """Verify that Docker daemon is reachable."""
        return is_docker_available()

    def verify_environment(self) -> None:
        """Verify container sandbox readiness or validate double opt-in for degraded mode."""
        if self._is_unsafe_allowed():
            logger.warning(
                "SECURITY WARNING: SandboxedCodeRunner operating in DEGRADED MODE (bare subprocess). "
                "Both allow_unsafe_execution=True and VIPYM_ALLOW_UNSAFE=1 are set."
            )
            return

        if not self.check_docker_connectivity():
            raise SandboxUnavailableError(
                "Docker daemon is not running or unreachable. Cannot safely execute untrusted code in sandbox. "
                "To allow unsafe non-containerized execution for local testing, set 'allow_unsafe_execution: true' "
                "in evaluation config and set environment variable 'VIPYM_ALLOW_UNSAFE=1'."
            )
        logger.info(
            f"Docker Sandbox initialized successfully (Image: {self.config.docker_image}, "
            f"Timeout: {self.config.timeout_seconds}s, Memory: {self.config.memory_limit_mb}MB)."
        )

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

        # Defense-in-depth: If double opt-in is active, execute in degraded subprocess mode
        if self._is_unsafe_allowed():
            return self._execute_degraded_subprocess(full_code, timeout, start_time)

        # Otherwise, strictly require Docker daemon connectivity
        if not self.check_docker_connectivity():
            raise SandboxUnavailableError(
                "Docker daemon is not available or unreachable. Cannot execute code in sandbox. "
                "To allow unsafe execution, both allow_unsafe_execution=True and VIPYM_ALLOW_UNSAFE=1 are required."
            )

        return self._execute_docker(full_code, timeout, start_time)

    def _execute_docker(self, full_code: str, timeout: int, start_time: float) -> ExecutionResult:
        """Execute code inside an isolated Docker container with strict constraints."""
        docker_bin = shutil.which("docker") or "docker"
        cmd = [
            docker_bin,
            "run",
            "--rm",
            "-i",
        ]
        if self.config.network_disabled:
            cmd.extend(["--network", "none"])
        if self.config.memory_limit_mb:
            cmd.extend(["--memory", f"{self.config.memory_limit_mb}m"])
        if self.config.cpu_quota:
            cmd.extend(["--cpus", str(self.config.cpu_quota)])
        if self.config.pids_limit:
            cmd.extend(["--pids-limit", str(self.config.pids_limit)])
        if self.config.read_only_rootfs:
            cmd.extend(["--read-only"])
        if self.config.drop_capabilities:
            for cap in self.config.drop_capabilities:
                cmd.extend(["--cap-drop", cap])

        cmd.extend([self.config.docker_image, "python", "-u", "-"])

        logger.info(f"Executing untrusted code in Docker container ({self.config.docker_image}).")
        try:
            res = subprocess.run(
                cmd,
                input=full_code,
                capture_output=True,
                text=True,
                timeout=timeout,
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

    def _execute_degraded_subprocess(
        self, full_code: str, timeout: int, start_time: float
    ) -> ExecutionResult:
        """Execute code in bare subprocess with sanitized environment (DEGRADED MODE)."""
        logger.warning(
            "DEGRADED MODE: Running code in bare subprocess without container isolation. "
            "allow_unsafe_execution=True and VIPYM_ALLOW_UNSAFE=1 are both set."
        )
        script_path = None
        try:
            from vipym.security.sanitizer import sanitize_execution_environment

            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", encoding="utf-8", delete=False
            ) as f:
                f.write(full_code)
                script_path = f.name

            clean_env = sanitize_execution_environment()
            res = subprocess.run(
                [sys.executable, script_path],
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
            if script_path:
                try:
                    Path(script_path).unlink(missing_ok=True)
                except Exception:
                    pass
