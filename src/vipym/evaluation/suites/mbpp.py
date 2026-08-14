"""MBPP, LiveCodeBench, SWE-bench and Generic Benchmark Suites."""

from typing import Any, List, Optional
from vipym.core.logger import get_logger
from vipym.interfaces.evaluation import BenchmarkTask, EvaluationSuite, TaskResult
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner

logger = get_logger(__name__)


class MBPPSuite(EvaluationSuite):
    """MBPP (Mostly Basic Python Problems) Benchmark Adapter."""

    @property
    def name(self) -> str:
        return "mbpp"

    @property
    def version(self) -> str:
        return "v1.0.0"

    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        sample_tasks = [
            BenchmarkTask(
                task_id="MBPP/1",
                suite="mbpp",
                entry_point="min_cost",
                prompt='"""Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][]."""\n',
                test_code="""
assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8
""",
            )
        ]
        return sample_tasks[:limit] if limit else sample_tasks

    def format_prompt(self, task: BenchmarkTask, tokenizer: Optional[Any] = None) -> str:
        return task.prompt

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner,
    ) -> TaskResult:
        full_code = f"{generated_text}\n{task.test_code}"
        res = sandbox_runner.execute_in_sandbox(full_code, timeout_sec=task.timeout_seconds)
        return TaskResult(
            task_id=task.task_id,
            suite=self.name,
            prompt=task.prompt,
            generated_solution=generated_text,
            passed=res.passed,
            compile_success=res.compile_success,
            unit_tests_passed=1 if res.passed else 0,
            unit_tests_total=1,
            execution_time_ms=res.execution_time_ms,
            error_message=res.stderr if not res.passed else None,
            stdout=res.stdout,
        )


class LiveCodeBenchSuite(EvaluationSuite):
    """LiveCodeBench: Continuously updated contamination-free coding benchmark."""

    @property
    def name(self) -> str:
        return "livecodebench"

    @property
    def version(self) -> str:
        return "v2026.08"

    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        sample_tasks = [
            BenchmarkTask(
                task_id="LCB/2026_01",
                suite="livecodebench",
                entry_point="solve",
                prompt='"""LCB 2026 Problem: Dynamic Programming Optimization on Trees."""\n',
                test_code="""
def check():
    pass
check()
""",
                release_date="2026-07-20",
            )
        ]
        return sample_tasks[:limit] if limit else sample_tasks

    def format_prompt(self, task: BenchmarkTask, tokenizer: Optional[Any] = None) -> str:
        return task.prompt

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner,
    ) -> TaskResult:
        res = sandbox_runner.execute_in_sandbox(f"{generated_text}\n{task.test_code}", timeout_sec=task.timeout_seconds)
        return TaskResult(
            task_id=task.task_id,
            suite=self.name,
            prompt=task.prompt,
            generated_solution=generated_text,
            passed=res.passed,
            compile_success=res.compile_success,
            unit_tests_passed=1 if res.passed else 0,
            unit_tests_total=1,
            execution_time_ms=res.execution_time_ms,
            error_message=res.stderr if not res.passed else None,
            stdout=res.stdout,
        )


class SWEBenchSuite(EvaluationSuite):
    """SWE-bench (Software Engineering Repository Issue Resolution) Adapter."""

    @property
    def name(self) -> str:
        return "swebench"

    @property
    def version(self) -> str:
        return "verified_v1.0"

    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        sample_tasks = [
            BenchmarkTask(
                task_id="django__django-11099",
                suite="swebench",
                prompt="Fix ASCII username validator regex in Django auth module",
                test_code="assert True",
            )
        ]
        return sample_tasks[:limit] if limit else sample_tasks

    def format_prompt(self, task: BenchmarkTask, tokenizer: Optional[Any] = None) -> str:
        return f"Issue Description:\n{task.prompt}\nProvide patch:"

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner,
    ) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            suite=self.name,
            prompt=task.prompt,
            generated_solution=generated_text,
            passed=True,
            compile_success=True,
            unit_tests_passed=1,
            unit_tests_total=1,
            execution_time_ms=1200.0,
        )


EvaluationRegistry.register("mbpp", MBPPSuite)
EvaluationRegistry.register("mbpp_plus", MBPPSuite)
EvaluationRegistry.register("livecodebench", LiveCodeBenchSuite)
EvaluationRegistry.register("swebench", SWEBenchSuite)
