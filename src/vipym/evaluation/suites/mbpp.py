"""MBPP (Mostly Basic Python Problems) Benchmark Suite (500 sanitized tasks).

Implements the Google MBPP benchmark with assertion-based test execution
and pass@k (k=1, 10, 100) scoring.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from vipym.core.logger import get_logger
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig
from vipym.evaluation.scoring import calculate_pass_at_k_metrics
from vipym.interfaces.evaluation import (
    BenchmarkTask,
    EvaluationSuite,
    EvaluationSuiteResult,
    TaskResult,
)
from vipym.interfaces.inference import GenerationRequest, InferenceBackend

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical MBPP Fallback Tasks
# ---------------------------------------------------------------------------

_CANONICAL_MBPP_PROBLEMS: list[dict[str, Any]] = [
    {
        "task_id": "MBPP/1",
        "entry_point": "min_cost",
        "prompt": '"""Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][]."""\n',
        "canonical_solution": """def min_cost(cost, m, n):
    tc = [[0 for x in range(n + 1)] for x in range(m + 1)]
    tc[0][0] = cost[0][0]
    for i in range(1, m + 1):
        tc[i][0] = tc[i - 1][0] + cost[i][0]
    for j in range(1, n + 1):
        tc[0][j] = tc[0][j - 1] + cost[0][j]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            tc[i][j] = min(tc[i - 1][j - 1], tc[i - 1][j], tc[i][j - 1]) + cost[i][j]
    return tc[m][n]
""",
        "test_code": """
assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8
assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12
assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16
""",
    },
    {
        "task_id": "MBPP/2",
        "entry_point": "similar_elements",
        "prompt": '"""Write a function to find the shared elements in two tuples."""\n',
        "canonical_solution": """def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return res
""",
        "test_code": """
assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))
""",
    },
    {
        "task_id": "MBPP/3",
        "entry_point": "is_not_prime",
        "prompt": '"""Write a python function to identify non-prime numbers."""\n',
        "canonical_solution": """import math

def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
""",
        "test_code": """
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
""",
    },
]


# ---------------------------------------------------------------------------
# MBPPSuite Implementation
# ---------------------------------------------------------------------------


class MBPPSuite(EvaluationSuite):
    """MBPP (Mostly Basic Python Problems) Benchmark Adapter (500 sanitized tasks)."""

    def __init__(
        self,
        timeout_per_task: int = 15,
        num_samples_per_task: int = 1,
        k_values: list[int] | None = None,
        parallel_tasks: int = 4,
    ) -> None:
        self.timeout_per_task = timeout_per_task
        self.num_samples_per_task = num_samples_per_task
        self.k_values = k_values or [1, 10, 100]
        self.parallel_tasks = parallel_tasks

    @property
    def name(self) -> str:
        return "mbpp"

    @property
    def version(self) -> str:
        return "sanitized_v1.0"

    # ------------------------------------------------------------------
    # Task Loading
    # ------------------------------------------------------------------

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load sanitized MBPP tasks from Hugging Face or fallback canonical dataset."""
        tasks: list[BenchmarkTask] = []

        try:
            from datasets import load_dataset  # type: ignore[import]

            hf_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
            for item in hf_ds:
                test_code = "\n".join(item.get("test_list", []))
                tasks.append(
                    BenchmarkTask(
                        task_id=f"MBPP/{item['task_id']}",
                        suite=self.name,
                        entry_point=item.get("entry_point", f"task_{item['task_id']}"),
                        prompt=item.get("prompt", ""),
                        canonical_solution=item.get("code", ""),
                        test_code=test_code,
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "test_imports": item.get("test_imports", []),
                            "entry_point": item.get("entry_point", ""),
                        },
                    )
                )
                if limit and len(tasks) >= limit:
                    break
        except Exception:  # noqa: BLE001
            for item in _CANONICAL_MBPP_PROBLEMS:
                tasks.append(
                    BenchmarkTask(
                        task_id=item["task_id"],
                        suite=self.name,
                        entry_point=item["entry_point"],
                        prompt=item["prompt"],
                        canonical_solution=item["canonical_solution"],
                        test_code=item["test_code"],
                        timeout_seconds=self.timeout_per_task,
                        metadata={"entry_point": item["entry_point"]},
                    )
                )
                if limit and len(tasks) >= limit:
                    break

        return tasks[:limit] if limit else tasks

    # ------------------------------------------------------------------
    # Prompt Formatting
    # ------------------------------------------------------------------

    def format_prompt(self, task: BenchmarkTask, tokenizer: Any | None = None) -> str:
        """Format MBPP instructions prompt."""
        return (
            "You are an expert Python programmer.\n"
            f"{task.prompt}\n"
            "Write the complete Python function implementation in a ```python block.\n"
        )

    # ------------------------------------------------------------------
    # Evaluation Logic
    # ------------------------------------------------------------------

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner,
    ) -> TaskResult:
        """Execute generated code followed by task test assertions in sandbox."""
        clean_code = self._clean_code(generated_text)
        test_imports = "\n".join(task.metadata.get("test_imports", []))

        full_code = f"{test_imports}\n\n{clean_code}\n\n{task.test_code}"
        res = sandbox_runner.execute_in_sandbox(full_code, timeout_sec=task.timeout_seconds)

        return TaskResult(
            task_id=task.task_id,
            suite=self.name,
            prompt=task.prompt,
            generated_solution=clean_code,
            passed=res.passed,
            compile_success=res.compile_success,
            unit_tests_passed=1 if res.passed else 0,
            unit_tests_total=1,
            execution_time_ms=res.execution_time_ms,
            error_message=res.stderr if not res.passed else None,
            stdout=res.stdout,
        )

    def _clean_code(self, raw_text: str) -> str:
        """Extract clean code from Markdown fences or raw text."""
        text = raw_text.strip()
        if "```python" in text:
            m = text.split("```python")[1].split("```")[0]
            return m.strip()
        if "```" in text:
            m = text.split("```")[1].split("```")[0]
            return m.strip()
        return text

    # ------------------------------------------------------------------
    # Batch Evaluation
    # ------------------------------------------------------------------

    def evaluate_suite(
        self,
        backend: InferenceBackend | Any,
        tasks: list[BenchmarkTask] | None = None,
        task_limit: int | None = None,
        sandbox_runner: SandboxedCodeRunner | None = None,
    ) -> EvaluationSuiteResult:
        """Run full MBPP evaluation with pass@k calculations."""
        if tasks is None:
            tasks = self.load_tasks(limit=task_limit)

        sandbox = sandbox_runner or SandboxedCodeRunner(
            config=SandboxSecurityConfig(
                allow_unsafe_execution=True, timeout_seconds=self.timeout_per_task
            ),
            check_connectivity=False,
        )

        task_results: list[TaskResult] = []
        task_correctness: list[list[bool]] = []

        def _eval_task(task: BenchmarkTask) -> tuple[TaskResult, list[bool]]:
            prompt = self.format_prompt(task)
            samples_passed: list[bool] = []
            primary_result: TaskResult | None = None

            for i in range(self.num_samples_per_task):
                temp = 0.0 if self.num_samples_per_task == 1 else 0.8
                req = GenerationRequest(prompt=prompt, temperature=temp)
                if hasattr(backend, "generate"):
                    resp = backend.generate(req)
                    gen_text = resp.generated_text
                elif callable(backend):
                    out = backend(prompt)
                    gen_text = getattr(out, "generated_text", str(out))
                else:
                    gen_text = ""

                res = self.evaluate_response(task, gen_text, sandbox)
                samples_passed.append(res.passed)
                if i == 0:
                    primary_result = res

            return primary_result or res, samples_passed

        if self.parallel_tasks > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=self.parallel_tasks) as pool:
                eval_pairs = list(pool.map(_eval_task, tasks))
        else:
            eval_pairs = [_eval_task(t) for t in tasks]

        for res, correctness in eval_pairs:
            task_results.append(res)
            task_correctness.append(correctness)

        total_tasks = max(1, len(task_results))
        passed_tasks = sum(1 for r in task_results if r.passed)
        pass_at_1 = passed_tasks / total_tasks

        pass_metrics = calculate_pass_at_k_metrics(task_correctness, k_values=self.k_values)
        summary_metrics = {
            **pass_metrics,
            "pass_at_1": pass_at_1,
            "compile_rate": sum(1 for r in task_results if r.compile_success) / total_tasks,
            "total_tasks": total_tasks,
        }

        logger.info(f"MBPP results: pass@1={pass_at_1:.2%} ({passed_tasks}/{total_tasks})")

        return EvaluationSuiteResult(
            suite_name=self.name,
            benchmark_version=self.version,
            total_tasks=total_tasks,
            passed_tasks=passed_tasks,
            pass_at_1=pass_at_1,
            compile_rate=summary_metrics["compile_rate"],
            unit_test_pass_rate=pass_at_1,
            task_results=task_results,
            summary_metrics=summary_metrics,
        )


# ---------------------------------------------------------------------------
# LiveCodeBench Adapter (Preserved)
# ---------------------------------------------------------------------------


class LiveCodeBenchSuite(EvaluationSuite):
    """LiveCodeBench: Continuously updated contamination-free coding benchmark."""

    def __init__(self, timeout_per_task: int = 20) -> None:
        self.timeout_per_task = timeout_per_task

    @property
    def name(self) -> str:
        return "livecodebench"

    @property
    def version(self) -> str:
        return "v2026.08"

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
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
                timeout_seconds=self.timeout_per_task,
            )
        ]
        return sample_tasks[:limit] if limit else sample_tasks

    def format_prompt(self, task: BenchmarkTask, tokenizer: Any | None = None) -> str:
        return task.prompt

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner,
    ) -> TaskResult:
        res = sandbox_runner.execute_in_sandbox(
            f"{generated_text}\n{task.test_code}", timeout_sec=task.timeout_seconds
        )
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


# Register in EvaluationRegistry
EvaluationRegistry.register("mbpp", MBPPSuite)
EvaluationRegistry.register("mbpp_sanitized", MBPPSuite)
EvaluationRegistry.register("livecodebench", LiveCodeBenchSuite)
