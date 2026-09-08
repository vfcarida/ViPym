"""LiveCodeBench Suite: Contamination-free coding benchmark.

Implements competitive programming and dynamic algorithmic tasks with strict
test harness verification, supporting both HuggingFace datasets and bundled canonical tasks.
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
# Canonical Bundled Tasks for LiveCodeBench
# ---------------------------------------------------------------------------

_CANONICAL_LCB_PROBLEMS: list[dict[str, Any]] = [
    {
        "task_id": "LCB/2026_01",
        "entry_point": "longest_valid_subsequence",
        "prompt": '''def longest_valid_subsequence(nums: list[int], k: int) -> int:
    """Given an integer array `nums` and an integer `k`, return the length of the longest
    subsequence such that the absolute difference between any two consecutive elements
    in the subsequence is at most `k`.

    >>> longest_valid_subsequence([1, 4, 3, 2, 7, 5], 1)
    3
    >>> longest_valid_subsequence([10, 20, 30], 5)
    1
    """
''',
        "canonical_solution": """def longest_valid_subsequence(nums: list[int], k: int) -> int:
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if abs(nums[i] - nums[j]) <= k:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
    return max(dp)
""",
        "test_code": """
def check(candidate):
    assert candidate([1, 4, 3, 2, 7, 5], 1) == 3
    assert candidate([10, 20, 30], 5) == 1
    assert candidate([1, 2, 3, 4, 5], 2) == 5
    assert candidate([5, 1, 4, 2, 3], 1) == 3
    assert candidate([], 5) == 0
    assert candidate([42], 0) == 1

check(longest_valid_subsequence)
""",
    },
    {
        "task_id": "LCB/2026_02",
        "entry_point": "min_operations_to_balance",
        "prompt": '''def min_operations_to_balance(s: str) -> int:
    """Given a string `s` consisting only of '(' and ')', return the minimum number of
    insertions or deletions needed to make the string valid/balanced.

    >>> min_operations_to_balance("())")
    1
    >>> min_operations_to_balance("(((")
    3
    >>> min_operations_to_balance("()()")
    0
    """
''',
        "canonical_solution": """def min_operations_to_balance(s: str) -> int:
    open_count = 0
    needed_ops = 0
    for char in s:
        if char == '(':
            open_count += 1
        elif char == ')':
            if open_count > 0:
                open_count -= 1
            else:
                needed_ops += 1
    return needed_ops + open_count
""",
        "test_code": """
def check(candidate):
    assert candidate("())") == 1
    assert candidate("(((") == 3
    assert candidate("()()") == 0
    assert candidate("()))((") == 4
    assert candidate("") == 0
    assert candidate(")(()(") == 3

check(min_operations_to_balance)
""",
    },
    {
        "task_id": "LCB/2026_03",
        "entry_point": "max_subarray_xor",
        "prompt": '''def max_subarray_xor(nums: list[int]) -> int:
    """Given an integer array `nums`, return the maximum possible bitwise XOR sum
    of any non-empty contiguous subarray.

    >>> max_subarray_xor([1, 2, 3, 4])
    7
    >>> max_subarray_xor([8, 1, 2, 12, 7, 6])
    15
    """
''',
        "canonical_solution": """def max_subarray_xor(nums: list[int]) -> int:
    if not nums:
        return 0
    max_xor = 0
    prefix = 0
    trie: dict[Any, Any] = {}

    def insert(val: int) -> None:
        node = trie
        for bit in range(31, -1, -1):
            b = (val >> bit) & 1
            if b not in node:
                node[b] = {}
            node = node[b]

    def query(val: int) -> int:
        node = trie
        res = 0
        for bit in range(31, -1, -1):
            b = (val >> bit) & 1
            opp = 1 - b
            if opp in node:
                res |= (1 << bit)
                node = node[opp]
            elif b in node:
                node = node[b]
            else:
                break
        return res

    insert(0)
    for x in nums:
        prefix ^= x
        insert(prefix)
        max_xor = max(max_xor, query(prefix))
    return max_xor
""",
        "test_code": """
def check(candidate):
    assert candidate([1, 2, 3, 4]) == 7
    assert candidate([8, 1, 2, 12, 7, 6]) == 15
    assert candidate([4, 6]) == 6
    assert candidate([5]) == 5

check(max_subarray_xor)
""",
    },
]


# ---------------------------------------------------------------------------
# LiveCodeBench Suite Implementation
# ---------------------------------------------------------------------------


class LiveCodeBenchSuite(EvaluationSuite):
    """LiveCodeBench: Continuously updated contamination-free coding benchmark."""

    def __init__(
        self,
        timeout_per_task: int = 20,
        num_samples_per_task: int = 1,
        k_values: list[int] | None = None,
        parallel_tasks: int = 4,
    ) -> None:
        self.timeout_per_task = timeout_per_task
        self.num_samples_per_task = num_samples_per_task
        self.k_values = k_values or [1, 5, 10]
        self.parallel_tasks = parallel_tasks

    @property
    def name(self) -> str:
        return "livecodebench"

    @property
    def version(self) -> str:
        return "v2026.08"

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load LiveCodeBench tasks from HF or fallback to verified canonical tasks."""
        tasks: list[BenchmarkTask] = []
        try:
            from datasets import load_dataset  # type: ignore[import]

            hf_ds = load_dataset("livecodebench/code_generation_lite", split="test")
            for item in hf_ds:
                tasks.append(
                    BenchmarkTask(
                        task_id=f"LCB/{item.get('question_id', len(tasks))}",
                        suite=self.name,
                        entry_point=item.get("entry_point", "solve"),
                        prompt=item.get("prompt", item.get("question_content", "")),
                        canonical_solution=item.get("canonical_solution", ""),
                        test_code=item.get("test", ""),
                        release_date=str(item.get("contest_date", "2026-01-01")),
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "difficulty": item.get("difficulty", "medium"),
                            "contest_id": item.get("contest_id", ""),
                        },
                    )
                )
                if limit and len(tasks) >= limit:
                    break
        except Exception:  # noqa: BLE001
            for item in _CANONICAL_LCB_PROBLEMS:
                tasks.append(
                    BenchmarkTask(
                        task_id=item["task_id"],
                        suite=self.name,
                        entry_point=item["entry_point"],
                        prompt=item["prompt"],
                        canonical_solution=item["canonical_solution"],
                        test_code=item["test_code"],
                        release_date="2026-07-20",
                        timeout_seconds=self.timeout_per_task,
                        metadata={"entry_point": item["entry_point"]},
                    )
                )
                if limit and len(tasks) >= limit:
                    break

        return tasks[:limit] if limit else tasks

    def format_prompt(self, task: BenchmarkTask, tokenizer: Any | None = None) -> str:
        """Format prompt for the code generator."""
        return (
            "You are an expert algorithmic programmer.\n"
            f"{task.prompt}\n"
            "Implement the function with optimal time and space complexity.\n"
            "Return ONLY the executable Python code inside a ```python block.\n"
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

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner,
    ) -> TaskResult:
        """Evaluate a model generation against task test assertions in sandbox."""
        clean_code = self._clean_code(generated_text)
        full_code = f"{clean_code}\n\n{task.test_code}"

        res = sandbox_runner.execute_in_sandbox(
            full_code,
            timeout_sec=task.timeout_seconds,
        )

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

    def evaluate_suite(
        self,
        backend: InferenceBackend | Any,
        tasks: list[BenchmarkTask] | None = None,
        task_limit: int | None = None,
        sandbox_runner: SandboxedCodeRunner | None = None,
    ) -> EvaluationSuiteResult:
        """Run full LiveCodeBench evaluation suite."""
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

        logger.info(f"LiveCodeBench results: pass@1={pass_at_1:.2%} ({passed_tasks}/{total_tasks})")

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


# Register in EvaluationRegistry
EvaluationRegistry.register("livecodebench", LiveCodeBenchSuite)
