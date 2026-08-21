"""HumanEval Benchmark Suite (164 tasks).

Implements the official OpenAI HumanEval benchmark with pass@k (k=1, 10, 100) scoring.
Supports single-sample and multi-sample temperature generation.
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
# Bundled Fallback Dataset (Curated Canonical HumanEval Problems)
# ---------------------------------------------------------------------------

_CANONICAL_HUMANEVAL_PROBLEMS: list[dict[str, Any]] = [
    {
        "task_id": "HumanEval/0",
        "entry_point": "has_close_elements",
        "prompt": """def has_close_elements(numbers: list[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
""",
        "canonical_solution": """    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
""",
        "test": """
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
""",
    },
    {
        "task_id": "HumanEval/1",
        "entry_point": "separate_paren_groups",
        "prompt": """def separate_paren_groups(paren_string: str) -> list[str]:
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced, each group is delimited by whitespace.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"
""",
        "canonical_solution": """    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()

    return result
""",
        "test": """
def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == [
        '(()())', '((()))', '()', '((())()())'
    ]
    assert candidate('() (()) ((())) (((())))') == [
        '()', '(())', '((()))', '(((())))'
    ]
    assert candidate('(()(())((())))') == [
        '(()(())((())))'
    ]
    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
""",
    },
    {
        "task_id": "HumanEval/2",
        "entry_point": "truncate_number",
        "prompt": """def truncate_number(number: float) -> float:
    \"\"\" Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1, also called fractional part).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    \"\"\"
""",
        "canonical_solution": """    return number % 1.0
""",
        "test": """
def check(candidate):
    assert candidate(3.5) == 0.5
    assert abs(candidate(1.33) - 0.33) < 1e-4
    assert abs(candidate(123.456) - 0.456) < 1e-4
""",
    },
    {
        "task_id": "HumanEval/3",
        "entry_point": "below_zero",
        "prompt": """def below_zero(operations: list[int]) -> bool:
    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should returns True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    \"\"\"
""",
        "canonical_solution": """    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False
""",
        "test": """
def check(candidate):
    assert candidate([]) == False
    assert candidate([1, 2, -3, 1, 2, -3]) == False
    assert candidate([1, 2, -4, 5, 6]) == True
    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False
    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True
    assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) == True
""",
    },
]


# ---------------------------------------------------------------------------
# HumanEvalSuite Implementation
# ---------------------------------------------------------------------------


class HumanEvalSuite(EvaluationSuite):
    """HumanEval execution-based coding benchmark suite (164 tasks)."""

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
        return "humaneval"

    @property
    def version(self) -> str:
        return "v1.0.0"

    # ------------------------------------------------------------------
    # Task Loading
    # ------------------------------------------------------------------

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load HumanEval tasks from Hugging Face or fallback canonical dataset."""
        tasks: list[BenchmarkTask] = []

        try:
            from datasets import load_dataset  # type: ignore[import]

            hf_ds = load_dataset("openai/openai_humaneval", split="test")
            for item in hf_ds:
                tasks.append(
                    BenchmarkTask(
                        task_id=item["task_id"],
                        suite=self.name,
                        entry_point=item["entry_point"],
                        prompt=item["prompt"],
                        canonical_solution=item["canonical_solution"],
                        test_code=item["test"],
                        timeout_seconds=self.timeout_per_task,
                        metadata={"entry_point": item["entry_point"]},
                    )
                )
                if limit and len(tasks) >= limit:
                    break
        except Exception:  # noqa: BLE001
            for item in _CANONICAL_HUMANEVAL_PROBLEMS:
                tasks.append(
                    BenchmarkTask(
                        task_id=item["task_id"],
                        suite=self.name,
                        entry_point=item["entry_point"],
                        prompt=item["prompt"],
                        canonical_solution=item["canonical_solution"],
                        test_code=item["test"],
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
        """Return raw prompt for direct code completion."""
        return task.prompt

    # ------------------------------------------------------------------
    # Evaluation Logic
    # ------------------------------------------------------------------

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner,
    ) -> TaskResult:
        """Run generated code against task test assertions in sandbox."""
        clean_code = self._clean_code(task, generated_text)
        entry_point = task.entry_point

        # Build full executable test harness
        full_code = (
            f"{clean_code}\n\n"
            f"{task.test_code}\n\n"
            f"if __name__ == '__main__':\n"
            f"    check({entry_point})\n"
        )

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

    def _clean_code(self, task: BenchmarkTask, raw_text: str) -> str:
        """Extract clean executable code from completion or markdown fences."""
        import textwrap

        text = raw_text
        if "```python" in text:
            m = text.split("```python")[1].split("```")[0]
            text = m
        elif "```" in text:
            m = text.split("```")[1].split("```")[0]
            text = m

        entry_point = task.entry_point
        if f"def {entry_point}" not in text:
            dedented = textwrap.dedent(text).strip()
            indented = textwrap.indent(dedented, "    ")
            text = f"{task.prompt.rstrip()}\n{indented}"
        else:
            text = textwrap.dedent(text).strip()

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
        """Run full HumanEval evaluation with pass@k calculations."""
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

        logger.info(f"HumanEval results: pass@1={pass_at_1:.2%} ({passed_tasks}/{total_tasks})")

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
EvaluationRegistry.register("humaneval", HumanEvalSuite)
EvaluationRegistry.register("human_eval", HumanEvalSuite)
