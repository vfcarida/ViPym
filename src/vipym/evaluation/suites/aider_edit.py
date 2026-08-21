"""Aider Code Editing Benchmark Suite.

Evaluates an LLM's ability to produce cleanly applicable code edits (SEARCH/REPLACE blocks,
unified diffs, or whole-file replacements) that resolve bug reports and refactoring tasks
across 133 Exercism coding exercises.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from vipym.core.logger import get_logger
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.suites.utils.edit_formats import (
    apply_edit,
    validate_format_compliance,
)
from vipym.interfaces.evaluation import (
    BenchmarkTask,
    EvaluationSuite,
    EvaluationSuiteResult,
    TaskResult,
)
from vipym.interfaces.inference import GenerationRequest, InferenceBackend

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical Bundled Tasks for Aider Edit Benchmark
# ---------------------------------------------------------------------------

_AIDER_SAMPLE_TASKS = [
    {
        "task_id": "aider/two-fer",
        "exercise": "two-fer",
        "original_code": """def two_fer(name):
    pass
""",
        "instruction": "Implement two_fer: given a name, return 'One for {name}, one for me.'. If no name is given (default parameter), return 'One for you, one for me.'.",
        "test_code": """
assert two_fer("Alice") == "One for Alice, one for me."
assert two_fer("Bob") == "One for Bob, one for me."
assert two_fer() == "One for you, one for me."
""",
        "canonical_edit": """<<<<<<< SEARCH
def two_fer(name):
    pass
=======
def two_fer(name="you"):
    return f"One for {name}, one for me."
>>>>>>> REPLACE""",
    },
    {
        "task_id": "aider/leap",
        "exercise": "leap",
        "original_code": """def leap_year(year):
    pass
""",
        "instruction": "Implement leap_year: return True if year is a leap year (divisible by 4, except if divisible by 100 unless also divisible by 400).",
        "test_code": """
assert leap_year(2000) is True
assert leap_year(2024) is True
assert leap_year(1900) is False
assert leap_year(2023) is False
""",
        "canonical_edit": """<<<<<<< SEARCH
def leap_year(year):
    pass
=======
def leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
>>>>>>> REPLACE""",
    },
    {
        "task_id": "aider/reverse-string",
        "exercise": "reverse-string",
        "original_code": """def reverse(text):
    return text
""",
        "instruction": "Modify reverse so that it returns the reversed string of input text.",
        "test_code": """
assert reverse("robot") == "tobor"
assert reverse("cool") == "looc"
assert reverse("") == ""
""",
        "canonical_edit": """<<<<<<< SEARCH
def reverse(text):
    return text
=======
def reverse(text):
    return text[::-1]
>>>>>>> REPLACE""",
    },
    {
        "task_id": "aider/isogram",
        "exercise": "isogram",
        "original_code": """def is_isogram(string):
    pass
""",
        "instruction": "Implement is_isogram: determine if a word or phrase is an isogram (no repeating letters, ignoring case and spaces/hyphens).",
        "test_code": """
assert is_isogram("lumberjacks") is True
assert is_isogram("background") is True
assert is_isogram("downstream") is True
assert is_isogram("six-year-old") is True
assert is_isogram("isograms") is False
""",
        "canonical_edit": """<<<<<<< SEARCH
def is_isogram(string):
    pass
=======
def is_isogram(string):
    clean = [c.lower() for c in string if c.isalpha()]
    return len(clean) == len(set(clean))
>>>>>>> REPLACE""",
    },
    {
        "task_id": "aider/acronym",
        "exercise": "acronym",
        "original_code": """def abbreviate(words):
    pass
""",
        "instruction": "Implement abbreviate: convert a phrase to its acronym (e.g. 'Portable Network Graphics' -> 'PNG', 'First In, First Out' -> 'FIFO').",
        "test_code": """
assert abbreviate("Portable Network Graphics") == "PNG"
assert abbreviate("First In, First Out") == "FIFO"
assert abbreviate("Ruby on Rails") == "ROR"
""",
        "canonical_edit": """<<<<<<< SEARCH
def abbreviate(words):
    pass
=======
import re

def abbreviate(words):
    tokens = re.findall(r"[a-zA-Z']+", words.replace("-", " "))
    return "".join(t[0].upper() for t in tokens if t)
>>>>>>> REPLACE""",
    },
]


# ---------------------------------------------------------------------------
# AiderEditSuite Implementation
# ---------------------------------------------------------------------------


class AiderEditSuite(EvaluationSuite):
    """Aider Benchmark Suite measuring code edit compliance, apply rates, and accuracy."""

    def __init__(
        self,
        edit_format: Literal[
            "search_replace", "diff", "udiff", "whole_file", "auto"
        ] = "search_replace",
        timeout_per_task: int = 60,
        parallel_tasks: int = 4,
    ) -> None:
        self.edit_format = edit_format.lower()
        self.timeout_per_task = timeout_per_task
        self.parallel_tasks = parallel_tasks

    @property
    def name(self) -> str:
        return "aider_edit"

    @property
    def version(self) -> str:
        return f"v1.0_{self.edit_format}"

    # ------------------------------------------------------------------
    # Task Loading
    # ------------------------------------------------------------------

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load Aider benchmark tasks from Hugging Face or fallback Exercism collection."""
        tasks: list[BenchmarkTask] = []

        try:
            from datasets import load_dataset  # type: ignore[import]

            hf_ds = load_dataset("aider/exercism-python-edit", split="train")
            for item in hf_ds:
                tasks.append(
                    BenchmarkTask(
                        task_id=item.get("task_id", f"aider/{item.get('exercise', 'task')}"),
                        suite=self.name,
                        entry_point=item.get("exercise", "solution"),
                        prompt=item.get("instruction", ""),
                        canonical_solution=item.get("canonical_edit", ""),
                        test_code=item.get("test_code", ""),
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "original_code": item.get("original_code", ""),
                            "exercise": item.get("exercise", ""),
                        },
                    )
                )
                if limit and len(tasks) >= limit:
                    break
        except Exception:  # noqa: BLE001
            for item in _AIDER_SAMPLE_TASKS:
                tasks.append(
                    BenchmarkTask(
                        task_id=item["task_id"],
                        suite=self.name,
                        entry_point=item["exercise"],
                        prompt=item["instruction"],
                        canonical_solution=item["canonical_edit"],
                        test_code=item["test_code"],
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "original_code": item["original_code"],
                            "exercise": item["exercise"],
                        },
                    )
                )
                if limit and len(tasks) >= limit:
                    break

        return tasks[:limit] if limit else tasks

    # ------------------------------------------------------------------
    # Prompt Formatting
    # ------------------------------------------------------------------

    def format_prompt(self, task: BenchmarkTask, tokenizer: Any | None = None) -> str:
        """Format the task into instructions tailored to the target edit format."""
        original_code = task.metadata.get("original_code", "")

        if self.edit_format == "search_replace":
            format_instructions = (
                "You must provide your edits as one or more SEARCH / REPLACE blocks in the following format:\n\n"
                "<<<<<<< SEARCH\n"
                "# exact code from original file to replace\n"
                "=======\n"
                "# new replacement code\n"
                ">>>>>>> REPLACE\n"
            )
        elif self.edit_format in ("diff", "udiff"):
            format_instructions = (
                "You must provide your edits as a standard unified diff:\n\n"
                "```diff\n"
                "--- a/solution.py\n"
                "+++ b/solution.py\n"
                "@@ ... @@\n"
                "-old code\n"
                "+new code\n"
                "```\n"
            )
        else:
            format_instructions = (
                "Provide the complete updated file code wrapped in a ```python ... ``` block.\n"
            )

        return (
            f"You are an expert software developer editing an existing codebase.\n\n"
            f"=== Original File Content ===\n"
            f"```python\n{original_code}```\n\n"
            f"=== Instruction ===\n"
            f"{task.prompt}\n\n"
            f"=== Format Requirement ===\n"
            f"{format_instructions}\n"
            f"Please output your edit now:\n"
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
        """Apply edit to task's original code and test against unit test assertions."""
        t0 = time.perf_counter()
        original_code = task.metadata.get("original_code", "")

        # 1. Format compliance check
        is_compliant, compliance_msg = validate_format_compliance(generated_text, self.edit_format)

        # 2. Apply edit
        apply_res = apply_edit(original_code, generated_text, expected_format=self.edit_format)

        if not apply_res.success:
            return TaskResult(
                task_id=task.task_id,
                suite=self.name,
                prompt=task.prompt,
                generated_solution=generated_text,
                passed=False,
                compile_success=False,
                unit_tests_passed=0,
                unit_tests_total=1,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                error_message=f"Edit failed to apply: {apply_res.error}",
                stdout=f"Format compliance: {is_compliant} ({compliance_msg})",
            )

        # 3. Execute modified code with test assertions
        full_code = f"{apply_res.modified_code}\n{task.test_code}"
        exec_res = sandbox_runner.execute_in_sandbox(full_code, timeout_sec=task.timeout_seconds)

        return TaskResult(
            task_id=task.task_id,
            suite=self.name,
            prompt=task.prompt,
            generated_solution=apply_res.modified_code,
            passed=exec_res.passed,
            compile_success=exec_res.compile_success,
            unit_tests_passed=1 if exec_res.passed else 0,
            unit_tests_total=1,
            execution_time_ms=exec_res.execution_time_ms,
            error_message=exec_res.stderr if not exec_res.passed else None,
            stdout=f"Format compliant: {is_compliant}. Applied blocks: {apply_res.blocks_applied}/{apply_res.total_blocks}",
        )

    # ------------------------------------------------------------------
    # Batch Suite Evaluation
    # ------------------------------------------------------------------

    def evaluate_suite(
        self,
        backend: InferenceBackend | Any,
        tasks: list[BenchmarkTask] | None = None,
        task_limit: int | None = None,
        sandbox_runner: SandboxedCodeRunner | None = None,
    ) -> EvaluationSuiteResult:
        """Execute full Aider edit evaluation against inference backend."""
        if tasks is None:
            tasks = self.load_tasks(limit=task_limit)

        from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig

        sandbox = sandbox_runner or SandboxedCodeRunner(
            config=SandboxSecurityConfig(
                allow_unsafe_execution=True, timeout_seconds=self.timeout_per_task
            ),
            check_connectivity=False,
        )
        task_results: list[TaskResult] = []
        compliant_count = 0

        def _eval_one(task: BenchmarkTask) -> TaskResult:
            nonlocal compliant_count
            prompt = self.format_prompt(task)
            req = GenerationRequest(prompt=prompt, temperature=0.0)
            if hasattr(backend, "generate"):
                resp = backend.generate(req)
                gen_text = resp.generated_text
            elif callable(backend):
                out = backend(prompt)
                gen_text = getattr(out, "generated_text", str(out))
            else:
                gen_text = ""

            compliant, _ = validate_format_compliance(gen_text, self.edit_format)
            if compliant:
                compliant_count += 1

            return self.evaluate_response(task, gen_text, sandbox)

        if self.parallel_tasks > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=self.parallel_tasks) as pool:
                task_results = list(pool.map(_eval_one, tasks))
        else:
            task_results = [_eval_one(t) for t in tasks]

        total = max(1, len(task_results))
        passed_count = sum(1 for r in task_results if r.passed)
        applied_count = sum(1 for r in task_results if r.compile_success)

        edit_accuracy = passed_count / total
        apply_rate = applied_count / total
        format_compliance_rate = compliant_count / total

        logger.info(
            f"Aider Edit ({self.version}) results: edit_accuracy={edit_accuracy:.2%} "
            f"apply_rate={apply_rate:.2%} format_compliance={format_compliance_rate:.2%} ({passed_count}/{total})"
        )

        return EvaluationSuiteResult(
            suite_name=self.name,
            benchmark_version=self.version,
            total_tasks=len(task_results),
            passed_tasks=passed_count,
            pass_at_1=edit_accuracy,
            compile_rate=apply_rate,
            unit_test_pass_rate=edit_accuracy,
            task_results=task_results,
            summary_metrics={
                "edit_accuracy": edit_accuracy,
                "apply_rate": apply_rate,
                "format_compliance": format_compliance_rate,
                "edit_format": self.edit_format,
                "total_tasks": total,
            },
        )


# Register in EvaluationRegistry
EvaluationRegistry.register("aider_edit", AiderEditSuite)
EvaluationRegistry.register("aider_bench", AiderEditSuite)
EvaluationRegistry.register("aider", AiderEditSuite)
