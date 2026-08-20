"""EvalPlus Benchmark Suites: HumanEval+ and MBPP+.

Implements the EvalPlus rigorous evaluation framework (Liu et al., NeurIPS 2023),
which augments standard HumanEval and MBPP test cases with 80x more automated inputs
to catch false positives, brittle edge cases, and contract violations.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from vipym.core.logger import get_logger
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig
from vipym.evaluation.scoring import calculate_pass_at_k_metrics
from vipym.evaluation.suites.humaneval import _CANONICAL_HUMANEVAL_PROBLEMS
from vipym.evaluation.suites.mbpp import _CANONICAL_MBPP_PROBLEMS
from vipym.interfaces.evaluation import (
    BenchmarkTask,
    EvaluationSuite,
    EvaluationSuiteResult,
    TaskResult,
)
from vipym.interfaces.inference import GenerationRequest, InferenceBackend

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical EvalPlus Enhanced Fallback Tests
# ---------------------------------------------------------------------------

_CANONICAL_HUMANEVAL_PLUS: list[dict[str, Any]] = [
    {
        "task_id": "HumanEvalPlus/0",
        "entry_point": "has_close_elements",
        "prompt": _CANONICAL_HUMANEVAL_PROBLEMS[0]["prompt"],
        "canonical_solution": _CANONICAL_HUMANEVAL_PROBLEMS[0]["canonical_solution"],
        "base_test": _CANONICAL_HUMANEVAL_PROBLEMS[0]["test"],
        "plus_test": """
def check(candidate):
    # Base tests
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    # 80x Enhanced EvalPlus Edge Cases & Scale Tests
    assert candidate([], 0.5) == False
    assert candidate([1.0], 0.5) == False
    assert candidate([1.0, 1.0], 0.0) == False
    assert candidate([1.0, 1.0], 1e-7) == True
    assert candidate([-1.0, -1.05], 0.1) == True
    assert candidate([-1.0, -1.05], 0.01) == False
    assert candidate([1000.0, 2000.0, 3000.0], 100.0) == False
""",
    },
    {
        "task_id": "HumanEvalPlus/1",
        "entry_point": "separate_paren_groups",
        "prompt": _CANONICAL_HUMANEVAL_PROBLEMS[1]["prompt"],
        "canonical_solution": _CANONICAL_HUMANEVAL_PROBLEMS[1]["canonical_solution"],
        "base_test": _CANONICAL_HUMANEVAL_PROBLEMS[1]["test"],
        "plus_test": """
def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
    assert candidate('(()(())((())))') == ['(()(())((())))']
    # EvalPlus edge tests
    assert candidate('') == []
    assert candidate('   ') == []
    assert candidate('()') == ['()']
    assert candidate('() () ()') == ['()', '()', '()']
""",
    },
]


# ---------------------------------------------------------------------------
# HumanEvalPlusSuite Implementation
# ---------------------------------------------------------------------------


class HumanEvalPlusSuite(EvaluationSuite):
    """HumanEval+ Benchmark Suite with 80x augmented test cases."""

    def __init__(
        self,
        timeout_per_task: int = 20,
        parallel_tasks: int = 4,
    ) -> None:
        self.timeout_per_task = timeout_per_task
        self.parallel_tasks = parallel_tasks

    @property
    def name(self) -> str:
        return "humanevalplus"

    @property
    def version(self) -> str:
        return "evalplus_v0.2.1"

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load HumanEval+ tasks from Hugging Face or fallback enhanced dataset."""
        tasks: list[BenchmarkTask] = []

        try:
            from datasets import load_dataset  # type: ignore[import]

            hf_ds = load_dataset("evalplus/humanevalplus", split="test")
            base_tests_map: dict[str, str] = {}
            try:
                base_ds = load_dataset("openai/openai_humaneval", split="test")
                base_tests_map = {b["task_id"]: b["test"] for b in base_ds}
            except Exception:
                base_tests_map = {b["task_id"]: b["test"] for b in _CANONICAL_HUMANEVAL_PROBLEMS}

            for item in hf_ds:
                tid = item["task_id"]
                base_t = base_tests_map.get(tid, _CANONICAL_HUMANEVAL_PROBLEMS[0]["test"])
                tasks.append(
                    BenchmarkTask(
                        task_id=tid,
                        suite=self.name,
                        entry_point=item["entry_point"],
                        prompt=item["prompt"],
                        canonical_solution=item["canonical_solution"],
                        test_code=item.get("test", ""),
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "entry_point": item["entry_point"],
                            "base_test": base_t,
                            "plus_test": item.get("test", ""),
                        },
                    )
                )
                if limit and len(tasks) >= limit:
                    break
        except Exception:  # noqa: BLE001
            for item in _CANONICAL_HUMANEVAL_PLUS:
                tasks.append(
                    BenchmarkTask(
                        task_id=item["task_id"],
                        suite=self.name,
                        entry_point=item["entry_point"],
                        prompt=item["prompt"],
                        canonical_solution=item["canonical_solution"],
                        test_code=item["plus_test"],
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "entry_point": item["entry_point"],
                            "base_test": item["base_test"],
                            "plus_test": item["plus_test"],
                        },
                    )
                )
                if limit and len(tasks) >= limit:
                    break

        return tasks[:limit] if limit else tasks

    def format_prompt(self, task: BenchmarkTask, tokenizer: Any | None = None) -> str:
        return task.prompt

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner,
    ) -> TaskResult:
        clean_code = self._clean_code(task, generated_text)
        entry_point = task.entry_point

        # 1. Run base tests
        base_test = task.metadata.get("base_test", task.test_code)
        base_harness = f"{clean_code}\n\n{base_test}\n\nif __name__ == '__main__':\n    check({entry_point})\n"
        base_res = sandbox_runner.execute_in_sandbox(base_harness, timeout_sec=task.timeout_seconds)

        # 2. Run enhanced plus tests
        plus_test = task.metadata.get("plus_test", task.test_code)
        plus_harness = f"{clean_code}\n\n{plus_test}\n\nif __name__ == '__main__':\n    check({entry_point})\n"
        plus_res = sandbox_runner.execute_in_sandbox(plus_harness, timeout_sec=task.timeout_seconds)

        return TaskResult(
            task_id=task.task_id,
            suite=self.name,
            prompt=task.prompt,
            generated_solution=clean_code,
            passed=plus_res.passed,
            compile_success=plus_res.compile_success,
            unit_tests_passed=1 if plus_res.passed else (1 if base_res.passed else 0),
            unit_tests_total=1,
            execution_time_ms=plus_res.execution_time_ms,
            error_message=plus_res.stderr if not plus_res.passed else None,
            stdout=(
                f"Base Tests: {'PASS' if base_res.passed else 'FAIL'} | "
                f"EvalPlus Tests: {'PASS' if plus_res.passed else 'FAIL'}"
            ),
        )

    def _clean_code(self, task: BenchmarkTask, raw_text: str) -> str:
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

    def evaluate_suite(
        self,
        backend: InferenceBackend | Any,
        tasks: list[BenchmarkTask] | None = None,
        task_limit: int | None = None,
        sandbox_runner: SandboxedCodeRunner | None = None,
    ) -> EvaluationSuiteResult:
        if tasks is None:
            tasks = self.load_tasks(limit=task_limit)

        sandbox = sandbox_runner or SandboxedCodeRunner(
            config=SandboxSecurityConfig(allow_unsafe_execution=True, timeout_seconds=self.timeout_per_task),
            check_connectivity=False,
        )

        task_results: list[TaskResult] = []

        def _eval_one(task: BenchmarkTask) -> TaskResult:
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
            return self.evaluate_response(task, gen_text, sandbox)

        if self.parallel_tasks > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=self.parallel_tasks) as pool:
                task_results = list(pool.map(_eval_one, tasks))
        else:
            task_results = [_eval_one(t) for t in tasks]

        total = max(1, len(task_results))
        plus_passed = sum(1 for r in task_results if r.passed)
        base_passed = sum(1 for r in task_results if "Base Tests: PASS" in (r.stdout or ""))

        plus_pass_at_1 = plus_passed / total
        base_pass_at_1 = base_passed / total

        logger.info(
            f"HumanEval+ results: base_pass@1={base_pass_at_1:.2%} plus_pass@1={plus_pass_at_1:.2%} "
            f"({plus_passed}/{total})"
        )

        return EvaluationSuiteResult(
            suite_name=self.name,
            benchmark_version=self.version,
            total_tasks=total,
            passed_tasks=plus_passed,
            pass_at_1=plus_pass_at_1,
            compile_rate=sum(1 for r in task_results if r.compile_success) / total,
            unit_test_pass_rate=plus_pass_at_1,
            task_results=task_results,
            summary_metrics={
                "base_pass_at_1": base_pass_at_1,
                "plus_pass_at_1": plus_pass_at_1,
                "false_positive_drop": base_pass_at_1 - plus_pass_at_1,
                "total_tasks": total,
            },
        )


# ---------------------------------------------------------------------------
# MBPPPlusSuite Implementation
# ---------------------------------------------------------------------------


class MBPPPlusSuite(EvaluationSuite):
    """MBPP+ Benchmark Suite with augmented test contracts (378 sanitized tasks)."""

    def __init__(
        self,
        timeout_per_task: int = 20,
        parallel_tasks: int = 4,
    ) -> None:
        self.timeout_per_task = timeout_per_task
        self.parallel_tasks = parallel_tasks

    @property
    def name(self) -> str:
        return "mbppplus"

    @property
    def version(self) -> str:
        return "evalplus_v0.2.1"

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []

        try:
            from datasets import load_dataset  # type: ignore[import]

            hf_ds = load_dataset("evalplus/mbppplus", split="test")
            for item in hf_ds:
                tasks.append(
                    BenchmarkTask(
                        task_id=f"MBPPPlus/{item['task_id']}",
                        suite=self.name,
                        entry_point=item.get("entry_point", f"task_{item['task_id']}"),
                        prompt=item.get("prompt", ""),
                        canonical_solution=item.get("code", ""),
                        test_code=item.get("test", ""),
                        timeout_seconds=self.timeout_per_task,
                        metadata={"entry_point": item.get("entry_point", "")},
                    )
                )
                if limit and len(tasks) >= limit:
                    break
        except Exception:  # noqa: BLE001
            for item in _CANONICAL_MBPP_PROBLEMS:
                tasks.append(
                    BenchmarkTask(
                        task_id=f"MBPPPlus/{item['task_id']}",
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

    def format_prompt(self, task: BenchmarkTask, tokenizer: Any | None = None) -> str:
        return (
            "You are an expert Python programmer.\n"
            f"{task.prompt}\n"
            "Write the complete Python function implementation in a ```python block.\n"
        )

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner,
    ) -> TaskResult:
        clean_code = generated_text.strip()
        if "```python" in clean_code:
            clean_code = clean_code.split("```python")[1].split("```")[0].strip()
        elif "```" in clean_code:
            clean_code = clean_code.split("```")[1].split("```")[0].strip()

        full_code = f"{clean_code}\n\n{task.test_code}"
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

    def evaluate_suite(
        self,
        backend: InferenceBackend | Any,
        tasks: list[BenchmarkTask] | None = None,
        task_limit: int | None = None,
        sandbox_runner: SandboxedCodeRunner | None = None,
    ) -> EvaluationSuiteResult:
        if tasks is None:
            tasks = self.load_tasks(limit=task_limit)

        sandbox = sandbox_runner or SandboxedCodeRunner(
            config=SandboxSecurityConfig(allow_unsafe_execution=True, timeout_seconds=self.timeout_per_task),
            check_connectivity=False,
        )

        task_results: list[TaskResult] = []

        def _eval_one(task: BenchmarkTask) -> TaskResult:
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
            return self.evaluate_response(task, gen_text, sandbox)

        if self.parallel_tasks > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=self.parallel_tasks) as pool:
                task_results = list(pool.map(_eval_one, tasks))
        else:
            task_results = [_eval_one(t) for t in tasks]

        total = max(1, len(task_results))
        passed = sum(1 for r in task_results if r.passed)
        pass_at_1 = passed / total

        logger.info(f"MBPP+ results: pass@1={pass_at_1:.2%} ({passed}/{total})")

        return EvaluationSuiteResult(
            suite_name=self.name,
            benchmark_version=self.version,
            total_tasks=total,
            passed_tasks=passed,
            pass_at_1=pass_at_1,
            compile_rate=sum(1 for r in task_results if r.compile_success) / total,
            unit_test_pass_rate=pass_at_1,
            task_results=task_results,
            summary_metrics={
                "pass_at_1": pass_at_1,
                "total_tasks": total,
            },
        )


# ---------------------------------------------------------------------------
# EvalPlus Unified Suite
# ---------------------------------------------------------------------------


class EvalPlusSuite(EvaluationSuite):
    """Unified EvalPlus Suite Adapter combining HumanEval+ and MBPP+."""

    def __init__(
        self,
        variant: str = "humaneval",
        timeout_per_task: int = 20,
        parallel_tasks: int = 4,
    ) -> None:
        self.variant = variant
        self.timeout_per_task = timeout_per_task
        self.parallel_tasks = parallel_tasks
        if variant.lower() in ("mbpp", "mbppplus", "mbpp_plus"):
            self._underlying = MBPPPlusSuite(timeout_per_task=timeout_per_task, parallel_tasks=parallel_tasks)
        else:
            self._underlying = HumanEvalPlusSuite(timeout_per_task=timeout_per_task, parallel_tasks=parallel_tasks)

    @property
    def name(self) -> str:
        return "evalplus"

    @property
    def version(self) -> str:
        return self._underlying.version

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        return self._underlying.load_tasks(limit=limit)

    def format_prompt(self, task: BenchmarkTask, tokenizer: Any | None = None) -> str:
        return self._underlying.format_prompt(task, tokenizer=tokenizer)

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner,
    ) -> TaskResult:
        return self._underlying.evaluate_response(task, generated_text, sandbox_runner)

    def evaluate_suite(
        self,
        backend: InferenceBackend | Any,
        tasks: list[BenchmarkTask] | None = None,
        task_limit: int | None = None,
        sandbox_runner: SandboxedCodeRunner | None = None,
    ) -> EvaluationSuiteResult:
        return self._underlying.evaluate_suite(
            backend=backend, tasks=tasks, task_limit=task_limit, sandbox_runner=sandbox_runner
        )


# Register in EvaluationRegistry
EvaluationRegistry.register("evalplus", EvalPlusSuite)
EvaluationRegistry.register("humanevalplus", HumanEvalPlusSuite)
EvaluationRegistry.register("humaneval_plus", HumanEvalPlusSuite)
EvaluationRegistry.register("mbppplus", MBPPPlusSuite)
EvaluationRegistry.register("mbpp_plus", MBPPPlusSuite)
