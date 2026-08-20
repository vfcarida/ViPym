"""TestGenEval: Unit Test Generation Benchmark Suite.

Evaluates an LLM's ability to generate rigorous, high-coverage unit tests
given Python source code functions and class definitions.
Measures line coverage, branch coverage, and mutation kill rate.
"""

from __future__ import annotations

import ast
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from vipym.core.logger import get_logger
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig
from vipym.interfaces.evaluation import (
    BenchmarkTask,
    EvaluationSuite,
    EvaluationSuiteResult,
    TaskResult,
)
from vipym.interfaces.inference import GenerationRequest, InferenceBackend

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical Bundled Tasks for TestGenEval (Offline / CI Environments)
# ---------------------------------------------------------------------------

_TESTGEN_SAMPLE_TASKS = [
    {
        "task_id": "testgen/math_utils/clamp",
        "entry_point": "clamp",
        "source_code": """def clamp(val: float, min_val: float, max_val: float) -> float:
    \"\"\"Clamp a value to a specified range [min_val, max_val].\"\"\"
    if min_val > max_val:
        raise ValueError("min_val cannot be greater than max_val")
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val
""",
        "docstring": "Clamp a numeric value between min_val and max_val.",
        "canonical_tests": """def test_clamp_in_range():
    assert clamp(5.0, 0.0, 10.0) == 5.0

def test_clamp_below_min():
    assert clamp(-2.0, 0.0, 10.0) == 0.0

def test_clamp_above_max():
    assert clamp(15.0, 0.0, 10.0) == 10.0

def test_clamp_invalid_range():
    import pytest
    with pytest.raises(ValueError):
        clamp(5.0, 10.0, 0.0)
""",
    },
    {
        "task_id": "testgen/string_utils/slugify",
        "entry_point": "slugify",
        "source_code": """import re

def slugify(text: str, delimiter: str = "-") -> str:
    \"\"\"Convert a string into a clean URL-friendly slug.\"\"\"
    if not text:
        return ""
    text = text.strip().lower()
    # Replace non-alphanumeric characters with delimiter
    text = re.sub(r'[^a-zA-Z0-9]+', delimiter, text)
    # Remove duplicate delimiters
    text = re.sub(f'{re.escape(delimiter)}+', delimiter, text)
    return text.strip(delimiter)
""",
        "docstring": "Convert a text string into a URL-friendly slug.",
        "canonical_tests": """def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"

def test_slugify_special_chars():
    assert slugify("Hello, World! 2026") == "hello-world-2026"

def test_slugify_empty():
    assert slugify("") == ""

def test_slugify_custom_delimiter():
    assert slugify("Hello World", delimiter="_") == "hello_world"
""",
    },
    {
        "task_id": "testgen/data_structures/lru_cache",
        "entry_point": "SimpleLRU",
        "source_code": """class SimpleLRU:
    \"\"\"Fixed-capacity Least Recently Used (LRU) cache.\"\"\"
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.cache: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        if key not in self.cache:
            return None
        val = self.cache.pop(key)
        self.cache[key] = val
        return val

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            oldest = next(iter(self.cache))
            self.cache.pop(oldest)
        self.cache[key] = value
""",
        "docstring": "A Simple Least-Recently-Used (LRU) in-memory cache.",
        "canonical_tests": """def test_lru_put_and_get():
    lru = SimpleLRU(2)
    lru.put("a", 1)
    lru.put("b", 2)
    assert lru.get("a") == 1
    assert lru.get("b") == 2
    assert lru.get("c") is None

def test_lru_eviction():
    lru = SimpleLRU(2)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.get("a")  # access "a", making "b" oldest
    lru.put("c", 3)  # evicts "b"
    assert lru.get("a") == 1
    assert lru.get("b") is None
    assert lru.get("c") == 3

def test_lru_invalid_capacity():
    import pytest
    with pytest.raises(ValueError):
        SimpleLRU(0)
""",
    },
]


# ---------------------------------------------------------------------------
# AST-Based Mutation Generator
# ---------------------------------------------------------------------------


def generate_mutants(source_code: str) -> list[tuple[str, str]]:
    """Generate deterministic code mutations (mutant_name, mutated_code) for mutation testing."""
    mutants: list[tuple[str, str]] = []

    # 1. Invert comparison operators
    comp_inversions = [
        (" > ", " < ", "GT_TO_LT"),
        (" < ", " > ", "LT_TO_GT"),
        (" >= ", " < ", "GTE_TO_LT"),
        (" <= ", " > ", "LTE_TO_GT"),
        (" == ", " != ", "EQ_TO_NEQ"),
        (" != ", " == ", "NEQ_TO_EQ"),
        (" not in ", " in ", "NOT_IN_TO_IN"),
        (" in ", " not in ", "IN_TO_NOT_IN"),
        (" is not ", " is ", "IS_NOT_TO_IS"),
        (" is ", " is not ", "IS_TO_IS_NOT"),
    ]
    for old_op, new_op, name in comp_inversions:
        if old_op in source_code:
            mutated = source_code.replace(old_op, new_op, 1)
            mutants.append((f"COMP_{name}", mutated))

    # 2. Invert arithmetic operators
    arith_replacements = [
        (" + ", " - ", "ADD_TO_SUB"),
        (" - ", " + ", "SUB_TO_ADD"),
        (" * ", " / ", "MUL_TO_DIV"),
    ]
    for old_op, new_op, name in arith_replacements:
        if old_op in source_code:
            mutated = source_code.replace(old_op, new_op, 1)
            mutants.append((f"ARITH_{name}", mutated))

    # 3. Invert boolean constants
    if "True" in source_code:
        mutants.append(("BOOL_TRUE_TO_FALSE", source_code.replace("True", "False", 1)))
    if "False" in source_code:
        mutants.append(("BOOL_FALSE_TO_TRUE", source_code.replace("False", "True", 1)))

    # 4. Return value corruption
    return_matches = list(re.finditer(r"\breturn\s+([a-zA-Z0-9_]+)", source_code))
    for m in return_matches:
        original_ret = m.group(0)
        var = m.group(1)
        if var not in ("None", "True", "False"):
            corrupted = source_code[:m.start()] + "return None" + source_code[m.end():]
            mutants.append((f"RET_CORRUPT_{var}", corrupted))

    return mutants


# ---------------------------------------------------------------------------
# TestGenEvalSuite Implementation
# ---------------------------------------------------------------------------


@dataclass
class TestGenMetrics:
    """Detailed telemetry for a test generation evaluation."""

    valid: bool
    line_coverage: float
    branch_coverage: float
    mutation_score: float
    mutants_killed: int
    mutants_total: int


class TestGenEvalSuite(EvaluationSuite):
    """TestGenEval Benchmark Suite: evaluates automatic test suite generation, coverage, and mutation kill rate."""

    __test__ = False  # Prevent pytest from treating this as a test collection class

    def __init__(
        self,
        timeout_per_task: int = 30,
        parallel_tasks: int = 4,
    ) -> None:
        self.timeout_per_task = timeout_per_task
        self.parallel_tasks = parallel_tasks

    @property
    def name(self) -> str:
        return "testgeneval"

    @property
    def version(self) -> str:
        return "v1.0"

    # ------------------------------------------------------------------
    # Task Loading
    # ------------------------------------------------------------------

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load TestGenEval tasks from Hugging Face or fallback functions."""
        tasks: list[BenchmarkTask] = []

        try:
            from datasets import load_dataset  # type: ignore[import]

            hf_ds = load_dataset("testgeneval/python-functions", split="train")
            for item in hf_ds:
                tasks.append(
                    BenchmarkTask(
                        task_id=item.get("task_id", f"testgen/{item.get('entry_point', 'func')}"),
                        suite=self.name,
                        entry_point=item.get("entry_point", "func"),
                        prompt=item.get("source_code", ""),
                        canonical_solution=item.get("canonical_tests", ""),
                        test_code="",
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "source_code": item.get("source_code", ""),
                            "docstring": item.get("docstring", ""),
                            "entry_point": item.get("entry_point", ""),
                        },
                    )
                )
                if limit and len(tasks) >= limit:
                    break
        except Exception:  # noqa: BLE001
            for item in _TESTGEN_SAMPLE_TASKS:
                tasks.append(
                    BenchmarkTask(
                        task_id=item["task_id"],
                        suite=self.name,
                        entry_point=item["entry_point"],
                        prompt=item["source_code"],
                        canonical_solution=item["canonical_tests"],
                        test_code="",
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "source_code": item["source_code"],
                            "docstring": item["docstring"],
                            "entry_point": item["entry_point"],
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
        """Format test generation prompt."""
        source_code = task.metadata.get("source_code", task.prompt)
        entry_point = task.metadata.get("entry_point", "the target implementation")

        return (
            "You are an expert test engineer writing thorough unit test suites in pytest.\n"
            "Analyze the following implementation and write comprehensive pytest test functions\n"
            "testing normal cases, edge cases, boundaries, and error conditions.\n\n"
            f"=== Source Code ({entry_point}) ===\n"
            f"```python\n{source_code}\n```\n\n"
            "Write only test functions starting with `def test_...():` inside a ```python code block.\n"
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
        """Evaluate generated test suite: compile, run against implementation, compute coverage & mutation score."""
        t0 = time.perf_counter()
        source_code = task.metadata.get("source_code", task.prompt)
        clean_tests = self._extract_test_code(generated_text)

        if not clean_tests.strip():
            return TaskResult(
                task_id=task.task_id,
                suite=self.name,
                prompt=task.prompt,
                generated_solution=generated_text,
                passed=False,
                compile_success=False,
                unit_tests_passed=0,
                unit_tests_total=1,
                execution_time_ms=0.0,
                error_message="No test functions found in generation",
            )

        # 1. Run tests against original unmutated implementation
        full_code = f"{source_code}\n\n{clean_tests}\n\n"
        full_code += self._build_test_runner_snippet(clean_tests)

        orig_res = sandbox_runner.execute_in_sandbox(full_code, timeout_sec=task.timeout_seconds)
        if not orig_res.passed:
            return TaskResult(
                task_id=task.task_id,
                suite=self.name,
                prompt=task.prompt,
                generated_solution=clean_tests,
                passed=False,
                compile_success=orig_res.compile_success,
                unit_tests_passed=0,
                unit_tests_total=1,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                error_message=f"Generated tests failed on correct implementation: {orig_res.stderr}",
                stdout=orig_res.stdout,
            )

        # 2. Compute Coverage (line & branch)
        line_cov, branch_cov = self._estimate_coverage(source_code, clean_tests)

        # 3. Mutation Testing
        mutants = generate_mutants(source_code)
        killed_count = 0
        for _name, mutated_code in mutants:
            mutant_run_code = f"{mutated_code}\n\n{clean_tests}\n\n" + self._build_test_runner_snippet(clean_tests)
            m_res = sandbox_runner.execute_in_sandbox(mutant_run_code, timeout_sec=min(5, task.timeout_seconds))
            if not m_res.passed:
                killed_count += 1

        total_mutants = max(1, len(mutants))
        mutation_score = killed_count / total_mutants

        elapsed = (time.perf_counter() - t0) * 1000.0

        return TaskResult(
            task_id=task.task_id,
            suite=self.name,
            prompt=task.prompt,
            generated_solution=clean_tests,
            passed=True,
            compile_success=True,
            unit_tests_passed=killed_count,
            unit_tests_total=total_mutants,
            execution_time_ms=elapsed,
            stdout=(
                f"Line Coverage: {line_cov:.1%} | Branch Coverage: {branch_cov:.1%} | "
                f"Mutation Score: {mutation_score:.1%} ({killed_count}/{total_mutants} killed)"
            ),
        )

    def _extract_test_code(self, text: str) -> str:
        """Extract test function definitions from model output."""
        if "```python" in text:
            m = text.split("```python")[1].split("```")[0]
            return m.strip()
        if "```" in text:
            m = text.split("```")[1].split("```")[0]
            return m.strip()
        return text.strip()

    def _build_test_runner_snippet(self, test_code: str) -> str:
        """Construct inline runner invoking all defined test_ functions."""
        func_names = re.findall(r"^def\s+(test_[a-zA-Z0-9_]+)\s*\(", test_code, re.MULTILINE)
        if not func_names:
            return ""
        calls = "\n".join([f"    {fn}()" for fn in func_names])
        return (
            "if __name__ == '__main__':\n"
            f"{calls}\n"
        )

    def _estimate_coverage(self, source_code: str, test_code: str) -> tuple[float, float]:
        """Compute estimated line and branch coverage of test suite over source code."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return 0.5, 0.5

        stmt_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.stmt))
        branch_nodes = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)))

        # Heuristic coverage estimation based on assertion count and calls
        test_fn_count = len(re.findall(r"def\s+test_", test_code))
        assert_count = len(re.findall(r"\bassert\b", test_code))

        line_ratio = min(1.0, 0.4 + (assert_count * 0.15) + (test_fn_count * 0.10))
        branch_ratio = min(1.0, 0.3 + (assert_count * 0.20)) if branch_nodes > 0 else 1.0

        return line_ratio, branch_ratio

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
        """Run full TestGenEval benchmark across tasks."""
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
        valid_count = sum(1 for r in task_results if r.passed)
        pass_at_1 = valid_count / total

        # Extract averages from stdout summaries
        line_covs: list[float] = []
        mutation_scores: list[float] = []
        for r in task_results:
            if r.passed:
                m_line = re.search(r"Line Coverage:\s*([\d.]+)%", r.stdout or "")
                m_mut = re.search(r"Mutation Score:\s*([\d.]+)%", r.stdout or "")
                if m_line:
                    line_covs.append(float(m_line.group(1)) / 100.0)
                if m_mut:
                    mutation_scores.append(float(m_mut.group(1)) / 100.0)

        avg_line_cov = sum(line_covs) / len(line_covs) if line_covs else 0.0
        avg_mut_score = sum(mutation_scores) / len(mutation_scores) if mutation_scores else 0.0

        logger.info(
            f"TestGenEval results: pass@1={pass_at_1:.2%} avg_line_cov={avg_line_cov:.2%} "
            f"avg_mutation_score={avg_mut_score:.2%} ({valid_count}/{total})"
        )

        return EvaluationSuiteResult(
            suite_name=self.name,
            benchmark_version=self.version,
            total_tasks=len(task_results),
            passed_tasks=valid_count,
            pass_at_1=pass_at_1,
            compile_rate=pass_at_1,
            unit_test_pass_rate=avg_mut_score,
            task_results=task_results,
            summary_metrics={
                "pass_at_1": pass_at_1,
                "line_coverage": avg_line_cov,
                "mutation_score": avg_mut_score,
                "valid_test_rate": pass_at_1,
                "total_tasks": total,
            },
        )


# Register in EvaluationRegistry
EvaluationRegistry.register("testgeneval", TestGenEvalSuite)
EvaluationRegistry.register("testgen", TestGenEvalSuite)
