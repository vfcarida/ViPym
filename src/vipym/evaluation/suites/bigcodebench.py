"""BigCodeBench Evaluation Suite.

Evaluates foundational and compressed LLMs on 1,140 practical Python coding tasks
with real-world library usage (pandas, numpy, scikit-learn, requests, scipy, etc.).
Measures pass@1, compile rate, and library coverage (% of required libraries utilized).
"""

from __future__ import annotations

import ast
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from vipym.core.logger import get_logger
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.interfaces.evaluation import (
    BenchmarkTask,
    EvaluationSuite,
    EvaluationSuiteResult,
    TaskResult,
)
from vipym.interfaces.inference import GenerationRequest, InferenceBackend

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Hugging Face Dataset Mappings
# ---------------------------------------------------------------------------

_BIGCODEBENCH_HF_DATASET = "bigcode/bigcodebench"
_BIGCODEBENCH_HARD_DATASET = "bigcode/bigcodebench-hard"

# ---------------------------------------------------------------------------
# Canonical Bundled Tasks for BigCodeBench (for offline / CI environments)
# ---------------------------------------------------------------------------

_BIGCODEBENCH_SAMPLE_TASKS = [
    {
        "task_id": "BigCodeBench/0",
        "entry_point": "task_func",
        "prompt": """def task_func(numbers: list[int | float]) -> dict[str, float]:
    \"\"\"
    Compute statistical summary (mean, std, median) of a numeric list using numpy.

    Parameters:
        numbers (list): A list of numbers.

    Returns:
        dict: {'mean': float, 'std': float, 'median': float}
    \"\"\"
""",
        "test_code": """
import numpy as np

res = task_func([1, 2, 3, 4, 5])
assert np.isclose(res['mean'], 3.0)
assert np.isclose(res['median'], 3.0)
assert np.isclose(res['std'], np.std([1, 2, 3, 4, 5]))
""",
        "canonical_solution": """import numpy as np

def task_func(numbers: list[int | float]) -> dict[str, float]:
    arr = np.array(numbers)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
    }
""",
        "libs": ["numpy"],
    },
    {
        "task_id": "BigCodeBench/1",
        "entry_point": "filter_dataframe",
        "prompt": """import pandas as pd

def filter_dataframe(df: pd.DataFrame, column_name: str, threshold: float) -> pd.DataFrame:
    \"\"\"
    Filter a pandas DataFrame to rows where the specified column is greater than threshold.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        column_name (str): Column to filter by.
        threshold (float): Minimum value (exclusive).

    Returns:
        pd.DataFrame: Filtered dataframe.
    \"\"\"
""",
        "test_code": """
import pandas as pd

df = pd.DataFrame({'a': [10, 20, 30], 'b': [1, 2, 3]})
filtered = filter_dataframe(df, 'a', 15)
assert len(filtered) == 2
assert list(filtered['a']) == [20, 30]
""",
        "canonical_solution": """import pandas as pd

def filter_dataframe(df: pd.DataFrame, column_name: str, threshold: float) -> pd.DataFrame:
    return df[df[column_name] > threshold]
""",
        "libs": ["pandas"],
    },
    {
        "task_id": "BigCodeBench/2",
        "entry_point": "fit_linear_model",
        "prompt": """import numpy as np

def fit_linear_model(x: list[float], y: list[float]) -> tuple[float, float]:
    \"\"\"
    Fit a simple 1D linear regression y = m * x + c using numpy.polyfit.

    Parameters:
        x (list): Feature values.
        y (list): Target values.

    Returns:
        tuple: (slope m, intercept c)
    \"\"\"
""",
        "test_code": """
import numpy as np

m, c = fit_linear_model([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
assert np.isclose(m, 2.0)
assert np.isclose(c, 1.0)
""",
        "canonical_solution": """import numpy as np

def fit_linear_model(x: list[float], y: list[float]) -> tuple[float, float]:
    poly = np.polyfit(x, y, 1)
    return float(poly[0]), float(poly[1])
""",
        "libs": ["numpy"],
    },
]


# ---------------------------------------------------------------------------
# AST Library Coverage Inspection
# ---------------------------------------------------------------------------


def compute_library_coverage(code: str, expected_libs: list[str]) -> float:
    """Compute the fraction of expected third-party libraries imported or referenced."""
    if not expected_libs:
        return 1.0

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback to string matching if AST parse fails
        found = sum(1 for lib in expected_libs if lib.lower() in code.lower())
        return found / len(expected_libs)

    imported_modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0].lower())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0].lower())
        elif isinstance(node, ast.Name):
            # Check for common library aliases (np, pd, plt, sns, tf, torch, etc.)
            alias_map = {
                "np": "numpy",
                "pd": "pandas",
                "plt": "matplotlib",
                "sns": "seaborn",
                "sklearn": "scikit-learn",
            }
            if node.id in alias_map:
                imported_modules.add(alias_map[node.id])

    matched_count = 0
    for expected in expected_libs:
        exp_clean = expected.lower().replace("-", "_")
        if any(imp.replace("-", "_") == exp_clean or exp_clean in imp for imp in imported_modules):
            matched_count += 1
        elif expected.lower() in code.lower():
            matched_count += 1

    return matched_count / len(expected_libs)


# ---------------------------------------------------------------------------
# BigCodeBenchSuite Implementation
# ---------------------------------------------------------------------------


class BigCodeBenchSuite(EvaluationSuite):
    """BigCodeBench Evaluation Suite: 1,140 practical Python coding tasks with complex libraries."""

    def __init__(
        self,
        variant: Literal["full", "hard", "lite"] = "full",
        timeout_per_task: int = 60,
        parallel_tasks: int = 4,
    ) -> None:
        self.variant = variant.lower()
        self.timeout_per_task = timeout_per_task
        self.parallel_tasks = parallel_tasks

    @property
    def name(self) -> str:
        return "bigcodebench"

    @property
    def version(self) -> str:
        return f"{self.variant}_v0.1.2"

    # ------------------------------------------------------------------
    # Task Loading
    # ------------------------------------------------------------------

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load BigCodeBench tasks from Hugging Face or fallback collection."""
        tasks: list[BenchmarkTask] = []
        dataset_name = _BIGCODEBENCH_HARD_DATASET if self.variant == "hard" else _BIGCODEBENCH_HF_DATASET

        try:
            from datasets import load_dataset  # type: ignore[import]

            hf_ds = load_dataset(dataset_name, split="v0.1.2" if self.variant != "hard" else "train")
            for item in hf_ds:
                libs_val = item.get("libs", [])
                if isinstance(libs_val, str):
                    try:
                        libs_list = json.loads(libs_val)
                    except json.JSONDecodeError:
                        libs_list = [l.strip() for l in libs_val.split(",") if l.strip()]
                else:
                    libs_list = list(libs_val)

                tasks.append(
                    BenchmarkTask(
                        task_id=item.get("task_id", "BigCodeBench/task"),
                        suite=self.name,
                        entry_point=item.get("entry_point", "task_func"),
                        prompt=item.get("complete_prompt", item.get("prompt", "")),
                        canonical_solution=item.get("canonical_solution", ""),
                        test_code=item.get("test", ""),
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "libs": libs_list,
                            "variant": self.variant,
                        },
                    )
                )
                if limit and len(tasks) >= limit:
                    break
        except Exception:  # noqa: BLE001
            for item in _BIGCODEBENCH_SAMPLE_TASKS:
                tasks.append(
                    BenchmarkTask(
                        task_id=item["task_id"],
                        suite=self.name,
                        entry_point=item["entry_point"],
                        prompt=item["prompt"],
                        canonical_solution=item["canonical_solution"],
                        test_code=item["test_code"],
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "libs": item.get("libs", []),
                            "variant": self.variant,
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
        """Format Python coding task prompt with type annotations and docstrings."""
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
        """Execute generated function implementation in sandbox and verify assertions."""
        # Clean generation (strip markdown wrapper if model output was enclosed)
        clean_code = generated_text
        if "```python" in generated_text:
            m = generated_text.split("```python")[1].split("```")[0]
            clean_code = m
        elif "```" in generated_text:
            m = generated_text.split("```")[1].split("```")[0]
            clean_code = m

        test_harness = task.test_code
        if "unittest.TestCase" in test_harness and "unittest.main" not in test_harness:
            test_harness += "\nimport unittest\nsuite = unittest.TestLoader().loadTestsFromTestCase(TestCases)\nrunner = unittest.TextTestRunner(verbosity=0)\nres = runner.run(suite)\nassert res.wasSuccessful(), f'Unit tests failed: {len(res.failures)} failures, {len(res.errors)} errors'\n"

        full_code = f"{task.prompt}\n{clean_code}\n{test_harness}"
        res = sandbox_runner.execute_in_sandbox(full_code, timeout_sec=task.timeout_seconds)

        # Compute library coverage
        expected_libs = task.metadata.get("libs", [])
        lib_cov = compute_library_coverage(clean_code, expected_libs)

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
            stdout=f"Library coverage: {lib_cov:.1%} ({expected_libs})",
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
        """Run full BigCodeBench evaluation suite against backend."""
        if tasks is None:
            tasks = self.load_tasks(limit=task_limit)

        from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig

        sandbox = sandbox_runner or SandboxedCodeRunner(
            config=SandboxSecurityConfig(allow_unsafe_execution=True, timeout_seconds=self.timeout_per_task),
            check_connectivity=False,
        )
        task_results: list[TaskResult] = []
        coverage_scores: list[float] = []

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

            expected_libs = task.metadata.get("libs", [])
            cov = compute_library_coverage(gen_text, expected_libs)
            coverage_scores.append(cov)

            return self.evaluate_response(task, gen_text, sandbox)

        if self.parallel_tasks > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=self.parallel_tasks) as pool:
                task_results = list(pool.map(_eval_one, tasks))
        else:
            task_results = [_eval_one(t) for t in tasks]

        total = max(1, len(task_results))
        passed_count = sum(1 for r in task_results if r.passed)
        compile_count = sum(1 for r in task_results if r.compile_success)

        pass_at_1 = passed_count / total
        compile_rate = compile_count / total
        avg_lib_cov = (sum(coverage_scores) / len(coverage_scores)) if coverage_scores else 1.0

        logger.info(
            f"BigCodeBench ({self.version}) results: pass@1={pass_at_1:.2%} "
            f"compile_rate={compile_rate:.2%} library_coverage={avg_lib_cov:.2%} ({passed_count}/{total})"
        )

        return EvaluationSuiteResult(
            suite_name=self.name,
            benchmark_version=self.version,
            total_tasks=len(task_results),
            passed_tasks=passed_count,
            pass_at_1=pass_at_1,
            compile_rate=compile_rate,
            unit_test_pass_rate=pass_at_1,
            task_results=task_results,
            summary_metrics={
                "pass_at_1": pass_at_1,
                "compile_rate": compile_rate,
                "library_coverage": avg_lib_cov,
                "variant": self.variant,
                "total_tasks": total,
            },
        )


# ---------------------------------------------------------------------------
# Specialized Variant Subclasses & Registry Registration
# ---------------------------------------------------------------------------


class BigCodeBenchFullSuite(BigCodeBenchSuite):
    """BigCodeBench Full (1,140 tasks)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(variant="full", **kwargs)


class BigCodeBenchHardSuite(BigCodeBenchSuite):
    """BigCodeBench Hard (148 challenging tasks)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(variant="hard", **kwargs)


class BigCodeBenchLiteSuite(BigCodeBenchSuite):
    """BigCodeBench Lite (fast screening subset)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(variant="lite", **kwargs)


# Register in EvaluationRegistry
EvaluationRegistry.register("bigcodebench", BigCodeBenchSuite)
EvaluationRegistry.register("bigcodebench_full", BigCodeBenchFullSuite)
EvaluationRegistry.register("bigcodebench_hard", BigCodeBenchHardSuite)
EvaluationRegistry.register("bigcodebench_lite", BigCodeBenchLiteSuite)
