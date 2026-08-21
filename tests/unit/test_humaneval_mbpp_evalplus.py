"""Unit tests for P011 — HumanEval, MBPP, EvalPlus (HumanEval+, MBPP+), and pass@k scoring.

Test classes:
  TestPassAtKScoring            — Exact combinatorial pass@k math, edge cases, multi-sample aggregation
  TestHumanEvalSuite            — HumanEval task loading, code cleaning, execution, pass@1/10/100
  TestMBPPSuite                 — MBPP sanitized tasks, prompt formatting, assertion execution
  TestEvalPlusSuites            — HumanEval+, MBPP+, false-positive detection, enhanced contract testing
  TestEvaluationRegistrySuite   — Registry discovery for humaneval, mbpp, evalplus, and aliases
  TestBenchmarkRunnerSuite      — Integration with BenchmarkRunner
"""

from __future__ import annotations

from typing import Any

import pytest

from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.runner import BenchmarkRunner
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig
from vipym.evaluation.scoring import (
    calculate_pass_at_k_metrics,
    compute_pass_at_k,
)
from vipym.evaluation.suites.evalplus import (
    EvalPlusSuite,
    HumanEvalPlusSuite,
    MBPPPlusSuite,
)
from vipym.evaluation.suites.humaneval import HumanEvalSuite
from vipym.evaluation.suites.mbpp import LiveCodeBenchSuite, MBPPSuite
from vipym.interfaces.evaluation import EvaluationSuiteResult, TaskResult
from vipym.interfaces.inference import GenerationRequest, GenerationResponse, InferenceBackend

# ============================================================
# Fixtures & Mocks
# ============================================================


@pytest.fixture(autouse=True)
def setup_unsafe_sandbox(monkeypatch):
    monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")


@pytest.fixture
def sandbox_runner():
    return SandboxedCodeRunner(
        config=SandboxSecurityConfig(allow_unsafe_execution=True),
        check_connectivity=False,
    )


class MockInferenceBackend(InferenceBackend):
    def __init__(self, response_generator=None) -> None:
        self.response_generator = response_generator or (lambda req: "def solution(): pass")

    def start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        text = (
            self.response_generator(request)
            if callable(self.response_generator)
            else str(self.response_generator)
        )
        return GenerationResponse(
            generated_text=text,
            prompt_tokens=len(request.prompt) // 4,
            completion_tokens=len(text) // 4,
            time_to_first_token_ms=5.0,
            inter_token_latency_ms=1.0,
            total_time_ms=15.0,
        )

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        return self.generate(request)

    def stop(self) -> None:
        pass


# ============================================================
# TestPassAtKScoring
# ============================================================


class TestPassAtKScoring:
    def test_single_sample_exact(self):
        # 1 sample, correct -> pass@1 = 1.0
        assert compute_pass_at_k(n=1, c=1, k=1) == 1.0
        # 1 sample, failed -> pass@1 = 0.0
        assert compute_pass_at_k(n=1, c=0, k=1) == 0.0

    def test_combinatorial_exact_math(self):
        # n=10, c=5, k=1 -> pass@1 = 5/10 = 0.5
        assert compute_pass_at_k(n=10, c=5, k=1) == pytest.approx(0.5, abs=1e-5)
        # n=10, c=10, k=5 -> pass@5 = 1.0
        assert compute_pass_at_k(n=10, c=10, k=5) == 1.0
        # n=10, c=2, k=1 -> pass@1 = 0.2
        assert compute_pass_at_k(n=10, c=2, k=1) == pytest.approx(0.2, abs=1e-5)

    def test_edge_cases(self):
        assert compute_pass_at_k(n=0, c=0, k=1) == 0.0
        assert compute_pass_at_k(n=5, c=0, k=1) == 0.0
        assert compute_pass_at_k(n=5, c=5, k=0) == 0.0
        # n - c < k (e.g. 5 - 4 = 1 < 2) -> pass@2 = 1.0
        assert compute_pass_at_k(n=5, c=4, k=2) == 1.0

    def test_list_input_averaging(self):
        # Problem 1: n=10, c=5 (pass@1=0.5), Problem 2: n=10, c=10 (pass@1=1.0)
        # Mean = 0.75
        avg_score = compute_pass_at_k(n=[10, 10], c=[5, 10], k=1)
        assert avg_score == pytest.approx(0.75, abs=1e-5)

    def test_calculate_pass_at_k_metrics(self):
        # Task 1: 2 samples, 1 passed [True, False]
        # Task 2: 2 samples, 2 passed [True, True]
        correctness = [
            [True, False],
            [True, True],
        ]
        metrics = calculate_pass_at_k_metrics(correctness, k_values=[1, 2])
        # Task 1 pass@1 = 0.5, Task 2 pass@1 = 1.0 -> avg = 0.75
        assert metrics["pass@1"] == pytest.approx(0.75, abs=1e-5)
        # Task 1 pass@2 = 1.0, Task 2 pass@2 = 1.0 -> avg = 1.0
        assert metrics["pass@2"] == pytest.approx(1.0, abs=1e-5)


# ============================================================
# TestHumanEvalSuite
# ============================================================


class TestHumanEvalSuite:
    def test_init_and_defaults(self):
        suite = HumanEvalSuite()
        assert suite.name == "humaneval"
        assert suite.version == "v1.0.0"
        assert suite.timeout_per_task == 15

    def test_load_tasks(self):
        suite = HumanEvalSuite()
        tasks = suite.load_tasks(limit=3)
        assert len(tasks) == 3
        assert tasks[0].suite == "humaneval"
        assert tasks[0].entry_point == "has_close_elements"

    def test_format_prompt(self):
        suite = HumanEvalSuite()
        task = suite.load_tasks(limit=1)[0]
        prompt = suite.format_prompt(task)
        assert "def has_close_elements" in prompt

    def test_clean_code_fences_and_raw(self):
        suite = HumanEvalSuite()
        task = suite.load_tasks(limit=1)[0]

        # Markdown fence
        fence = "```python\ndef has_close_elements(numbers, threshold):\n    return False\n```"
        assert suite._clean_code(task, fence).startswith("def has_close_elements")

        # Function body only (should prepend prompt header)
        body = "    return False"
        cleaned = suite._clean_code(task, body)
        assert "def has_close_elements" in cleaned

    def test_evaluate_response_success(self, sandbox_runner):
        suite = HumanEvalSuite()
        task = suite.load_tasks(limit=1)[0]

        res = suite.evaluate_response(task, task.canonical_solution, sandbox_runner)
        assert isinstance(res, TaskResult)
        assert res.passed is True
        assert res.compile_success is True

    def test_evaluate_response_failure(self, sandbox_runner):
        suite = HumanEvalSuite()
        task = suite.load_tasks(limit=1)[0]

        broken_code = "def has_close_elements(numbers, threshold): return None"
        res = suite.evaluate_response(task, broken_code, sandbox_runner)
        assert res.passed is False
        assert res.error_message is not None

    def test_evaluate_suite_batch(self, sandbox_runner):
        suite = HumanEvalSuite()
        tasks = suite.load_tasks(limit=2)
        backend = MockInferenceBackend(response_generator=lambda req: tasks[0].canonical_solution)

        suite_res = suite.evaluate_suite(backend, tasks=tasks, sandbox_runner=sandbox_runner)
        assert isinstance(suite_res, EvaluationSuiteResult)
        assert suite_res.total_tasks == 2
        assert "pass@1" in suite_res.summary_metrics


# ============================================================
# TestMBPPSuite
# ============================================================


class TestMBPPSuite:
    def test_init_and_defaults(self):
        suite = MBPPSuite()
        assert suite.name == "mbpp"
        assert "sanitized" in suite.version

    def test_load_tasks(self):
        suite = MBPPSuite()
        tasks = suite.load_tasks(limit=3)
        assert len(tasks) == 3
        assert tasks[0].suite == "mbpp"
        assert tasks[0].entry_point is not None

    def test_format_prompt(self):
        suite = MBPPSuite()
        task = suite.load_tasks(limit=1)[0]
        prompt = suite.format_prompt(task)
        assert "expert Python programmer" in prompt
        assert task.prompt in prompt

    def test_evaluate_response_success(self, sandbox_runner):
        suite = MBPPSuite()
        task = suite.load_tasks(limit=1)[0]

        res = suite.evaluate_response(task, task.canonical_solution, sandbox_runner)
        assert res.passed is True
        assert res.compile_success is True

    def test_evaluate_response_failure(self, sandbox_runner):
        suite = MBPPSuite()
        task = suite.load_tasks(limit=1)[0]

        broken = "def broken(): return 0"
        res = suite.evaluate_response(task, broken, sandbox_runner)
        assert res.passed is False

    def test_evaluate_suite_batch(self, sandbox_runner):
        suite = MBPPSuite()
        tasks = suite.load_tasks(limit=2)
        backend = MockInferenceBackend(response_generator=lambda req: tasks[0].canonical_solution)

        suite_res = suite.evaluate_suite(backend, tasks=tasks, sandbox_runner=sandbox_runner)
        assert isinstance(suite_res, EvaluationSuiteResult)
        assert "pass@1" in suite_res.summary_metrics

    def test_livecodebench_suite(self, sandbox_runner):
        suite = LiveCodeBenchSuite()
        assert suite.name == "livecodebench"
        tasks = suite.load_tasks(limit=1)
        res = suite.evaluate_response(tasks[0], "def solve(): pass", sandbox_runner)
        assert res.passed is True


# ============================================================
# TestEvalPlusSuites
# ============================================================


class TestEvalPlusSuites:
    def test_humanevalplus_load_and_evaluate_success(self, sandbox_runner):
        suite = HumanEvalPlusSuite()
        tasks = suite.load_tasks(limit=1)
        assert len(tasks) == 1

        res = suite.evaluate_response(tasks[0], tasks[0].canonical_solution, sandbox_runner)
        assert res.passed is True
        assert "EvalPlus Tests: PASS" in res.stdout

    def test_humanevalplus_catches_false_positive(self, sandbox_runner):
        suite = HumanEvalPlusSuite()
        task = suite.load_tasks(limit=1)[0]

        # Brittle solution: passes all base tests (len >= 5) but fails on short lists in EvalPlus
        brittle_solution = """def has_close_elements(numbers, threshold):
    if len(numbers) < 3:
        return True
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2 and abs(elem - elem2) < threshold:
                return True
    return False
"""
        res = suite.evaluate_response(task, brittle_solution, sandbox_runner)
        assert res.passed is False
        assert "Base Tests: PASS" in res.stdout
        assert "EvalPlus Tests: FAIL" in res.stdout

    def test_mbppplus_load_and_evaluate_success(self, sandbox_runner):
        suite = MBPPPlusSuite()
        tasks = suite.load_tasks(limit=1)
        assert len(tasks) == 1

        res = suite.evaluate_response(tasks[0], tasks[0].canonical_solution, sandbox_runner)
        assert res.passed is True

    def test_evalplus_unified_wrapper(self):
        suite_he = EvalPlusSuite(variant="humaneval")
        assert suite_he.name == "evalplus"
        assert isinstance(suite_he._underlying, HumanEvalPlusSuite)

        suite_mbpp = EvalPlusSuite(variant="mbpp")
        assert isinstance(suite_mbpp._underlying, MBPPPlusSuite)


# ============================================================
# TestEvaluationRegistrySuite
# ============================================================


class TestEvaluationRegistrySuite:
    def test_humaneval_registered(self):
        assert isinstance(EvaluationRegistry.get("humaneval"), HumanEvalSuite)
        assert isinstance(EvaluationRegistry.get("human_eval"), HumanEvalSuite)

    def test_mbpp_registered(self):
        assert isinstance(EvaluationRegistry.get("mbpp"), MBPPSuite)
        assert isinstance(EvaluationRegistry.get("mbpp_sanitized"), MBPPSuite)

    def test_evalplus_registered(self):
        assert isinstance(EvaluationRegistry.get("evalplus"), EvalPlusSuite)
        assert isinstance(EvaluationRegistry.get("humanevalplus"), HumanEvalPlusSuite)
        assert isinstance(EvaluationRegistry.get("humaneval_plus"), HumanEvalPlusSuite)
        assert isinstance(EvaluationRegistry.get("mbppplus"), MBPPPlusSuite)
        assert isinstance(EvaluationRegistry.get("mbpp_plus"), MBPPPlusSuite)


# ============================================================
# TestBenchmarkRunnerSuite
# ============================================================


class TestBenchmarkRunnerSuite:
    def test_benchmark_runner_runs_humaneval(self, sandbox_runner):
        runner = BenchmarkRunner(sandbox_runner=sandbox_runner)
        suite = EvaluationRegistry.get("humaneval")
        first_task = suite.load_tasks(limit=1)[0]
        backend = MockInferenceBackend(response_generator=lambda req: first_task.canonical_solution)

        result = runner.run_suite("humaneval", backend=backend, task_limit=1)
        assert isinstance(result, EvaluationSuiteResult)
        assert result.suite_name == "humaneval"
        assert result.passed_tasks == 1

    def test_benchmark_runner_runs_mbpp(self, sandbox_runner):
        runner = BenchmarkRunner(sandbox_runner=sandbox_runner)
        suite = EvaluationRegistry.get("mbpp")
        first_task = suite.load_tasks(limit=1)[0]
        backend = MockInferenceBackend(response_generator=lambda req: first_task.canonical_solution)

        result = runner.run_suite("mbpp", backend=backend, task_limit=1)
        assert isinstance(result, EvaluationSuiteResult)
        assert result.suite_name == "mbpp"
        assert result.passed_tasks == 1

    def test_benchmark_runner_runs_humanevalplus(self, sandbox_runner):
        runner = BenchmarkRunner(sandbox_runner=sandbox_runner)
        suite = EvaluationRegistry.get("humanevalplus")
        first_task = suite.load_tasks(limit=1)[0]
        backend = MockInferenceBackend(response_generator=lambda req: first_task.canonical_solution)

        result = runner.run_suite("humanevalplus", backend=backend, task_limit=1)
        assert isinstance(result, EvaluationSuiteResult)
        assert result.suite_name == "humanevalplus"
        assert result.passed_tasks == 1
