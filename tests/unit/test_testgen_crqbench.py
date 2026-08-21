"""Unit tests for P010 — TestGenEval, CRQBench, and SE Composite Score.

Test classes:
  TestTestGenEvalSuite          — TestGenEval tasks, prompt formatting, mutant generation, coverage & kill rate
  TestCRQBenchSuite             — CRQBench tasks, diff prompts, defect precision/recall matching, actionability
  TestSECompositeCalculator     — Weighted SE composite scoring, missing suite re-weighting, production readiness gates
  TestEvaluationRegistrySuite   — Registry discovery for testgeneval, testgen, crqbench, code_review
  TestBenchmarkRunnerSuite      — Integration with BenchmarkRunner for testgeneval and crqbench
"""

from __future__ import annotations

from typing import Any

import pytest

from vipym.evaluation.composite import (
    SECompositeCalculator,
    SECompositeReport,
    compute_se_composite_score,
)
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.runner import BenchmarkRunner
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig
from vipym.evaluation.suites.crqbench import CRQBenchSuite
from vipym.evaluation.suites.testgeneval import (
    TestGenEvalSuite,
    generate_mutants,
)
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
        self.response_generator = response_generator or (lambda req: "def test_ok(): assert True")

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
# TestTestGenEvalSuite
# ============================================================


class TestTestGenEvalSuite:
    def test_init_defaults(self):
        suite = TestGenEvalSuite()
        assert suite.name == "testgeneval"
        assert suite.version == "v1.0"
        assert suite.timeout_per_task == 30

    def test_load_tasks(self):
        suite = TestGenEvalSuite()
        tasks = suite.load_tasks(limit=3)
        assert len(tasks) == 3
        assert tasks[0].suite == "testgeneval"
        assert "source_code" in tasks[0].metadata

    def test_format_prompt(self):
        suite = TestGenEvalSuite()
        task = suite.load_tasks(limit=1)[0]
        prompt = suite.format_prompt(task)
        assert "pytest" in prompt
        assert "clamp" in prompt or task.entry_point in prompt

    def test_generate_mutants(self):
        code = "def check(x):\n    if x < 10 and x == 5:\n        return True\n    return False\n"
        mutants = generate_mutants(code)
        assert len(mutants) >= 3
        mutant_codes = [m[1] for m in mutants]
        assert any(" > 10" in c for c in mutant_codes)
        assert any(" != 5" in c for c in mutant_codes)
        assert any("False" in c for c in mutant_codes)

    def test_evaluate_response_success(self, sandbox_runner):
        suite = TestGenEvalSuite()
        task = suite.load_tasks(limit=1)[0]

        res = suite.evaluate_response(task, task.canonical_solution, sandbox_runner)
        assert isinstance(res, TaskResult)
        assert res.passed is True
        assert res.compile_success is True
        assert res.unit_tests_passed > 0
        assert "Line Coverage" in res.stdout
        assert "Mutation Score" in res.stdout

    def test_evaluate_response_broken_tests(self, sandbox_runner):
        suite = TestGenEvalSuite()
        task = suite.load_tasks(limit=1)[0]

        broken_tests = "def test_fail(): assert False"
        res = suite.evaluate_response(task, broken_tests, sandbox_runner)
        assert res.passed is False
        assert "failed on correct implementation" in res.error_message

    def test_evaluate_response_empty(self, sandbox_runner):
        suite = TestGenEvalSuite()
        task = suite.load_tasks(limit=1)[0]

        res = suite.evaluate_response(task, "", sandbox_runner)
        assert res.passed is False
        assert res.compile_success is False

    def test_evaluate_suite_batch(self, sandbox_runner):
        suite = TestGenEvalSuite()
        tasks = suite.load_tasks(limit=2)
        backend = MockInferenceBackend(response_generator=lambda req: tasks[0].canonical_solution)

        suite_res = suite.evaluate_suite(backend, tasks=tasks, sandbox_runner=sandbox_runner)
        assert isinstance(suite_res, EvaluationSuiteResult)
        assert suite_res.total_tasks == 2
        assert "line_coverage" in suite_res.summary_metrics
        assert "mutation_score" in suite_res.summary_metrics


# ============================================================
# TestCRQBenchSuite
# ============================================================


class TestCRQBenchSuite:
    def test_init_defaults(self):
        suite = CRQBenchSuite()
        assert suite.name == "crqbench"
        assert suite.version == "v1.0"

    def test_load_tasks(self):
        suite = CRQBenchSuite()
        tasks = suite.load_tasks(limit=3)
        assert len(tasks) == 3
        assert tasks[0].suite == "crqbench"
        assert "expert_annotations" in tasks[0].metadata

    def test_format_prompt(self):
        suite = CRQBenchSuite()
        task = suite.load_tasks(limit=1)[0]
        prompt = suite.format_prompt(task)
        assert "code review" in prompt.lower()
        assert task.prompt in prompt

    def test_evaluate_response_detects_defects(self):
        suite = CRQBenchSuite()
        task = suite.load_tasks(limit=1)[0]  # Race condition task

        model_review = (
            "Line 13 introduces a concurrency race condition. The dictionary update is not thread-safe.\n"
            "Suggested fix: Use a threading lock (`with self._lock:`) around data updates."
        )
        res = suite.evaluate_response(task, model_review)
        assert res.passed is True
        assert "Precision: 100.0%" in res.stdout
        assert "Recall: 100.0%" in res.stdout

    def test_evaluate_response_missed_defects(self):
        suite = CRQBenchSuite()
        task = suite.load_tasks(limit=1)[0]

        generic_review = "Looks good to me! Minor formatting comment: add more blank lines."
        res = suite.evaluate_response(task, generic_review)
        assert res.passed is False
        assert "Recall: 0.0%" in res.stdout

    def test_evaluate_response_empty(self):
        suite = CRQBenchSuite()
        task = suite.load_tasks(limit=1)[0]

        res = suite.evaluate_response(task, "")
        assert res.passed is False

    def test_evaluate_suite_batch(self):
        suite = CRQBenchSuite()
        tasks = suite.load_tasks(limit=2)
        backend = MockInferenceBackend(
            response_generator=lambda req: (
                "Race condition hazard. Use a mutex lock. Suggest: with lock: pass"
            )
        )

        suite_res = suite.evaluate_suite(backend, tasks=tasks)
        assert isinstance(suite_res, EvaluationSuiteResult)
        assert "precision" in suite_res.summary_metrics
        assert "recall" in suite_res.summary_metrics
        assert "actionability" in suite_res.summary_metrics


# ============================================================
# TestSECompositeCalculator
# ============================================================


class TestSECompositeCalculator:
    def test_full_composite_calculation(self):
        calc = SECompositeCalculator()
        scores = {
            "swebench": 0.40,  # 0.40 * 0.30 = 0.12
            "aider_edit": 0.80,  # 0.80 * 0.25 = 0.20
            "bigcodebench": 0.60,  # 0.60 * 0.20 = 0.12
            "testgen": 0.70,  # 0.70 * 0.15 = 0.105
            "code_review": 0.50,  # 0.50 * 0.10 = 0.05
        }
        # Total = 0.12 + 0.20 + 0.12 + 0.105 + 0.05 = 0.595
        report = calc.compute(scores)
        assert isinstance(report, SECompositeReport)
        assert report.composite_score == pytest.approx(0.595, abs=1e-3)
        assert len(report.category_scores) == 5

    def test_missing_suites_reweighting(self):
        calc = SECompositeCalculator()
        # Only swebench (0.30) and aider_edit (0.25) provided -> total weight = 0.55
        scores = {
            "swebench": 0.50,
            "aider_edit": 0.80,
        }
        # reweighted: swebench = 0.30/0.55 = 0.5454..., aider = 0.25/0.55 = 0.4545...
        # composite = 0.50 * (30/55) + 0.80 * (25/55) = 0.2727 + 0.3636 = 0.6363...
        report = calc.compute(scores)
        expected = (0.50 * 0.30 + 0.80 * 0.25) / 0.55
        assert report.composite_score == pytest.approx(expected, abs=1e-3)

    def test_production_readiness_pass(self):
        calc = SECompositeCalculator(min_composite_threshold=0.65)
        scores = {
            "swebench": 0.60,
            "aider_edit": 0.90,
            "bigcodebench": 0.75,
            "testgen": {"pass_at_1": 0.80, "line_coverage": 0.75},
            "code_review": {"pass_at_1": 0.60, "precision": 0.55},
        }
        report = calc.compute(scores)
        assert report.is_production_ready is True
        assert all(report.thresholds_passed.values())

    def test_production_readiness_fail_low_composite(self):
        calc = SECompositeCalculator(min_composite_threshold=0.65)
        scores = {
            "swebench": 0.20,
            "aider_edit": 0.30,
            "bigcodebench": 0.30,
            "testgen": {"pass_at_1": 0.30, "line_coverage": 0.65},
            "code_review": {"pass_at_1": 0.30, "precision": 0.50},
        }
        report = calc.compute(scores)
        assert report.is_production_ready is False
        assert report.thresholds_passed["composite_threshold"] is False

    def test_production_readiness_fail_low_coverage(self):
        calc = SECompositeCalculator(min_composite_threshold=0.60, min_testgen_coverage=0.60)
        scores = {
            "swebench": 0.70,
            "aider_edit": 0.85,
            "bigcodebench": 0.70,
            "testgen": {"pass_at_1": 0.70, "line_coverage": 0.45},  # Low coverage (<0.60)
            "code_review": {"pass_at_1": 0.60, "precision": 0.50},
        }
        report = calc.compute(scores)
        assert report.is_production_ready is False
        assert report.thresholds_passed["testgen_coverage"] is False

    def test_production_readiness_fail_low_precision(self):
        calc = SECompositeCalculator(min_composite_threshold=0.60, min_review_precision=0.40)
        scores = {
            "swebench": 0.70,
            "aider_edit": 0.85,
            "bigcodebench": 0.70,
            "testgen": {"pass_at_1": 0.70, "line_coverage": 0.70},
            "code_review": {"pass_at_1": 0.50, "precision": 0.25},  # Low precision (<0.40)
        }
        report = calc.compute(scores)
        assert report.is_production_ready is False
        assert report.thresholds_passed["review_precision"] is False

    def test_relative_baseline_calculation(self):
        calc = SECompositeCalculator()
        student_scores = {"swebench": 0.40, "aider_edit": 0.70}
        teacher_scores = {"swebench": 0.50, "aider_edit": 0.80}

        report = calc.compute(student_scores, baseline_results=teacher_scores)
        assert report.relative_score is not None
        assert report.relative_score > 0.80

    def test_convenience_function(self):
        report = compute_se_composite_score({"swebench": 0.50, "aider_edit": 0.80})
        assert isinstance(report, SECompositeReport)


# ============================================================
# TestEvaluationRegistrySuite
# ============================================================


class TestEvaluationRegistrySuite:
    def test_testgeneval_registered(self):
        suite1 = EvaluationRegistry.get("testgeneval")
        assert isinstance(suite1, TestGenEvalSuite)
        suite2 = EvaluationRegistry.get("testgen")
        assert isinstance(suite2, TestGenEvalSuite)

    def test_crqbench_registered(self):
        suite1 = EvaluationRegistry.get("crqbench")
        assert isinstance(suite1, CRQBenchSuite)
        suite2 = EvaluationRegistry.get("code_review")
        assert isinstance(suite2, CRQBenchSuite)


# ============================================================
# TestBenchmarkRunnerSuite
# ============================================================


class TestBenchmarkRunnerSuite:
    def test_benchmark_runner_runs_testgeneval(self, sandbox_runner):
        runner = BenchmarkRunner(sandbox_runner=sandbox_runner)
        suite = EvaluationRegistry.get("testgeneval")
        first_task = suite.load_tasks(limit=1)[0]
        backend = MockInferenceBackend(response_generator=lambda req: first_task.canonical_solution)

        result = runner.run_suite("testgeneval", backend=backend, task_limit=1)
        assert isinstance(result, EvaluationSuiteResult)
        assert result.suite_name == "testgeneval"
        assert result.total_tasks == 1
        assert result.passed_tasks == 1

    def test_benchmark_runner_runs_crqbench(self, sandbox_runner):
        runner = BenchmarkRunner(sandbox_runner=sandbox_runner)
        backend = MockInferenceBackend(
            response_generator=lambda req: (
                "Race condition concurrency error. Line 13 lock needed. Suggest: with lock: pass"
            )
        )

        result = runner.run_suite("crqbench", backend=backend, task_limit=1)
        assert isinstance(result, EvaluationSuiteResult)
        assert result.suite_name == "crqbench"
        assert result.total_tasks == 1
        assert result.passed_tasks == 1
