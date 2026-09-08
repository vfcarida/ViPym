"""Unit tests for LiveCodeBenchSuite verifying benchmark honesty and test harnesses."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig
from vipym.evaluation.suites.livecodebench import LiveCodeBenchSuite
from vipym.interfaces.evaluation import BenchmarkTask, EvaluationSuite


@pytest.fixture(autouse=True)
def setup_unsafe_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")

    def _mock_load_dataset(*args: object, **kwargs: object) -> None:
        raise ConnectionError("Offline test")

    try:
        import datasets  # type: ignore[import]

        monkeypatch.setattr(datasets, "load_dataset", _mock_load_dataset)
    except ImportError:
        pass


@pytest.fixture
def sandbox_runner() -> SandboxedCodeRunner:
    return SandboxedCodeRunner(
        config=SandboxSecurityConfig(allow_unsafe_execution=True, timeout_seconds=10),
        check_connectivity=False,
    )


class TestLiveCodeBenchSuite:
    def test_registry_integration(self):
        """Verify LiveCodeBenchSuite is properly registered in EvaluationRegistry."""
        suite_cls = (
            EvaluationRegistry.get_class("livecodebench")
            if hasattr(EvaluationRegistry, "get_class")
            else EvaluationRegistry.list_suites()["livecodebench"]
        )
        assert suite_cls is LiveCodeBenchSuite
        suite = EvaluationRegistry.get("livecodebench")
        assert isinstance(suite, EvaluationSuite)
        assert suite.name == "livecodebench"

    def test_canonical_tasks_contain_honest_assertions(self):
        """Verify LiveCodeBench tasks contain real assertions, NOT 'def check(): pass' dummy stubs."""
        suite = LiveCodeBenchSuite()
        tasks = suite.load_tasks()
        assert len(tasks) >= 3

        for task in tasks:
            assert isinstance(task, BenchmarkTask)
            assert task.entry_point
            assert len(task.prompt) > 20
            assert len(task.canonical_solution) > 20
            assert "def check(" in task.test_code or "assert " in task.test_code
            assert "pass\ncheck()" not in task.test_code.replace(" ", "")

    def test_canonical_solution_passes(self, sandbox_runner: SandboxedCodeRunner):
        """Verify that the bundled canonical solution passes the test harness."""
        suite = LiveCodeBenchSuite()
        tasks = suite.load_tasks(limit=1)
        task = tasks[0]

        result = suite.evaluate_response(task, task.canonical_solution, sandbox_runner)
        assert result.passed is True
        assert result.compile_success is True
        assert result.unit_tests_passed == 1
        assert result.error_message is None

    def test_incorrect_solution_fails_honestly(self, sandbox_runner: SandboxedCodeRunner):
        """Verify that a broken solution fails with AssertionError rather than passing falsely."""
        suite = LiveCodeBenchSuite()
        tasks = suite.load_tasks(limit=1)
        task = tasks[0]

        broken_solution = f"def {task.entry_point}(*args, **kwargs):\n    return -999999\n"
        result = suite.evaluate_response(task, broken_solution, sandbox_runner)
        assert result.passed is False
        assert result.unit_tests_passed == 0

    def test_evaluate_suite_computes_pass_at_k(self, sandbox_runner: SandboxedCodeRunner):
        """Verify batch evaluation computes proper pass_at_1 and summary metrics."""
        suite = LiveCodeBenchSuite()
        tasks = suite.load_tasks(limit=2)

        # Mock backend returning canonical solution
        backend = MagicMock()
        backend.generate.return_value = MagicMock(generated_text=tasks[0].canonical_solution)

        res = suite.evaluate_suite(
            backend=backend,
            tasks=tasks[:1],
            sandbox_runner=sandbox_runner,
        )
        assert res.suite_name == "livecodebench"
        assert res.total_tasks == 1
        assert res.passed_tasks == 1
        assert res.pass_at_1 == 1.0
        assert "pass@1" in res.summary_metrics
