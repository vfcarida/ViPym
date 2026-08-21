"""Unit tests for P008 — SWE-bench Evaluation Suite & Agents.

Test classes:
  TestSWEBenchConfig         — Config parsing, variant verification, agent parameters
  TestSWEBenchTaskLoading    — Offline fallback loading, task limits, HF datasets loading mock
  TestPatchExtraction        — Markdown diff fences, raw diffs, submit_patch tags, edge cases
  TestSWEBenchAgent          — Single-shot generation, iterative tool execution (view/list/search/submit)
  TestSWEBenchEvaluation     — Known-good patch evaluation, invalid syntax, empty diffs, metrics
  TestSWEBenchSuiteBatch     — Multi-instance evaluate_suite, parallel execution, resolve/apply rates
  TestEvaluationRegistry     — Registry lookup for swebench, swebench_lite, swebench_verified, swebench_full
  TestBenchmarkRunnerIntegration — Integration with ViPym BenchmarkRunner
"""

from __future__ import annotations

import textwrap
from typing import Any
from unittest.mock import patch

import pytest

from vipym.evaluation.agents import (
    AgentResult,
    SWEBenchAgent,
    SWEBenchAgentConfig,
)
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.runner import BenchmarkRunner
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.suites.swebench import (
    SWEBenchFullSuite,
    SWEBenchLiteSuite,
    SWEBenchSuite,
    SWEBenchVerifiedSuite,
)
from vipym.interfaces.evaluation import BenchmarkTask, EvaluationSuiteResult, TaskResult
from vipym.interfaces.inference import GenerationRequest, GenerationResponse, InferenceBackend

# ============================================================
# Fixtures & Mocks
# ============================================================


class MockInferenceBackend(InferenceBackend):
    """Mock inference backend returning canned responses."""

    def __init__(self, response_generator=None) -> None:
        self.response_generator = response_generator or (
            lambda req: "diff --git a/file.py b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        )
        self.call_history: list[GenerationRequest] = []

    def start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.call_history.append(request)
        text = (
            self.response_generator(request)
            if callable(self.response_generator)
            else str(self.response_generator)
        )
        return GenerationResponse(
            generated_text=text,
            prompt_tokens=len(request.prompt) // 4,
            completion_tokens=len(text) // 4,
            time_to_first_token_ms=10.0,
            inter_token_latency_ms=2.0,
            total_time_ms=25.0,
        )

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        return self.generate(request)

    def stop(self) -> None:
        pass


@pytest.fixture()
def mock_backend() -> MockInferenceBackend:
    return MockInferenceBackend()


@pytest.fixture()
def sample_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="django__django-11099",
        suite="swebench",
        entry_point="django/django",
        prompt="Fix ASCII username validator regex in Django auth module",
        canonical_solution="""diff --git a/django/contrib/auth/validators.py b/django/contrib/auth/validators.py
--- a/django/contrib/auth/validators.py
+++ b/django/contrib/auth/validators.py
@@ -7,7 +7,7 @@
 class ASCIIUsernameValidator(validators.RegexValidator):
-    regex = r'^[\\w.@+-]+$'
+    regex = r'\\A[\\w.@+-]+\\Z'
""",
        test_code="assert True",
        timeout_seconds=60,
        metadata={
            "instance_id": "django__django-11099",
            "repo": "django/django",
            "base_commit": "4f9d555c8c5c99e913a4bc4d420bb5c59df3fcf8",
            "FAIL_TO_PASS": ["tests.test_validators.test_ascii_validator_trailing_newline"],
            "PASS_TO_PASS": ["tests.test_validators.test_ascii_validator"],
            "files": {
                "django/contrib/auth/validators.py": "class ASCIIUsernameValidator:\n    regex = r'^[\\w.@+-]+$'\n",
                "django/contrib/auth/models.py": "# Models\n",
            },
        },
    )


# ============================================================
# TestSWEBenchConfig
# ============================================================


class TestSWEBenchConfig:
    def test_default_config(self):
        suite = SWEBenchSuite()
        assert suite.variant == "verified"
        assert suite.version == "verified_v1.0"
        assert suite.name == "swebench"
        assert suite.agent_config.strategy == "iterative"

    def test_custom_variants(self):
        suite_lite = SWEBenchSuite(variant="lite")
        assert suite_lite.variant == "lite"
        assert suite_lite.version == "lite_v1.0"

        suite_full = SWEBenchSuite(variant="full")
        assert suite_full.variant == "full"
        assert suite_full.version == "full_v1.0"

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown SWE-bench variant"):
            SWEBenchSuite(variant="invalid_variant")

    def test_agent_config_from_dict(self):
        cfg = SWEBenchAgentConfig.from_dict(
            {
                "strategy": "single_shot",
                "max_turns": 3,
                "context_window": 16000,
                "temperature": 0.2,
            }
        )
        assert cfg.strategy == "single_shot"
        assert cfg.max_turns == 3
        assert cfg.context_window == 16000
        assert cfg.temperature == pytest.approx(0.2)

    def test_suite_with_dict_agent_config(self):
        suite = SWEBenchSuite(
            variant="verified",
            agent_config={"strategy": "single_shot", "max_turns": 2},
        )
        assert suite.agent_config.strategy == "single_shot"
        assert suite.agent_config.max_turns == 2


# ============================================================
# TestSWEBenchTaskLoading
# ============================================================


class TestSWEBenchTaskLoading:
    def test_fallback_task_loading(self):
        suite = SWEBenchSuite(variant="verified")
        # In unit tests, HF dataset load fails or uses fallback
        tasks = suite.load_tasks()
        assert len(tasks) >= 1
        task = tasks[0]
        assert isinstance(task, BenchmarkTask)
        assert task.suite == "swebench"
        assert "instance_id" in task.metadata
        assert "repo" in task.metadata
        assert "FAIL_TO_PASS" in task.metadata
        assert "PASS_TO_PASS" in task.metadata

    def test_task_loading_limit(self):
        suite = SWEBenchSuite(variant="lite")
        tasks = suite.load_tasks(limit=1)
        assert len(tasks) == 1

    @patch("datasets.load_dataset")
    def test_hf_dataset_loading_mock(self, mock_load_dataset):
        mock_ds = [
            {
                "instance_id": "test__repo-123",
                "repo": "test/repo",
                "base_commit": "abc1234",
                "environment_setup_commit": "env1234",
                "problem_statement": "A mock bug description",
                "hints_text": "A hint",
                "test_patch": "diff --git a/test.py b/test.py\n",
                "patch": "diff --git a/src.py b/src.py\n",
                "FAIL_TO_PASS": '["test_fail_to_pass"]',
                "PASS_TO_PASS": '["test_pass_to_pass"]',
                "version": "1.0",
            }
        ]
        mock_load_dataset.return_value = mock_ds

        suite = SWEBenchSuite(variant="verified")
        tasks = suite.load_tasks(limit=1)
        assert len(tasks) == 1
        assert tasks[0].task_id == "test__repo-123"
        assert tasks[0].prompt == "A mock bug description"
        assert tasks[0].metadata["FAIL_TO_PASS"] == ["test_fail_to_pass"]


# ============================================================
# TestPatchExtraction
# ============================================================


class TestPatchExtraction:
    def test_extract_from_markdown_diff_block(self):
        text = textwrap.dedent("""\
            Here is the fix for the bug:
            ```diff
            diff --git a/django/validators.py b/django/validators.py
            --- a/django/validators.py
            +++ b/django/validators.py
            @@ -1,3 +1,3 @@
            -old_line
            +new_line
            ```
            Hope this helps!
        """)
        patch = SWEBenchAgent.extract_patch(text)
        assert "diff --git a/django/validators.py" in patch
        assert "+new_line" in patch

    def test_extract_from_submit_patch_tags(self):
        text = textwrap.dedent("""\
            <submit_patch>
            diff --git a/sympy/core.py b/sympy/core.py
            --- a/sympy/core.py
            +++ b/sympy/core.py
            @@ -10,3 +10,4 @@
            +__slots__ = ('name',)
            </submit_patch>
        """)
        patch = SWEBenchAgent.extract_patch(text)
        assert "diff --git a/sympy/core.py" in patch
        assert "+__slots__" in patch

    def test_extract_from_raw_diff(self):
        text = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
        patch = SWEBenchAgent.extract_patch(text)
        assert patch.startswith("diff --git")

    def test_extract_from_empty_returns_empty(self):
        assert SWEBenchAgent.extract_patch("") == ""
        assert SWEBenchAgent.extract_patch("No code diff here, just conversation.") == ""


# ============================================================
# TestSWEBenchAgent
# ============================================================


class TestSWEBenchAgent:
    def test_single_shot_agent(self, sample_task):
        cfg = SWEBenchAgentConfig(strategy="single_shot")
        agent = SWEBenchAgent(cfg)

        backend = MockInferenceBackend(
            response_generator=lambda req: "```diff\n" + sample_task.canonical_solution + "\n```"
        )
        result = agent.solve(sample_task, backend)

        assert isinstance(result, AgentResult)
        assert result.completed is True
        assert len(result.turns) == 1
        assert "regex" in result.patch

    def test_iterative_agent_tool_view_file(self, sample_task):
        cfg = SWEBenchAgentConfig(strategy="iterative", max_turns=3)
        agent = SWEBenchAgent(cfg)

        turn_responses = [
            '<view_file path="django/contrib/auth/validators.py" start="1" end="10"/>',
            "<submit_patch>\n" + sample_task.canonical_solution + "\n</submit_patch>",
        ]
        turn_counter = [0]

        def dynamic_backend(req: GenerationRequest) -> str:
            idx = turn_counter[0]
            turn_counter[0] += 1
            return turn_responses[min(idx, len(turn_responses) - 1)]

        backend = MockInferenceBackend(response_generator=dynamic_backend)
        result = agent.solve(sample_task, backend)

        assert result.completed is True
        assert len(result.turns) == 2
        assert result.turns[0].action.action_type == "view_file"
        assert "ASCIIUsernameValidator" in result.turns[0].observation.output
        assert "regex" in result.patch

    def test_iterative_agent_tool_list_files(self, sample_task):
        cfg = SWEBenchAgentConfig(strategy="iterative", max_turns=2)
        agent = SWEBenchAgent(cfg)

        turn_responses = [
            '<list_files dir="django/contrib/auth"/>',
            "<submit_patch>\n" + sample_task.canonical_solution + "\n</submit_patch>",
        ]
        turn_counter = [0]

        def dynamic_backend(req: GenerationRequest) -> str:
            idx = turn_counter[0]
            turn_counter[0] += 1
            return turn_responses[min(idx, len(turn_responses) - 1)]

        backend = MockInferenceBackend(response_generator=dynamic_backend)
        result = agent.solve(sample_task, backend)

        assert result.completed is True
        assert result.turns[0].action.action_type == "list_files"
        assert "django/contrib/auth/validators.py" in result.turns[0].observation.output

    def test_iterative_agent_tool_search_dir(self, sample_task):
        cfg = SWEBenchAgentConfig(strategy="iterative", max_turns=2)
        agent = SWEBenchAgent(cfg)

        turn_responses = [
            '<search_dir query="ASCIIUsernameValidator" dir="django"/>',
            "<submit_patch>\n" + sample_task.canonical_solution + "\n</submit_patch>",
        ]
        turn_counter = [0]

        def dynamic_backend(req: GenerationRequest) -> str:
            idx = turn_counter[0]
            turn_counter[0] += 1
            return turn_responses[min(idx, len(turn_responses) - 1)]

        backend = MockInferenceBackend(response_generator=dynamic_backend)
        result = agent.solve(sample_task, backend)

        assert result.completed is True
        assert result.turns[0].action.action_type == "search_dir"
        assert "ASCIIUsernameValidator" in result.turns[0].observation.output

    def test_iterative_agent_max_turns_limit(self, sample_task):
        cfg = SWEBenchAgentConfig(strategy="iterative", max_turns=2)
        agent = SWEBenchAgent(cfg)

        backend = MockInferenceBackend(
            response_generator=lambda req: '<view_file path="unknown.py"/>'
        )
        result = agent.solve(sample_task, backend)

        assert len(result.turns) == 2
        assert result.completed is False


# ============================================================
# TestSWEBenchEvaluation
# ============================================================


class TestSWEBenchEvaluation:
    def test_evaluate_canonical_solution_resolves(self, sample_task):
        suite = SWEBenchSuite()
        sandbox = SandboxedCodeRunner(check_connectivity=False)

        task_result = suite.evaluate_response(
            task=sample_task,
            generated_text=sample_task.canonical_solution,
            sandbox_runner=sandbox,
        )

        assert isinstance(task_result, TaskResult)
        assert task_result.passed is True
        assert task_result.compile_success is True
        assert task_result.unit_tests_passed >= 1

    def test_evaluate_invalid_patch_syntax(self, sample_task):
        suite = SWEBenchSuite()
        sandbox = SandboxedCodeRunner(check_connectivity=False)

        task_result = suite.evaluate_response(
            task=sample_task,
            generated_text="This is just some plain text without any diff.",
            sandbox_runner=sandbox,
        )

        assert task_result.passed is False
        assert task_result.compile_success is False
        assert task_result.error_message is not None

    def test_evaluate_empty_response(self, sample_task):
        suite = SWEBenchSuite()
        sandbox = SandboxedCodeRunner(check_connectivity=False)

        task_result = suite.evaluate_response(
            task=sample_task,
            generated_text="",
            sandbox_runner=sandbox,
        )

        assert task_result.passed is False
        assert task_result.compile_success is False


# ============================================================
# TestSWEBenchSuiteBatch
# ============================================================


class TestSWEBenchSuiteBatch:
    def test_evaluate_suite_success(self, sample_task):
        suite = SWEBenchSuite(agent_config={"strategy": "single_shot"})
        backend = MockInferenceBackend(
            response_generator=lambda req: "```diff\n" + sample_task.canonical_solution + "\n```"
        )

        result = suite.evaluate_suite(
            backend=backend,
            tasks=[sample_task],
        )

        assert isinstance(result, EvaluationSuiteResult)
        assert result.suite_name == "swebench"
        assert result.total_tasks == 1
        assert result.passed_tasks == 1
        assert result.pass_at_1 == 1.0
        assert result.compile_rate == 1.0
        assert "resolve_rate" in result.summary_metrics
        assert result.summary_metrics["resolve_rate"] == 1.0
        assert result.summary_metrics["apply_rate"] == 1.0

    def test_evaluate_suite_parallel(self, sample_task):
        suite = SWEBenchSuite(
            agent_config={"strategy": "single_shot"},
            parallel_instances=2,
        )
        backend = MockInferenceBackend(
            response_generator=lambda req: "```diff\n" + sample_task.canonical_solution + "\n```"
        )

        result = suite.evaluate_suite(
            backend=backend,
            tasks=[sample_task, sample_task],
        )

        assert result.total_tasks == 2
        assert result.passed_tasks == 2
        assert result.pass_at_1 == 1.0


# ============================================================
# TestEvaluationRegistry
# ============================================================


class TestEvaluationRegistry:
    def test_swebench_registered(self):
        suite = EvaluationRegistry.get("swebench")
        assert isinstance(suite, SWEBenchSuite)

    def test_swebench_lite_registered(self):
        suite = EvaluationRegistry.get("swebench_lite")
        assert isinstance(suite, SWEBenchLiteSuite)
        assert suite.variant == "lite"

    def test_swebench_verified_registered(self):
        suite = EvaluationRegistry.get("swebench_verified")
        assert isinstance(suite, SWEBenchVerifiedSuite)
        assert suite.variant == "verified"

    def test_swebench_full_registered(self):
        suite = EvaluationRegistry.get("swebench_full")
        assert isinstance(suite, SWEBenchFullSuite)
        assert suite.variant == "full"


# ============================================================
# TestBenchmarkRunnerIntegration
# ============================================================


class TestBenchmarkRunnerIntegration:
    def test_benchmark_runner_runs_swebench(self, monkeypatch):
        monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")
        sandbox = SandboxedCodeRunner(check_connectivity=False)
        runner = BenchmarkRunner(sandbox_runner=sandbox)

        # Mock backend that outputs canonical solution for sample tasks
        sample_task_solution = """diff --git a/django/contrib/auth/validators.py b/django/contrib/auth/validators.py
--- a/django/contrib/auth/validators.py
+++ b/django/contrib/auth/validators.py
@@ -7,7 +7,7 @@
 class ASCIIUsernameValidator(validators.RegexValidator):
-    regex = r'^[\\w.@+-]+$'
+    regex = r'\\A[\\w.@+-]+\\Z'
"""
        backend = MockInferenceBackend(
            response_generator=lambda req: "```diff\n" + sample_task_solution + "\n```"
        )

        result = runner.run_suite(
            suite_name="swebench",
            backend=backend,
            task_limit=1,
        )

        assert isinstance(result, EvaluationSuiteResult)
        assert result.suite_name == "swebench"
        assert result.total_tasks == 1
        assert len(result.task_results) == 1
