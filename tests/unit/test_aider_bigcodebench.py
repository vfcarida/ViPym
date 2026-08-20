"""Unit tests for P009 — Aider Edit, Aider Polyglot, and BigCodeBench Suites.

Test classes:
  TestEditFormats              — SEARCH/REPLACE parsing, unified diffs, whole file, fuzzy matching, compliance
  TestAiderEditSuite           — AiderEditSuite task loading, prompt formatting, apply rate, accuracy
  TestAiderPolyglotSuite       — Multi-language loading, language filtering, per-language breakdown metrics
  TestBigCodeBenchSuite        — BigCodeBench task loading, AST library coverage, execution, variants
  TestEvaluationRegistrySuite  — Registry discovery of all new suites and aliases
  TestBenchmarkRunnerSuite     — Integration with BenchmarkRunner for aider_edit and bigcodebench
"""

from __future__ import annotations

import textwrap
from typing import Any
from unittest.mock import patch

import pytest

from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.runner import BenchmarkRunner
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.suites.aider_edit import AiderEditSuite
from vipym.evaluation.suites.aider_polyglot import AiderPolyglotSuite
from vipym.evaluation.suites.bigcodebench import (
    BigCodeBenchFullSuite,
    BigCodeBenchHardSuite,
    BigCodeBenchLiteSuite,
    BigCodeBenchSuite,
    compute_library_coverage,
)
from vipym.evaluation.suites.utils.edit_formats import (
    EditApplyResult,
    EditBlock,
    apply_edit,
    apply_search_replace,
    apply_unified_diff,
    apply_whole_file,
    parse_search_replace_blocks,
    validate_format_compliance,
)
from vipym.interfaces.evaluation import BenchmarkTask, EvaluationSuiteResult, TaskResult
from vipym.interfaces.inference import GenerationRequest, GenerationResponse, InferenceBackend


# ============================================================
# Mock Inference Backend
# ============================================================


class MockInferenceBackend(InferenceBackend):
    def __init__(self, response_generator=None) -> None:
        self.response_generator = response_generator or (lambda req: "def solution(): pass")

    def start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        text = self.response_generator(request) if callable(self.response_generator) else str(self.response_generator)
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


@pytest.fixture(autouse=True)
def setup_unsafe_sandbox(monkeypatch):
    monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")


@pytest.fixture
def sandbox_runner():
    from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig

    return SandboxedCodeRunner(
        config=SandboxSecurityConfig(allow_unsafe_execution=True),
        check_connectivity=False,
    )


# ============================================================
# TestEditFormats
# ============================================================


class TestEditFormats:
    def test_parse_search_replace_single_block(self):
        text = textwrap.dedent("""\
            Here is the update:
            <<<<<<< SEARCH
            def old():
                return 1
            =======
            def old():
                return 2
            >>>>>>> REPLACE
        """)
        blocks = parse_search_replace_blocks(text)
        assert len(blocks) == 1
        assert "return 1" in blocks[0].search_content
        assert "return 2" in blocks[0].replace_content

    def test_parse_search_replace_multiple_blocks(self):
        text = textwrap.dedent("""\
            <<<<<<< SEARCH
            x = 1
            =======
            x = 10
            >>>>>>> REPLACE

            Some comment in between

            <<<<<<< SEARCH
            y = 2
            =======
            y = 20
            >>>>>>> REPLACE
        """)
        blocks = parse_search_replace_blocks(text)
        assert len(blocks) == 2
        assert "x = 10" in blocks[0].replace_content
        assert "y = 20" in blocks[1].replace_content

    def test_apply_search_replace_exact(self):
        code = "def foo():\n    return 42\n"
        blocks = [EditBlock(search_content="    return 42\n", replace_content="    return 100\n")]
        res = apply_search_replace(code, blocks)
        assert res.success is True
        assert res.modified_code == "def foo():\n    return 100\n"
        assert res.blocks_applied == 1

    def test_apply_search_replace_fuzzy_whitespace(self):
        code = "def foo():\n    x = 1\n    return x\n"
        # Search block has different indentation
        blocks = [EditBlock(search_content="x = 1\nreturn x", replace_content="    x = 2\n    return x * 2")]
        res = apply_search_replace(code, blocks, fuzzy=True)
        assert res.success is True
        assert "return x * 2" in res.modified_code

    def test_apply_search_replace_missing_target(self):
        code = "def bar(): pass\n"
        blocks = [EditBlock(search_content="def nonexistent(): pass", replace_content="pass")]
        res = apply_search_replace(code, blocks, fuzzy=True)
        assert res.success is False
        assert "not found" in res.error

    def test_apply_unified_diff(self):
        code = "def foo():\n    return 1\n"
        diff = textwrap.dedent("""\
            --- a/file.py
            +++ b/file.py
            @@ -1,2 +1,2 @@
             def foo():
            -    return 1
            +    return 99
        """)
        res = apply_unified_diff(code, diff)
        assert res.success is True
        assert "return 99" in res.modified_code

    def test_apply_unified_diff_invalid_header(self):
        code = "def foo(): pass"
        diff = "Just a diff without hunk headers"
        res = apply_unified_diff(code, diff)
        assert res.success is False
        assert "No unified diff hunk headers" in res.error

    def test_apply_whole_file_markdown_fence(self):
        code = "old"
        new_text = "```python\ndef new_func():\n    return 'new'\n```"
        res = apply_whole_file(code, new_text)
        assert res.success is True
        assert "def new_func():" in res.modified_code
        assert "```" not in res.modified_code

    def test_apply_edit_auto_detection(self):
        code = "def foo():\n    return 1\n"
        sr_edit = "<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE"
        res = apply_edit(code, sr_edit, expected_format="auto")
        assert res.success is True
        assert "return 2" in res.modified_code
        assert res.format_type == "search_replace"

    def test_validate_format_compliance(self):
        valid_sr = "<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE"
        invalid_sr = "def foo(): return 1"
        assert validate_format_compliance(valid_sr, "search_replace")[0] is True
        assert validate_format_compliance(invalid_sr, "search_replace")[0] is False

        valid_diff = "--- a/f\n+++ b/f\n@@ -1,1 +1,1 @@\n-a\n+b"
        assert validate_format_compliance(valid_diff, "diff")[0] is True
        assert validate_format_compliance(invalid_sr, "diff")[0] is False


# ============================================================
# TestAiderEditSuite
# ============================================================


class TestAiderEditSuite:
    def test_init_defaults_and_custom(self):
        suite = AiderEditSuite()
        assert suite.name == "aider_edit"
        assert suite.edit_format == "search_replace"
        assert "search_replace" in suite.version

        suite_diff = AiderEditSuite(edit_format="diff")
        assert suite_diff.edit_format == "diff"

    def test_load_tasks(self):
        suite = AiderEditSuite()
        tasks = suite.load_tasks(limit=3)
        assert len(tasks) == 3
        assert tasks[0].suite == "aider_edit"
        assert "original_code" in tasks[0].metadata

    def test_format_prompt_search_replace(self):
        suite = AiderEditSuite(edit_format="search_replace")
        task = suite.load_tasks(limit=1)[0]
        prompt = suite.format_prompt(task)
        assert "<<<<<<< SEARCH" in prompt
        assert task.prompt in prompt

    def test_format_prompt_diff(self):
        suite = AiderEditSuite(edit_format="diff")
        task = suite.load_tasks(limit=1)[0]
        prompt = suite.format_prompt(task)
        assert "unified diff" in prompt

    def test_evaluate_response_success(self, sandbox_runner):
        suite = AiderEditSuite()
        task = suite.load_tasks(limit=1)[0]

        res = suite.evaluate_response(task, task.canonical_solution, sandbox_runner)
        assert isinstance(res, TaskResult)
        assert res.passed is True
        assert res.compile_success is True

    def test_evaluate_response_failed_edit(self, sandbox_runner):
        suite = AiderEditSuite()
        task = suite.load_tasks(limit=1)[0]

        bad_edit = "<<<<<<< SEARCH\nnonexistent code\n=======\nreplacement\n>>>>>>> REPLACE"
        res = suite.evaluate_response(task, bad_edit, sandbox_runner)
        assert res.passed is False
        assert res.compile_success is False
        assert "Edit failed to apply" in res.error_message

    def test_evaluate_response_failed_tests(self, sandbox_runner):
        suite = AiderEditSuite()
        task = suite.load_tasks(limit=1)[0]

        # Edit applies cleanly but gives incorrect logic
        broken_logic_edit = "<<<<<<< SEARCH\n    pass\n=======\n    return 'wrong answer'\n>>>>>>> REPLACE"
        res = suite.evaluate_response(task, broken_logic_edit, sandbox_runner)
        assert res.compile_success is True
        assert res.passed is False

    def test_evaluate_suite_batch(self, sandbox_runner):
        suite = AiderEditSuite()
        tasks = suite.load_tasks(limit=2)
        backend = MockInferenceBackend(response_generator=lambda req: tasks[0].canonical_solution)

        suite_res = suite.evaluate_suite(backend, tasks=tasks, sandbox_runner=sandbox_runner)
        assert isinstance(suite_res, EvaluationSuiteResult)
        assert suite_res.total_tasks == 2
        assert "edit_accuracy" in suite_res.summary_metrics
        assert "apply_rate" in suite_res.summary_metrics
        assert "format_compliance" in suite_res.summary_metrics


# ============================================================
# TestAiderPolyglotSuite
# ============================================================


class TestAiderPolyglotSuite:
    def test_init_and_language_filtering(self):
        suite_all = AiderPolyglotSuite()
        assert suite_all.name == "aider_polyglot"
        assert "all_12_langs" in suite_all.version

        suite_filtered = AiderPolyglotSuite(languages=["python", "javascript", "go"])
        assert suite_filtered.languages == ["python", "javascript", "go"]
        assert "go" in suite_filtered.version
        assert "python" in suite_filtered.version

    def test_load_tasks_filtered(self):
        suite = AiderPolyglotSuite(languages=["javascript"])
        tasks = suite.load_tasks()
        assert len(tasks) >= 1
        for t in tasks:
            assert t.metadata["language"] == "javascript"

    def test_format_prompt_multilang(self):
        suite = AiderPolyglotSuite(languages=["go"])
        task = suite.load_tasks(limit=1)[0]
        prompt = suite.format_prompt(task)
        assert "go developer" in prompt.lower()
        assert task.prompt in prompt

    def test_evaluate_response_python(self, sandbox_runner):
        suite = AiderPolyglotSuite(languages=["python"])
        task = suite.load_tasks(limit=1)[0]

        res = suite.evaluate_response(task, task.canonical_solution, sandbox_runner)
        assert res.passed is True
        assert res.compile_success is True

    def test_evaluate_response_polyglot_js(self, sandbox_runner):
        suite = AiderPolyglotSuite(languages=["javascript"])
        task = suite.load_tasks(limit=1)[0]

        res = suite.evaluate_response(task, task.canonical_solution, sandbox_runner)
        assert res.compile_success is True
        assert res.passed is True

    def test_evaluate_suite_per_language_breakdown(self, sandbox_runner):
        suite = AiderPolyglotSuite(languages=["python", "javascript"])
        tasks = suite.load_tasks(limit=2)
        backend = MockInferenceBackend(response_generator=lambda req: tasks[0].canonical_solution)

        suite_res = suite.evaluate_suite(backend, tasks=tasks, sandbox_runner=sandbox_runner)
        assert isinstance(suite_res, EvaluationSuiteResult)
        assert "language_breakdown" in suite_res.summary_metrics
        breakdown = suite_res.summary_metrics["language_breakdown"]
        assert "python" in breakdown or "javascript" in breakdown


# ============================================================
# TestBigCodeBenchSuite
# ============================================================


class TestBigCodeBenchSuite:
    def test_init_and_variants(self):
        suite_full = BigCodeBenchSuite(variant="full")
        assert suite_full.variant == "full"
        assert "full" in suite_full.version

        suite_hard = BigCodeBenchHardSuite()
        assert suite_hard.variant == "hard"

        suite_lite = BigCodeBenchLiteSuite()
        assert suite_lite.variant == "lite"

    def test_load_tasks(self):
        suite = BigCodeBenchSuite()
        tasks = suite.load_tasks(limit=3)
        assert len(tasks) == 3
        assert tasks[0].suite == "bigcodebench"
        assert "libs" in tasks[0].metadata

    def test_compute_library_coverage_ast(self):
        code_numpy = "import numpy as np\ndef f(): return np.mean([1, 2])"
        cov = compute_library_coverage(code_numpy, ["numpy"])
        assert cov == 1.0

        code_multi = "import pandas as pd\nimport numpy as np\ndef f(): pass"
        cov_multi = compute_library_coverage(code_multi, ["pandas", "numpy"])
        assert cov_multi == 1.0

        code_partial = "import pandas as pd\ndef f(): pass"
        cov_partial = compute_library_coverage(code_partial, ["pandas", "scipy"])
        assert cov_partial == 0.5

        cov_none = compute_library_coverage("def f(): pass", ["matplotlib"])
        assert cov_none == 0.0

    def test_evaluate_response_success(self, sandbox_runner):
        suite = BigCodeBenchSuite()
        task = suite.load_tasks(limit=1)[0]

        res = suite.evaluate_response(task, task.canonical_solution, sandbox_runner)
        assert res.passed is True
        assert res.compile_success is True
        assert "Library coverage" in res.stdout

    def test_evaluate_suite_metrics(self, sandbox_runner):
        suite = BigCodeBenchSuite()
        tasks = suite.load_tasks(limit=2)
        backend = MockInferenceBackend(response_generator=lambda req: tasks[0].canonical_solution)

        suite_res = suite.evaluate_suite(backend, tasks=tasks, sandbox_runner=sandbox_runner)
        assert isinstance(suite_res, EvaluationSuiteResult)
        assert "library_coverage" in suite_res.summary_metrics
        assert "pass_at_1" in suite_res.summary_metrics
        assert "compile_rate" in suite_res.summary_metrics


# ============================================================
# TestEvaluationRegistrySuite
# ============================================================


class TestEvaluationRegistrySuite:
    def test_aider_edit_registered(self):
        suite = EvaluationRegistry.get("aider_edit")
        assert isinstance(suite, AiderEditSuite)
        assert isinstance(EvaluationRegistry.get("aider_bench"), AiderEditSuite)
        assert isinstance(EvaluationRegistry.get("aider"), AiderEditSuite)

    def test_aider_polyglot_registered(self):
        suite = EvaluationRegistry.get("aider_polyglot")
        assert isinstance(suite, AiderPolyglotSuite)
        assert isinstance(EvaluationRegistry.get("polyglot_bench"), AiderPolyglotSuite)

    def test_bigcodebench_registered(self):
        suite = EvaluationRegistry.get("bigcodebench")
        assert isinstance(suite, BigCodeBenchSuite)
        assert isinstance(EvaluationRegistry.get("bigcodebench_full"), BigCodeBenchFullSuite)
        assert isinstance(EvaluationRegistry.get("bigcodebench_hard"), BigCodeBenchHardSuite)
        assert isinstance(EvaluationRegistry.get("bigcodebench_lite"), BigCodeBenchLiteSuite)


# ============================================================
# TestBenchmarkRunnerSuite
# ============================================================


class TestBenchmarkRunnerSuite:
    def test_benchmark_runner_runs_aider_edit(self, sandbox_runner):
        runner = BenchmarkRunner(sandbox_runner=sandbox_runner)
        suite = EvaluationRegistry.get("aider_edit")
        first_task = suite.load_tasks(limit=1)[0]
        backend = MockInferenceBackend(response_generator=lambda req: first_task.canonical_solution)

        result = runner.run_suite("aider_edit", backend=backend, task_limit=1)
        assert isinstance(result, EvaluationSuiteResult)
        assert result.suite_name == "aider_edit"
        assert result.total_tasks == 1
        assert result.passed_tasks == 1

    def test_benchmark_runner_runs_bigcodebench(self, sandbox_runner):
        runner = BenchmarkRunner(sandbox_runner=sandbox_runner)
        suite = EvaluationRegistry.get("bigcodebench")
        first_task = suite.load_tasks(limit=1)[0]
        backend = MockInferenceBackend(response_generator=lambda req: first_task.canonical_solution)

        result = runner.run_suite("bigcodebench", backend=backend, task_limit=1)
        assert isinstance(result, EvaluationSuiteResult)
        assert result.suite_name == "bigcodebench"
        assert result.total_tasks == 1
        assert result.passed_tasks == 1
