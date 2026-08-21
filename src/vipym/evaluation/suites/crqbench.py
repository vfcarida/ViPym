"""CRQBench: Code Review Quality Benchmark Suite.

Evaluates an LLM's capability to review code diffs, detect real software defects,
and provide actionable improvements.
Measures Precision, Recall, F1 score, and Actionability.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

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
# Bundled Expert-Annotated Code Review Tasks (for offline / CI environments)
# ---------------------------------------------------------------------------

_CRQ_SAMPLE_TASKS = [
    {
        "task_id": "crq/concurrency/race_condition",
        "title": "Thread-unsafe shared cache update",
        "diff": """diff --git a/cache/shared_store.py b/cache/shared_store.py
--- a/cache/shared_store.py
+++ b/cache/shared_store.py
@@ -10,6 +10,8 @@
 class SharedStore:
     def __init__(self):
         self._data = {}
+    def increment(self, key: str):
+        self._data[key] = self._data.get(key, 0) + 1
""",
        "expert_annotations": [
            {
                "line": 13,
                "defect_type": "concurrency",
                "keywords": ["thread", "race condition", "lock", "atomic", "mutex", "concurrency"],
                "severity": "high",
                "description": "Non-atomic read-modify-write on shared dictionary causes race conditions under multi-threaded access.",
                "suggested_fix": "Use a threading.Lock around dictionary access or use collections.defaultdict with synchronization.",
            }
        ],
    },
    {
        "task_id": "crq/resource_leak/unclosed_file",
        "title": "Unclosed file handle in log parser",
        "diff": """diff --git a/logs/parser.py b/logs/parser.py
--- a/logs/parser.py
+++ b/logs/parser.py
@@ -25,4 +25,6 @@
 def parse_log_entries(filepath: str) -> list[str]:
+    f = open(filepath, 'r')
+    entries = [line.strip() for line in f if 'ERROR' in line]
+    return entries
""",
        "expert_annotations": [
            {
                "line": 26,
                "defect_type": "resource_leak",
                "keywords": [
                    "open",
                    "close",
                    "context manager",
                    "with open",
                    "leak",
                    "file handle",
                ],
                "severity": "medium",
                "description": "File opened with bare open() without closing or using 'with open(...) as f:', causing file descriptor leaks.",
                "suggested_fix": "Use 'with open(filepath, \"r\") as f:' context manager.",
            }
        ],
    },
    {
        "task_id": "crq/security/sql_injection",
        "title": "Raw SQL string formatting in repository query",
        "diff": """diff --git a/db/user_repo.py b/db/user_repo.py
--- a/db/user_repo.py
+++ b/db/user_repo.py
@@ -15,5 +15,5 @@
 def find_user(cursor, username: str):
-    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
+    cursor.execute(f"SELECT * FROM users WHERE username = '{username}'")
     return cursor.fetchone()
""",
        "expert_annotations": [
            {
                "line": 16,
                "defect_type": "security",
                "keywords": [
                    "sql injection",
                    "parameterized",
                    "f-string",
                    "security",
                    "sqli",
                    "prepared statement",
                ],
                "severity": "critical",
                "description": "Formatted raw SQL string introduces direct SQL injection vulnerability.",
                "suggested_fix": "Revert to parameterized query cursor.execute(query, (username,)).",
            }
        ],
    },
]


# ---------------------------------------------------------------------------
# Code Review Quality Metrics Models
# ---------------------------------------------------------------------------


@dataclass
class ReviewComment:
    """Individual review comment extracted from model output."""

    line: int | None = None
    defect_type: str = "general"
    comment: str = ""
    suggested_fix: str | None = None
    is_actionable: bool = False


@dataclass
class CRQMetrics:
    """Precision, recall, and actionability scores for code review."""

    precision: float
    recall: float
    f1_score: float
    actionability: float
    true_positives: int
    false_positives: int
    false_negatives: int


# ---------------------------------------------------------------------------
# CRQBenchSuite Implementation
# ---------------------------------------------------------------------------


class CRQBenchSuite(EvaluationSuite):
    """CRQBench Code Review Benchmark Suite."""

    def __init__(
        self,
        timeout_per_task: int = 30,
        parallel_tasks: int = 4,
    ) -> None:
        self.timeout_per_task = timeout_per_task
        self.parallel_tasks = parallel_tasks

    @property
    def name(self) -> str:
        return "crqbench"

    @property
    def version(self) -> str:
        return "v1.0"

    # ------------------------------------------------------------------
    # Task Loading
    # ------------------------------------------------------------------

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load code review tasks from Hugging Face or fallback annotations."""
        tasks: list[BenchmarkTask] = []

        try:
            from datasets import load_dataset  # type: ignore[import]

            hf_ds = load_dataset("crqbench/code-reviews", split="train")
            for item in hf_ds:
                tasks.append(
                    BenchmarkTask(
                        task_id=item.get("task_id", "crq/task"),
                        suite=self.name,
                        entry_point=item.get("title", "code_review"),
                        prompt=item.get("diff", ""),
                        canonical_solution="",
                        test_code="",
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "diff": item.get("diff", ""),
                            "expert_annotations": json.loads(item["expert_annotations"])
                            if isinstance(item.get("expert_annotations"), str)
                            else item.get("expert_annotations", []),
                        },
                    )
                )
                if limit and len(tasks) >= limit:
                    break
        except Exception:  # noqa: BLE001
            for item in _CRQ_SAMPLE_TASKS:
                tasks.append(
                    BenchmarkTask(
                        task_id=item["task_id"],
                        suite=self.name,
                        entry_point=item["title"],
                        prompt=item["diff"],
                        canonical_solution="",
                        test_code="",
                        timeout_seconds=self.timeout_per_task,
                        metadata={
                            "diff": item["diff"],
                            "expert_annotations": item["expert_annotations"],
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
        """Format code review request."""
        diff = task.metadata.get("diff", task.prompt)
        title = task.entry_point or "Code Changes"

        return (
            "You are a senior principal engineer performing a rigorous code review.\n"
            f"Review the following code diff for '{title}'.\n"
            "Identify critical bugs, security vulnerabilities, performance issues, resource leaks, or concurrency hazards.\n"
            "For each issue, state the defect clearly and provide a concrete actionable fix.\n\n"
            f"=== Code Diff ===\n"
            f"```diff\n{diff}\n```\n\n"
            "Provide your review comments below:\n"
        )

    # ------------------------------------------------------------------
    # Evaluation Logic
    # ------------------------------------------------------------------

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner | Any = None,
    ) -> TaskResult:
        """Compare model review comments against expert-annotated defect records."""
        t0 = time.perf_counter()
        annotations = task.metadata.get("expert_annotations", [])

        comments = self._parse_review_comments(generated_text)
        metrics = self._compute_review_metrics(comments, annotations, generated_text)

        elapsed = (time.perf_counter() - t0) * 1000.0
        passed = metrics.recall >= 0.50

        return TaskResult(
            task_id=task.task_id,
            suite=self.name,
            prompt=task.prompt,
            generated_solution=generated_text,
            passed=passed,
            compile_success=len(comments) > 0,
            unit_tests_passed=metrics.true_positives,
            unit_tests_total=len(annotations),
            execution_time_ms=elapsed,
            stdout=(
                f"Precision: {metrics.precision:.1%} | Recall: {metrics.recall:.1%} | "
                f"F1: {metrics.f1_score:.1%} | Actionability: {metrics.actionability:.1%}"
            ),
        )

    def _parse_review_comments(self, text: str) -> list[ReviewComment]:
        """Extract review points, severity, and actionability from text."""
        comments: list[ReviewComment] = []
        if not text.strip():
            return comments

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in paragraphs:
            # Check if paragraph discusses a potential defect
            is_actionable = bool(
                "```" in p
                or "suggest" in p.lower()
                or "use " in p.lower()
                or "replace" in p.lower()
            )
            comments.append(
                ReviewComment(
                    comment=p,
                    is_actionable=is_actionable,
                )
            )

        return comments

    def _compute_review_metrics(
        self,
        comments: list[ReviewComment],
        annotations: list[dict[str, Any]],
        full_text: str,
    ) -> CRQMetrics:
        """Calculate precision, recall, F1, and actionability."""
        if not annotations:
            return CRQMetrics(1.0, 1.0, 1.0, 1.0, 0, 0, 0)

        true_positives = 0
        text_lower = full_text.lower()

        # Check each ground truth defect
        for ann in annotations:
            keywords = ann.get("keywords", [])
            matched = any(kw.lower() in text_lower for kw in keywords)
            if matched:
                true_positives += 1

        total_annotated = len(annotations)
        total_comments = max(1, len(comments))

        recall = true_positives / total_annotated
        precision = min(1.0, true_positives / total_comments)
        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        actionable_count = sum(1 for c in comments if c.is_actionable)
        actionability = actionable_count / total_comments if comments else 0.0

        return CRQMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            actionability=actionability,
            true_positives=true_positives,
            false_positives=total_comments - true_positives,
            false_negatives=total_annotated - true_positives,
        )

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
        """Run full CRQBench evaluation across tasks."""
        if tasks is None:
            tasks = self.load_tasks(limit=task_limit)

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

            return self.evaluate_response(task, gen_text)

        if self.parallel_tasks > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=self.parallel_tasks) as pool:
                task_results = list(pool.map(_eval_one, tasks))
        else:
            task_results = [_eval_one(t) for t in tasks]

        total = max(1, len(task_results))
        passed_count = sum(1 for r in task_results if r.passed)

        precisions: list[float] = []
        recalls: list[float] = []
        actionabilities: list[float] = []

        for r in task_results:
            m_p = re.search(r"Precision:\s*([\d.]+)%", r.stdout or "")
            m_r = re.search(r"Recall:\s*([\d.]+)%", r.stdout or "")
            m_a = re.search(r"Actionability:\s*([\d.]+)%", r.stdout or "")
            if m_p:
                precisions.append(float(m_p.group(1)) / 100.0)
            if m_r:
                recalls.append(float(m_r.group(1)) / 100.0)
            if m_a:
                actionabilities.append(float(m_a.group(1)) / 100.0)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_actionability = sum(actionabilities) / len(actionabilities) if actionabilities else 0.0

        logger.info(
            f"CRQBench results: precision={avg_precision:.2%} recall={avg_recall:.2%} "
            f"actionability={avg_actionability:.2%} ({passed_count}/{total})"
        )

        return EvaluationSuiteResult(
            suite_name=self.name,
            benchmark_version=self.version,
            total_tasks=len(task_results),
            passed_tasks=passed_count,
            pass_at_1=avg_precision,
            compile_rate=1.0,
            unit_test_pass_rate=avg_recall,
            task_results=task_results,
            summary_metrics={
                "precision": avg_precision,
                "recall": avg_recall,
                "actionability": avg_actionability,
                "pass_rate": passed_count / total,
                "total_tasks": total,
            },
        )


# Register in EvaluationRegistry
EvaluationRegistry.register("crqbench", CRQBenchSuite)
EvaluationRegistry.register("code_review", CRQBenchSuite)
