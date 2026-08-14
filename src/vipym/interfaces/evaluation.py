"""Interfaces for Evaluation Suites and Benchmark Tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pydantic


class BenchmarkTask(pydantic.BaseModel):
    """Generic representation of a benchmark question/task."""
    task_id: str
    suite: str
    entry_point: Optional[str] = None
    prompt: str
    canonical_solution: Optional[str] = None
    test_code: str
    timeout_seconds: int = 15
    release_date: Optional[str] = None
    metadata: Dict[str, Any] = pydantic.Field(default_factory=dict)


class TaskResult(pydantic.BaseModel):
    """Execution result of an individual evaluation task."""
    task_id: str
    suite: str
    prompt: str
    generated_solution: str
    passed: bool
    compile_success: bool
    unit_tests_passed: int
    unit_tests_total: int
    execution_time_ms: float
    error_message: Optional[str] = None
    stdout: Optional[str] = None


class EvaluationSuiteResult(pydantic.BaseModel):
    """Aggregated score for a benchmark suite."""
    suite_name: str
    benchmark_version: str
    total_tasks: int
    passed_tasks: int
    pass_at_1: float
    compile_rate: float
    unit_test_pass_rate: float
    task_results: List[TaskResult]
    contamination_risk_score: float = 0.0
    summary_metrics: Dict[str, Any] = pydantic.Field(default_factory=dict)


class EvaluationSuite(ABC):
    """Abstract interface for domain benchmark suites."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique suite name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Version/revision of the benchmark suite."""
        pass

    @abstractmethod
    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """Load benchmark tasks."""
        pass

    @abstractmethod
    def format_prompt(self, task: BenchmarkTask, tokenizer: Optional[Any] = None) -> str:
        """Format task into model prompt."""
        pass

    @abstractmethod
    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: Any,
    ) -> TaskResult:
        """Evaluate generated solution against task assertions inside isolated sandbox."""
        pass
