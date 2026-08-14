"""HumanEval & HumanEval+ Benchmark Suite Adapter."""

from typing import Any

from vipym.core.logger import get_logger
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.interfaces.evaluation import BenchmarkTask, EvaluationSuite, TaskResult

logger = get_logger(__name__)


class HumanEvalSuite(EvaluationSuite):
    """HumanEval execution-based coding benchmark."""

    @property
    def name(self) -> str:
        return "humaneval"

    @property
    def version(self) -> str:
        return "v1.0.0"

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        # Reference canonical sample tasks
        sample_tasks = [
            BenchmarkTask(
                task_id="HumanEval/0",
                suite="humaneval",
                entry_point="has_close_elements",
                prompt='def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
                test_code="""
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False

check(has_close_elements)
""",
            ),
            BenchmarkTask(
                task_id="HumanEval/1",
                suite="humaneval",
                entry_point="separate_paren_groups",
                prompt='def separate_paren_groups(paren_string: str) -> list[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced, each group is delimited by whitespace.\n    """\n',
                test_code="""
def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']

check(separate_paren_groups)
""",
            ),
        ]
        if limit is not None:
            return sample_tasks[:limit]
        return sample_tasks

    def format_prompt(self, task: BenchmarkTask, tokenizer: Any | None = None) -> str:
        return task.prompt

    def evaluate_response(
        self,
        task: BenchmarkTask,
        generated_text: str,
        sandbox_runner: SandboxedCodeRunner,
    ) -> TaskResult:
        full_code = f"{task.prompt}\n{generated_text}\n{task.test_code}"
        res = sandbox_runner.execute_in_sandbox(full_code, timeout_sec=task.timeout_seconds)

        return TaskResult(
            task_id=task.task_id,
            suite=self.name,
            prompt=task.prompt,
            generated_solution=generated_text,
            passed=res.passed,
            compile_success=res.compile_success,
            unit_tests_passed=1 if res.passed else 0,
            unit_tests_total=1,
            execution_time_ms=res.execution_time_ms,
            error_message=res.stderr if not res.passed else None,
            stdout=res.stdout,
        )


EvaluationRegistry.register("humaneval", HumanEvalSuite)
EvaluationRegistry.register("humaneval_plus", HumanEvalSuite)
