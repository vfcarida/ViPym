"""Evaluation Suite Runner and Aggregator."""

import time
from typing import Dict, List, Optional
from vipym.core.logger import get_logger
from vipym.interfaces.evaluation import EvaluationSuite, EvaluationSuiteResult, TaskResult
from vipym.interfaces.inference import GenerationRequest, InferenceBackend
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner

logger = get_logger(__name__)


class BenchmarkRunner:
    """Orchestrates running multiple evaluation suites against an active inference backend."""

    def __init__(self, sandbox_runner: Optional[SandboxedCodeRunner] = None) -> None:
        self.sandbox = sandbox_runner or SandboxedCodeRunner()

    def run_suite(
        self,
        suite_name: str,
        backend: InferenceBackend,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        task_limit: Optional[int] = None,
    ) -> EvaluationSuiteResult:
        suite = EvaluationRegistry.get(suite_name)
        tasks = suite.load_tasks(limit=task_limit)
        logger.info(f"Evaluating suite '{suite.name}' ({suite.version}) with {len(tasks)} tasks")

        task_results: List[TaskResult] = []
        passed_count = 0
        compile_count = 0

        for idx, task in enumerate(tasks):
            prompt = suite.format_prompt(task)
            gen_req = GenerationRequest(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # Generate solution from serving engine
            resp = backend.generate(gen_req)

            # Evaluate solution in secure sandbox
            res = suite.evaluate_response(task, resp.generated_text, self.sandbox)
            task_results.append(res)

            if res.passed:
                passed_count += 1
            if res.compile_success:
                compile_count += 1

        total = max(1, len(tasks))
        pass_at_1 = passed_count / total
        compile_rate = compile_count / total

        logger.info(
            f"Suite '{suite.name}' completed: pass@1={pass_at_1:.3f} compile_rate={compile_rate:.3f} ({passed_count}/{total})"
        )

        return EvaluationSuiteResult(
            suite_name=suite.name,
            benchmark_version=suite.version,
            total_tasks=len(tasks),
            passed_tasks=passed_count,
            pass_at_1=pass_at_1,
            compile_rate=compile_rate,
            unit_test_pass_rate=pass_at_1,
            task_results=task_results,
        )
