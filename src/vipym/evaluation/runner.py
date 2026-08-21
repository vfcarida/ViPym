"""Evaluation Suite Runner and Aggregator with Real Inference Telemetry."""

from __future__ import annotations

import time
from typing import Any

from vipym.config.schema import EvaluationConfig
from vipym.core.logger import get_logger
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig
from vipym.interfaces.evaluation import EvaluationSuiteResult, TaskResult
from vipym.interfaces.inference import GenerationRequest, InferenceBackend
from vipym.telemetry.cost_tracker import CostSummaryReport, InferenceCostTracker
from vipym.telemetry.profiler import InferenceProfiler, InferenceTelemetryReport

logger = get_logger(__name__)


class BenchmarkRunner:
    """Orchestrates running multiple evaluation suites with real-time inference telemetry."""

    def __init__(
        self,
        sandbox_runner: SandboxedCodeRunner | None = None,
        evaluation_config: EvaluationConfig | None = None,
        model_variant: str = "default",
        warmup_requests: int = 3,
        hardware_type: str = "default",
        hourly_hardware_rate: float | None = None,
        baseline_api: str = "gpt-4o",
    ) -> None:
        if sandbox_runner is not None:
            self.sandbox = sandbox_runner
        elif evaluation_config is not None:
            sec_config = SandboxSecurityConfig(
                timeout_seconds=evaluation_config.timeout_per_task_sec,
                allow_unsafe_execution=evaluation_config.allow_unsafe_execution,
                use_gvisor_runsc=evaluation_config.isolate_with_gvisor,
            )
            self.sandbox = SandboxedCodeRunner(config=sec_config)
        else:
            self.sandbox = SandboxedCodeRunner()

        self.evaluation_config = evaluation_config
        self.model_variant = model_variant
        self.telemetry_profiler = InferenceProfiler(
            model_variant=model_variant,
            warmup_requests=warmup_requests,
        )
        self.cost_tracker = InferenceCostTracker(
            model_variant=model_variant,
            hardware_type=hardware_type,
            hourly_hardware_rate=hourly_hardware_rate,
            baseline_api=baseline_api,
        )

    def run_suite(
        self,
        suite_name: str,
        backend: InferenceBackend,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        task_limit: int | None = None,
    ) -> EvaluationSuiteResult:
        suite = EvaluationRegistry.get(suite_name)
        if not backend.health_check():
            raise RuntimeError(
                f"Serving backend {backend.__class__.__name__} failed pre-execution health check."
            )

        tasks = suite.load_tasks(limit=task_limit)
        logger.info(f"Evaluating suite '{suite.name}' ({suite.version}) with {len(tasks)} tasks")

        task_results: list[TaskResult] = []
        passed_count = 0
        compile_count = 0

        max_workers = (
            self.evaluation_config.max_workers
            if self.evaluation_config and hasattr(self.evaluation_config, "max_workers")
            else 1
        )

        def _run_single(task_item):
            prompt = suite.format_prompt(task_item)
            gen_req = GenerationRequest(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            t0 = time.perf_counter()
            resp = backend.generate(gen_req)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            p_tok = resp.prompt_tokens if resp.prompt_tokens > 0 else (len(prompt) // 4)
            c_tok = (
                resp.completion_tokens
                if resp.completion_tokens > 0
                else (len(resp.generated_text) // 4)
            )

            self.telemetry_profiler.record_request(
                request_id=f"{suite.name}/{task_item.task_id}",
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=elapsed_ms,
                ttft_ms=resp.time_to_first_token_ms,
                inter_token_latency_ms=resp.inter_token_latency_ms,
            )
            self.cost_tracker.record_usage(
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                duration_seconds=elapsed_ms / 1000.0,
            )

            return suite.evaluate_response(task_item, resp.generated_text, self.sandbox)

        if max_workers > 1 and len(tasks) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
                results = list(executor.map(_run_single, tasks))
                for res in results:
                    task_results.append(res)
                    if res.passed:
                        passed_count += 1
                    if res.compile_success:
                        compile_count += 1
        else:
            for task in tasks:
                res = _run_single(task)
                task_results.append(res)
                if res.passed:
                    passed_count += 1
                if res.compile_success:
                    compile_count += 1

        total = max(1, len(tasks))
        pass_at_1 = passed_count / total
        compile_rate = compile_count / total

        telemetry_rep = self.telemetry_profiler.get_report()
        cost_rep = self.cost_tracker.get_report()

        logger.info(
            f"Suite '{suite.name}' completed: pass@1={pass_at_1:.3f} compile_rate={compile_rate:.3f} "
            f"({passed_count}/{total}) | throughput={telemetry_rep.throughput_tok_s:.1f} tok/s "
            f"p50={telemetry_rep.latency_p50_ms:.1f}ms peak_vram={telemetry_rep.peak_vram_gb:.2f}GB"
        )

        summary_metrics: dict[str, Any] = {
            "pass_at_1": pass_at_1,
            "compile_rate": compile_rate,
            "latency_p50_ms": telemetry_rep.latency_p50_ms,
            "latency_p95_ms": telemetry_rep.latency_p95_ms,
            "throughput_tok_s": telemetry_rep.throughput_tok_s,
            "generation_throughput_tok_s": telemetry_rep.generation_throughput_tok_s,
            "peak_vram_gb": telemetry_rep.peak_vram_gb,
            "cost_per_1m_tokens": cost_rep.cost_per_1m_tokens,
            "telemetry": telemetry_rep.to_dict(),
            "cost": cost_rep.to_dict(),
        }

        return EvaluationSuiteResult(
            suite_name=suite.name,
            benchmark_version=suite.version,
            total_tasks=len(tasks),
            passed_tasks=passed_count,
            pass_at_1=pass_at_1,
            compile_rate=compile_rate,
            unit_test_pass_rate=pass_at_1,
            task_results=task_results,
            summary_metrics=summary_metrics,
        )

    def get_telemetry_report(self) -> InferenceTelemetryReport:
        """Return the current aggregated telemetry report."""
        return self.telemetry_profiler.get_report()

    def get_cost_report(self) -> CostSummaryReport:
        """Return the current financial cost report."""
        return self.cost_tracker.get_report()
