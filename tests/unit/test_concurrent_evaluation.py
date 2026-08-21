"""Unit tests for concurrent and batched sandbox evaluation."""

from __future__ import annotations

import time

import pytest

from vipym.config.schema import EvaluationConfig
from vipym.evaluation.runner import BenchmarkRunner
from vipym.interfaces.inference import GenerationRequest, GenerationResponse, InferenceBackend


class MockSlowInferenceBackend(InferenceBackend):
    """Mock backend simulating network or model generation latency."""

    def start(self, *args, **kwargs) -> None:
        pass

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        time.sleep(0.12)  # Simulate 120ms generation delay
        return GenerationResponse(
            generated_text="    return [abs(a - b) < threshold for a, b in zip(numbers, numbers[1:])]",
            prompt_tokens=10,
            completion_tokens=20,
            time_to_first_token_ms=10.0,
            inter_token_latency_ms=2.0,
            total_time_ms=120.0,
        )

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        return self.generate(request)

    def stop(self) -> None:
        pass


class TestConcurrentEvaluation:
    def test_concurrent_evaluation_speedup(self, monkeypatch: pytest.MonkeyPatch):
        """Verify concurrent execution with max_workers > 1 runs tasks in parallel."""
        monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")

        backend = MockSlowInferenceBackend()

        # 1. Sequential run (max_workers=1)
        cfg_seq = EvaluationConfig(
            suites=["humaneval"],
            task_limit=4,
            allow_unsafe_execution=True,
            isolate_with_gvisor=False,
            max_workers=1,
        )
        runner_seq = BenchmarkRunner(evaluation_config=cfg_seq)
        t0 = time.perf_counter()
        res_seq = runner_seq.run_suite(suite_name="humaneval", backend=backend, task_limit=4)
        seq_duration = time.perf_counter() - t0

        # 2. Concurrent run (max_workers=4)
        cfg_conc = EvaluationConfig(
            suites=["humaneval"],
            task_limit=4,
            allow_unsafe_execution=True,
            isolate_with_gvisor=False,
            max_workers=4,
        )
        runner_conc = BenchmarkRunner(evaluation_config=cfg_conc)
        t0 = time.perf_counter()
        res_conc = runner_conc.run_suite(suite_name="humaneval", backend=backend, task_limit=4)
        conc_duration = time.perf_counter() - t0

        assert len(res_seq.task_results) == 4
        assert len(res_conc.task_results) == 4
        assert conc_duration < seq_duration
