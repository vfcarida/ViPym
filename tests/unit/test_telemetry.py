"""Unit tests for P014 — Real Inference Telemetry, Profiler, Cost Tracker, and Runner Instrumentation.

Test classes:
  TestInferenceProfiler                — Latency distributions (p50/p90/p95/p99), throughput (tok/s), TTFT, warmup exclusion
  TestInferenceCostTracker             — Hardware hourly pricing, cost per 1M tokens, API baseline comparison
  TestBenchmarkRunnerInstrumentation   — Real-time measurement capture in BenchmarkRunner
  TestTelemetryJSONSerialization       — JSON reports serialization & schema validation
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from vipym.evaluation.runner import BenchmarkRunner
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig
from vipym.interfaces.evaluation import EvaluationSuiteResult
from vipym.interfaces.inference import GenerationRequest, GenerationResponse, InferenceBackend
from vipym.telemetry.cost_tracker import CostSummaryReport, InferenceCostTracker
from vipym.telemetry.profiler import (
    InferenceProfiler,
)

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


class TimedMockInferenceBackend(InferenceBackend):
    def __init__(
        self, latency_ms: float = 20.0, prompt_tokens: int = 100, completion_tokens: int = 50
    ) -> None:
        self.latency_ms = latency_ms
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        time.sleep(self.latency_ms / 1000.0)
        return GenerationResponse(
            generated_text="def solution(): return 42",
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            time_to_first_token_ms=self.latency_ms * 0.3,
            inter_token_latency_ms=1.5,
            total_time_ms=self.latency_ms,
        )

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        return self.generate(request)

    def stop(self) -> None:
        pass


# ============================================================
# TestInferenceProfiler
# ============================================================


class TestInferenceProfiler:
    def test_record_request_and_percentiles(self):
        profiler = InferenceProfiler(model_variant="kimi-k3-4bit", warmup_requests=0)

        # Record 10 requests with known latencies: 10ms, 20ms, ..., 100ms
        for i in range(1, 11):
            profiler.record_request(
                request_id=f"req_{i}",
                prompt_tokens=100,
                completion_tokens=50,
                latency_ms=float(i * 10),
                ttft_ms=5.0,
                inter_token_latency_ms=1.0,
                peak_vram_bytes=8 * (1024**3),  # 8 GB
            )

        report = profiler.get_report()
        assert report.model_variant == "kimi-k3-4bit"
        assert report.total_requests == 10
        assert report.measured_requests == 10
        assert report.warmup_requests == 0

        # Percentiles
        assert report.min_latency_ms == 10.0
        assert report.max_latency_ms == 100.0
        assert report.mean_latency_ms == pytest.approx(55.0, abs=1e-3)
        assert report.latency_p50_ms == pytest.approx(50.0, abs=1e-3)
        assert report.latency_p90_ms == pytest.approx(90.0, abs=1e-3)
        assert report.latency_p95_ms == pytest.approx(100.0, abs=1e-3)
        assert report.latency_p99_ms == pytest.approx(100.0, abs=1e-3)
        assert report.peak_vram_gb == pytest.approx(8.0, abs=1e-3)

    def test_throughput_calculations(self):
        profiler = InferenceProfiler(warmup_requests=0)

        # 5 requests, each 100 prompt tokens, 50 completion tokens, 100ms latency
        # Total tokens = 5 * 150 = 750 tokens
        # Total time = 5 * 0.1s = 0.5s
        # Total throughput = 750 / 0.5 = 1500 tok/s
        # Gen throughput = 250 / 0.5 = 500 tok/s
        # Prompt throughput = 500 / 0.5 = 1000 tok/s
        for i in range(5):
            profiler.record_request(
                request_id=f"req_{i}",
                prompt_tokens=100,
                completion_tokens=50,
                latency_ms=100.0,
            )

        report = profiler.get_report()
        assert report.throughput_tok_s == pytest.approx(1500.0, abs=1.0)
        assert report.generation_throughput_tok_s == pytest.approx(500.0, abs=1.0)
        assert report.prompt_throughput_tok_s == pytest.approx(1000.0, abs=1.0)

    def test_warmup_requests_exclusion(self):
        profiler = InferenceProfiler(warmup_requests=3)

        # 3 warmup requests with slow latencies (500ms)
        for i in range(3):
            profiler.record_request(
                request_id=f"warmup_{i}",
                prompt_tokens=10,
                completion_tokens=10,
                latency_ms=500.0,
            )

        # 5 measured requests with fast latencies (20ms)
        for i in range(5):
            profiler.record_request(
                request_id=f"measured_{i}",
                prompt_tokens=10,
                completion_tokens=10,
                latency_ms=20.0,
            )

        report = profiler.get_report()
        assert report.total_requests == 8
        assert report.measured_requests == 5
        assert report.warmup_requests == 3
        # Statistics should only reflect measured requests (20ms)
        assert report.mean_latency_ms == pytest.approx(20.0, abs=1e-3)
        assert report.latency_p50_ms == pytest.approx(20.0, abs=1e-3)

    def test_measure_context_manager(self):
        profiler = InferenceProfiler(warmup_requests=0)

        with profiler.measure("ctx_test", prompt_tokens=40) as meta:
            time.sleep(0.02)  # 20ms
            meta["completion_tokens"] = 30
            meta["ttft_ms"] = 6.0
            meta["inter_token_latency_ms"] = 0.5

        assert len(profiler.records) == 1
        rec = profiler.records[0]
        assert rec.prompt_tokens == 40
        assert rec.completion_tokens == 30
        assert rec.latency_ms >= 15.0
        assert rec.time_to_first_token_ms == 6.0

    def test_empty_profiler_report(self):
        profiler = InferenceProfiler()
        report = profiler.get_report()
        assert report.total_requests == 0
        assert report.throughput_tok_s == 0.0
        assert report.latency_p50_ms == 0.0


# ============================================================
# TestInferenceCostTracker
# ============================================================


class TestInferenceCostTracker:
    def test_cost_accumulation_and_hardware_pricing(self):
        # A100 at $2.50/hr
        tracker = InferenceCostTracker(
            model_variant="kimi-k3-4bit",
            hourly_hardware_rate=2.50,
            baseline_api="gpt-4o",
        )

        # Process 1,000,000 prompt tokens and 500,000 completion tokens in 360 seconds (0.1 hours)
        tracker.record_usage(
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
            duration_seconds=360.0,
        )

        report = tracker.get_report()
        assert isinstance(report, CostSummaryReport)
        assert report.total_tokens == 1_500_000
        assert report.total_time_seconds == 360.0

        # Hardware cost: 0.1 hours * $2.50 = $0.25
        assert report.hardware_cost_usd == pytest.approx(0.25, abs=1e-4)

        # Cost per 1M tokens: ($0.25 / 1.5M) * 1M = $0.1667
        assert report.cost_per_1m_tokens == pytest.approx(0.1667, abs=1e-3)

        # Baseline GPT-4o cost: 1M * $2.50 + 0.5M * $10.00 = $2.50 + $5.00 = $7.50
        assert report.baseline_api_cost_usd == pytest.approx(7.50, abs=1e-3)

        # Savings: (1 - 0.25 / 7.50) = 96.67%
        assert report.cost_savings_percentage == pytest.approx(96.67, abs=1e-2)

    def test_custom_hardware_rates(self):
        tracker = InferenceCostTracker(hardware_type="H100-80GB")
        assert tracker.hourly_rate == 3.50

    def test_reset(self):
        tracker = InferenceCostTracker()
        tracker.record_usage(100, 50, 10.0)
        tracker.reset()
        assert tracker.total_prompt_tokens == 0
        assert tracker.total_completion_tokens == 0
        assert tracker.total_time_seconds == 0.0


# ============================================================
# TestBenchmarkRunnerInstrumentation
# ============================================================


class TestBenchmarkRunnerInstrumentation:
    def test_runner_instruments_real_telemetry(self, sandbox_runner):
        backend = TimedMockInferenceBackend(latency_ms=10.0, prompt_tokens=80, completion_tokens=40)
        runner = BenchmarkRunner(
            sandbox_runner=sandbox_runner,
            model_variant="kimi-k3-compressed",
            warmup_requests=0,
        )

        result = runner.run_suite("humaneval", backend=backend, task_limit=2)
        assert isinstance(result, EvaluationSuiteResult)

        # Check that telemetry summary metrics are present
        assert "telemetry" in result.summary_metrics
        assert "cost" in result.summary_metrics
        assert "latency_p50_ms" in result.summary_metrics
        assert "throughput_tok_s" in result.summary_metrics
        assert "cost_per_1m_tokens" in result.summary_metrics

        # Telemetry report retrieval
        telem_report = runner.get_telemetry_report()
        assert telem_report.model_variant == "kimi-k3-compressed"
        assert telem_report.total_requests == 2
        assert telem_report.total_tokens == (80 + 40) * 2
        assert telem_report.latency_p50_ms >= 5.0

        # Cost report retrieval
        cost_report = runner.get_cost_report()
        assert cost_report.total_tokens == (80 + 40) * 2
        assert cost_report.hardware_cost_usd > 0.0


# ============================================================
# TestTelemetryJSONSerialization
# ============================================================


class TestTelemetryJSONSerialization:
    def test_telemetry_report_json_roundtrip(self):
        profiler = InferenceProfiler(model_variant="kimi-k3-fp8", warmup_requests=0)
        profiler.record_request(
            request_id="task_1",
            prompt_tokens=50,
            completion_tokens=25,
            latency_ms=15.0,
            ttft_ms=4.0,
        )

        report = profiler.get_report()
        json_str = report.to_json()
        data = json.loads(json_str)

        assert data["model_variant"] == "kimi-k3-fp8"
        assert data["total_requests"] == 1
        assert "latency_p50_ms" in data
        assert "throughput_tok_s" in data

    def test_cost_report_json_roundtrip(self):
        tracker = InferenceCostTracker(model_variant="kimi-k3-fp8")
        tracker.record_usage(5000, 2500, 1.5)

        report = tracker.get_report()
        json_str = report.to_json()
        data = json.loads(json_str)

        assert data["model_variant"] == "kimi-k3-fp8"
        assert data["total_tokens"] == 7500
        assert "hardware_cost_usd" in data
        assert "cost_savings_percentage" in data
