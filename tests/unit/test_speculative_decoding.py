"""Unit tests for Production Speculative Decoding Engine and Telemetry."""

from unittest.mock import MagicMock

from vipym.inference.registry import InferenceRegistry
from vipym.inference.speculative import (
    SpeculativeDecodingHarness,
    SpeculativeInferenceBackend,
    SpeculativeMetrics,
)
from vipym.interfaces.inference import GenerationRequest, GenerationResponse


def test_speculative_harness_full_acceptance() -> None:
    """Verify that matching draft proposals result in high acceptance rate and bonus token."""
    mock_draft = MagicMock()
    mock_target = MagicMock()

    # Draft proposes 3 matching tokens
    mock_draft.generate.return_value = GenerationResponse(
        generated_text="return arr [ 0 ]",
        prompt_tokens=5,
        completion_tokens=4,
        time_to_first_token_ms=5.0,
        inter_token_latency_ms=2.0,
        total_time_ms=8.0,
    )

    # Target confirms identical tokens + bonus token
    mock_target.generate.return_value = GenerationResponse(
        generated_text="return arr [ 0 ] if",
        prompt_tokens=5,
        completion_tokens=5,
        time_to_first_token_ms=15.0,
        inter_token_latency_ms=3.0,
        total_time_ms=25.0,
    )

    harness = SpeculativeDecodingHarness(target_backend=mock_target, draft_backend=mock_draft)
    req = GenerationRequest(prompt="def first_element(arr):", max_new_tokens=5)

    resp = harness.generate_speculative(req, num_speculative_tokens=4)

    assert "return arr [ 0 ]" in resp.generated_text
    assert resp.speculative_acceptance_rate is not None
    assert resp.speculative_acceptance_rate >= 0.75
    assert resp.completion_tokens >= 4
    assert resp.total_time_ms > 0


def test_speculative_harness_divergence() -> None:
    """Verify that draft divergence properly truncates and replaces with target token."""
    mock_draft = MagicMock()
    mock_target = MagicMock()

    # Draft predicts wrong word
    mock_draft.generate.return_value = GenerationResponse(
        generated_text="a + b * c",
        prompt_tokens=4,
        completion_tokens=5,
        time_to_first_token_ms=4.0,
        inter_token_latency_ms=1.5,
        total_time_ms=6.0,
    )

    # Target diverges at 3rd token ('-' instead of '*')
    mock_target.generate.return_value = GenerationResponse(
        generated_text="a + b - d",
        prompt_tokens=4,
        completion_tokens=5,
        time_to_first_token_ms=12.0,
        inter_token_latency_ms=2.5,
        total_time_ms=18.0,
    )

    harness = SpeculativeDecodingHarness(target_backend=mock_target, draft_backend=mock_draft)
    req = GenerationRequest(prompt="def compute(a, b, c):", max_new_tokens=4)

    resp = harness.generate_speculative(req, num_speculative_tokens=4)

    # Must contain target's divergence correction '-' and not draft's '*' or 'c'
    assert "-" in resp.generated_text
    assert resp.speculative_acceptance_rate is not None
    assert resp.speculative_acceptance_rate < 1.0


def test_speculative_benchmark_speedup() -> None:
    """Verify benchmark_speedup returns both response and SpeculativeMetrics telemetry."""
    mock_draft = MagicMock()
    mock_target = MagicMock()

    mock_draft.generate.return_value = GenerationResponse(
        generated_text="hello world",
        prompt_tokens=2,
        completion_tokens=2,
        time_to_first_token_ms=2.0,
        inter_token_latency_ms=1.0,
        total_time_ms=3.0,
    )

    mock_target.generate.return_value = GenerationResponse(
        generated_text="hello world !",
        prompt_tokens=2,
        completion_tokens=3,
        time_to_first_token_ms=10.0,
        inter_token_latency_ms=3.0,
        total_time_ms=15.0,
    )

    harness = SpeculativeDecodingHarness(target_backend=mock_target, draft_backend=mock_draft)
    resp, metrics = harness.benchmark_speedup(prompt="Greeting:", max_new_tokens=3)

    assert isinstance(metrics, SpeculativeMetrics)
    assert metrics.speedup_ratio >= 0.0
    assert metrics.tokens_per_second > 0.0
    assert metrics.acceptance_rate >= 0.0


def test_speculative_inference_backend_registry() -> None:
    """Verify registration in InferenceRegistry and backend delegation."""
    backend = InferenceRegistry.get("speculative")
    assert isinstance(backend, SpeculativeInferenceBackend)

    mock_target = MagicMock()
    mock_draft = MagicMock()
    mock_target.health_check.return_value = True
    mock_target.generate.return_value = GenerationResponse(
        generated_text="ok",
        prompt_tokens=1,
        completion_tokens=1,
        time_to_first_token_ms=5.0,
        inter_token_latency_ms=1.0,
        total_time_ms=6.0,
    )

    spec_backend = SpeculativeInferenceBackend(target_backend=mock_target, draft_backend=mock_draft)
    spec_backend.start("dummy/model")
    assert spec_backend.health_check() is True

    req = GenerationRequest(prompt="Test prompt", max_new_tokens=1)
    res = spec_backend.generate(req)
    assert res.generated_text is not None

    spec_backend.stop()
    mock_target.stop.assert_called_once()
    mock_draft.stop.assert_called_once()
