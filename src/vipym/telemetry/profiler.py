"""Real Inference Telemetry Profiler.

Measures real-time latency distributions (p50, p90, p95, p99), token throughput
(prompt tokens/s, generation tokens/s, total tokens/s), time-to-first-token (TTFT),
inter-token latency (ITL), and GPU VRAM consumption.
Includes configurable warmup request exclusion to eliminate cold-start bias.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

from vipym.core.logger import get_logger

logger = get_logger(__name__)


def _get_cuda_memory() -> tuple[int, int]:
    """Return (allocated_bytes, peak_allocated_bytes) from PyTorch CUDA if available."""
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            return allocated, peak
    except Exception:
        pass
    return 0, 0


@dataclass
class RequestTelemetry:
    """Telemetry recording for an individual inference request."""

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    time_to_first_token_ms: float = 0.0
    inter_token_latency_ms: float = 0.0
    generation_throughput_tok_s: float = 0.0
    peak_vram_bytes: int = 0
    is_warmup: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceTelemetryReport:
    """Aggregated inference performance and hardware telemetry report."""

    model_variant: str
    total_requests: int
    measured_requests: int
    warmup_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    throughput_tok_s: float
    prompt_throughput_tok_s: float
    generation_throughput_tok_s: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    mean_ttft_ms: float
    mean_itl_ms: float
    peak_vram_gb: float
    allocated_vram_gb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class InferenceProfiler:
    """Real-time profiler collecting latency, throughput, TTFT, and GPU VRAM."""

    def __init__(
        self,
        model_variant: str = "default",
        warmup_requests: int = 3,
    ) -> None:
        self.model_variant = model_variant
        self.warmup_requests = warmup_requests
        self.records: list[RequestTelemetry] = []

    def record_request(
        self,
        request_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        ttft_ms: float = 0.0,
        inter_token_latency_ms: float = 0.0,
        peak_vram_bytes: int | None = None,
    ) -> RequestTelemetry:
        """Record telemetry for a single inference generation call."""
        if peak_vram_bytes is None:
            _, peak_vram_bytes = _get_cuda_memory()

        total_tok = prompt_tokens + completion_tokens
        gen_throughput = (completion_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

        is_warmup = len(self.records) < self.warmup_requests

        rec = RequestTelemetry(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tok,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_ms,
            inter_token_latency_ms=inter_token_latency_ms,
            generation_throughput_tok_s=gen_throughput,
            peak_vram_bytes=peak_vram_bytes,
            is_warmup=is_warmup,
        )
        self.records.append(rec)
        return rec

    @contextmanager
    def measure(
        self,
        request_id: str,
        prompt_tokens: int = 0,
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager to measure latency and VRAM for an inference call."""
        t0 = time.perf_counter()
        meta: dict[str, Any] = {
            "completion_tokens": 0,
            "prompt_tokens": prompt_tokens,
            "ttft_ms": 0.0,
            "inter_token_latency_ms": 0.0,
        }
        try:
            yield meta
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            comp_tokens = int(meta.get("completion_tokens", 0))
            p_tokens = int(meta.get("prompt_tokens", prompt_tokens))
            ttft = float(meta.get("ttft_ms", 0.0))
            itl = float(meta.get("inter_token_latency_ms", 0.0))

            self.record_request(
                request_id=request_id,
                prompt_tokens=p_tokens,
                completion_tokens=comp_tokens,
                latency_ms=elapsed_ms,
                ttft_ms=ttft,
                inter_token_latency_ms=itl,
            )

    def get_report(self) -> InferenceTelemetryReport:
        """Compute aggregated statistics and percentile distributions (excluding warmup)."""
        all_records = self.records
        measured = [r for r in all_records if not r.is_warmup]

        # If all requests were within warmup, fall back to evaluating all records
        target = measured if measured else all_records

        if not target:
            return InferenceTelemetryReport(
                model_variant=self.model_variant,
                total_requests=0,
                measured_requests=0,
                warmup_requests=self.warmup_requests,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                throughput_tok_s=0.0,
                prompt_throughput_tok_s=0.0,
                generation_throughput_tok_s=0.0,
                latency_p50_ms=0.0,
                latency_p90_ms=0.0,
                latency_p95_ms=0.0,
                latency_p99_ms=0.0,
                mean_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                mean_ttft_ms=0.0,
                mean_itl_ms=0.0,
                peak_vram_gb=0.0,
                allocated_vram_gb=0.0,
            )

        latencies = sorted(r.latency_ms for r in target)
        total_p_tok = sum(r.prompt_tokens for r in target)
        total_c_tok = sum(r.completion_tokens for r in target)
        total_tok = total_p_tok + total_c_tok
        total_time_s = sum(r.latency_ms for r in target) / 1000.0

        throughput = (total_tok / total_time_s) if total_time_s > 0 else 0.0
        prompt_throughput = (total_p_tok / total_time_s) if total_time_s > 0 else 0.0
        gen_throughput = (total_c_tok / total_time_s) if total_time_s > 0 else 0.0

        ttft_vals = [r.time_to_first_token_ms for r in target if r.time_to_first_token_ms > 0]
        mean_ttft = sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0.0

        itl_vals = [r.inter_token_latency_ms for r in target if r.inter_token_latency_ms > 0]
        mean_itl = sum(itl_vals) / len(itl_vals) if itl_vals else 0.0

        peak_bytes = max(r.peak_vram_bytes for r in target)
        peak_gb = peak_bytes / (1024.0**3)

        curr_alloc, _ = _get_cuda_memory()
        alloc_gb = curr_alloc / (1024.0**3)

        return InferenceTelemetryReport(
            model_variant=self.model_variant,
            total_requests=len(all_records),
            measured_requests=len(target),
            warmup_requests=len(all_records) - len(measured) if measured else 0,
            total_prompt_tokens=total_p_tok,
            total_completion_tokens=total_c_tok,
            total_tokens=total_tok,
            throughput_tok_s=throughput,
            prompt_throughput_tok_s=prompt_throughput,
            generation_throughput_tok_s=gen_throughput,
            latency_p50_ms=self._percentile(latencies, 0.50),
            latency_p90_ms=self._percentile(latencies, 0.90),
            latency_p95_ms=self._percentile(latencies, 0.95),
            latency_p99_ms=self._percentile(latencies, 0.99),
            mean_latency_ms=sum(latencies) / len(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            mean_ttft_ms=mean_ttft,
            mean_itl_ms=mean_itl,
            peak_vram_gb=peak_gb,
            allocated_vram_gb=alloc_gb,
        )

    def _percentile(self, sorted_vals: list[float], pct: float) -> float:
        """Compute percentile from sorted values."""
        if not sorted_vals:
            return 0.0
        idx = int(math.ceil(pct * len(sorted_vals))) - 1
        return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]

    def reset(self) -> None:
        """Clear all recorded telemetry records."""
        self.records.clear()
