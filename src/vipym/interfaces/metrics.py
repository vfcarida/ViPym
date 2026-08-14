"""Interfaces for Telemetry Collectors and Metrics."""

from abc import ABC, abstractmethod
from typing import Any

import pydantic


class TelemetrySnapshot(pydantic.BaseModel):
    """Runtime hardware and efficiency snapshot."""

    peak_gpu_memory_allocated_bytes: int = 0
    peak_gpu_memory_reserved_bytes: int = 0
    host_cpu_memory_rss_bytes: int = 0
    time_to_first_token_p50_ms: float = 0.0
    time_to_first_token_p95_ms: float = 0.0
    inter_token_latency_p50_ms: float = 0.0
    inter_token_latency_p95_ms: float = 0.0
    throughput_tokens_per_second: float = 0.0
    total_requests_processed: int = 0
    total_tokens_generated: int = 0


class Metric(ABC):
    """Abstract interface for a statistical or efficiency metric."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def calculate(self, data: Any) -> dict[str, float]:
        pass
