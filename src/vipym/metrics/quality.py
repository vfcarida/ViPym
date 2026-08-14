"""Quality, Performance, Memory and Footprint metrics calculations."""

from typing import Any

import numpy as np
import pydantic

from vipym.interfaces.metrics import Metric, TelemetrySnapshot


class QualityMetrics(pydantic.BaseModel):
    pass_at_1: float
    compile_rate: float
    unit_test_success_rate: float
    total_tasks_evaluated: int


class QualityMetricCalculator(Metric):
    @property
    def name(self) -> str:
        return "quality"

    def calculate(self, task_results: list[Any]) -> dict[str, float]:
        if not task_results:
            return {"pass_at_1": 0.0, "compile_rate": 0.0}
        total = len(task_results)
        passed = sum(1 for t in task_results if getattr(t, "passed", False))
        compiled = sum(1 for t in task_results if getattr(t, "compile_success", False))
        return {
            "pass_at_1": float(passed / total),
            "compile_rate": float(compiled / total),
            "total_tasks": float(total),
        }


class PerformanceMetricCalculator(Metric):
    @property
    def name(self) -> str:
        return "performance"

    def calculate(self, latency_records_ms: list[float]) -> dict[str, float]:
        if not latency_records_ms:
            return {"ttft_p50_ms": 0.0, "ttft_p95_ms": 0.0, "itl_p50_ms": 0.0}
        arr = np.array(latency_records_ms)
        return {
            "latency_p50_ms": float(np.percentile(arr, 50)),
            "latency_p90_ms": float(np.percentile(arr, 90)),
            "latency_p95_ms": float(np.percentile(arr, 95)),
            "latency_mean_ms": float(np.mean(arr)),
        }


class MemoryMetricCalculator(Metric):
    @property
    def name(self) -> str:
        return "memory"

    def calculate(self, telemetry: TelemetrySnapshot) -> dict[str, float]:
        return {
            "peak_gpu_vram_allocated_gb": telemetry.peak_gpu_memory_allocated_bytes / (1024**3),
            "peak_gpu_vram_reserved_gb": telemetry.peak_gpu_memory_reserved_bytes / (1024**3),
            "host_ram_rss_gb": telemetry.host_cpu_memory_rss_bytes / (1024**3),
        }


class FootprintMetricCalculator(Metric):
    @property
    def name(self) -> str:
        return "footprint"

    def calculate(self, original_bytes: int, compressed_bytes: int) -> dict[str, float]:
        ratio = (original_bytes / max(1, compressed_bytes)) if compressed_bytes > 0 else 1.0
        reduction_pct = ((original_bytes - compressed_bytes) / max(1, original_bytes)) * 100.0
        return {
            "original_disk_gb": original_bytes / (1024**3),
            "compressed_disk_gb": compressed_bytes / (1024**3),
            "compression_ratio": float(ratio),
            "storage_reduction_percent": float(reduction_pct),
        }
