"""Pipeline Progress Tracking, Sub-Stage Trackers, and ETA Estimation."""

from __future__ import annotations

import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from vipym.observability.logging import bind_context, emit_event, get_logger, unbind_context

logger = get_logger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable time string (e.g. '45s', '2m 15s', '1h 10m 05s')."""
    if seconds < 0:
        return "0s"
    s = int(seconds)
    hours = s // 3600
    minutes = (s % 3600) // 60
    secs = s % 60

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def create_progress_bar(
    console: Console | None = None,
    transient: bool = False,
    disable: bool = False,
) -> Progress:
    """Factory creating a standard Rich Progress bar with ETA and elapsed timing."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("• ETA:"),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
        disable=disable or not sys.stdout.isatty(),
    )


class PipelineProgressTracker:
    """Tracks overall experiment pipeline progress across multiple compression and evaluation stages."""

    def __init__(
        self,
        total_stages: int,
        pipeline_name: str = "compression_pipeline",
        pipeline_id: str | None = None,
        use_rich_progress: bool = False,
        console: Console | None = None,
    ) -> None:
        self.total_stages = max(1, total_stages)
        self.pipeline_name = pipeline_name
        self.pipeline_id = pipeline_id or pipeline_name
        self.completed_stages = 0
        self.current_stage_name: str | None = None
        self.pipeline_start_time = time.perf_counter()
        self.current_stage_start_time: float | None = None
        self.stage_durations: dict[str, float] = {}

        self.use_rich_progress = use_rich_progress
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        if self.use_rich_progress:
            self._progress = create_progress_bar(console=console)

    def __enter__(self) -> PipelineProgressTracker:
        if self._progress is not None:
            self._progress.start()
            self._task_id = self._progress.add_task(
                f"[bold]{self.pipeline_name}[/bold]",
                total=self.total_stages,
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._progress is not None:
            self._progress.stop()

    def start_stage(self, stage_name: str, stage_type: str = "compression") -> None:
        """Mark the start of a stage, bind logging context, and emit stage_started event."""
        self.current_stage_name = stage_name
        self.current_stage_start_time = time.perf_counter()

        bind_context(
            pipeline_id=self.pipeline_id,
            stage_name=stage_name,
            stage_type=stage_type,
        )
        emit_event(
            "stage_started",
            stage_name=stage_name,
            stage_type=stage_type,
            stage_index=self.completed_stages + 1,
            total_stages=self.total_stages,
            pipeline_id=self.pipeline_id,
        )

        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                description=f"[bold]{self.pipeline_name}[/bold] -> {stage_name}",
            )

    def complete_stage(
        self,
        stage_name: str,
        metrics: dict[str, Any] | None = None,
    ) -> float:
        """Mark completion of a stage, record duration, and emit stage_completed event."""
        t_end = time.perf_counter()
        t_start = self.current_stage_start_time or self.pipeline_start_time
        duration = t_end - t_start
        duration_rounded = round(duration, 2)
        self.stage_durations[stage_name] = duration_rounded
        self.completed_stages += 1
        self.current_stage_name = None
        self.current_stage_start_time = None

        eta_sec = self.get_estimated_remaining_seconds()

        emit_event(
            "stage_completed",
            stage_name=stage_name,
            duration_seconds=duration_rounded,
            completed_stages=self.completed_stages,
            total_stages=self.total_stages,
            progress_pct=self.get_progress_percentage(),
            eta_seconds=round(eta_sec, 2),
            pipeline_id=self.pipeline_id,
            metrics=metrics or {},
        )

        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                advance=1,
                description=f"[bold]{self.pipeline_name}[/bold] ({self.completed_stages}/{self.total_stages})",
            )

        unbind_context("stage_name", "stage_type")
        return duration_rounded

    def fail_stage(self, stage_name: str, error: Exception | str) -> float:
        """Mark a stage as failed and emit stage_failed event."""
        t_end = time.perf_counter()
        t_start = self.current_stage_start_time or self.pipeline_start_time
        duration = round(t_end - t_start, 2)
        self.stage_durations[stage_name] = duration

        emit_event(
            "stage_failed",
            level="error",
            stage_name=stage_name,
            error=str(error),
            duration_seconds=duration,
            completed_stages=self.completed_stages,
            total_stages=self.total_stages,
            pipeline_id=self.pipeline_id,
        )

        unbind_context("stage_name", "stage_type")
        return duration

    @contextmanager
    def track_stage(
        self,
        stage_name: str,
        stage_type: str = "compression",
        metrics_dict: dict[str, Any] | None = None,
    ) -> Generator[None, None, None]:
        """Context manager to cleanly track a pipeline stage."""
        self.start_stage(stage_name=stage_name, stage_type=stage_type)
        try:
            yield
            self.complete_stage(stage_name=stage_name, metrics=metrics_dict)
        except Exception as e:
            self.fail_stage(stage_name=stage_name, error=e)
            raise

    def get_elapsed_seconds(self) -> float:
        """Total elapsed seconds since pipeline start."""
        return time.perf_counter() - self.pipeline_start_time

    def get_progress_percentage(self) -> float:
        """Percentage of stages completed (0.0 - 100.0)."""
        return round((self.completed_stages / self.total_stages) * 100.0, 1)

    def get_estimated_remaining_seconds(self) -> float:
        """Compute estimated time of arrival (ETA) in seconds based on completed stage timings."""
        if self.completed_stages == 0:
            return 0.0

        elapsed = self.get_elapsed_seconds()
        avg_stage_time = elapsed / self.completed_stages
        remaining_stages = max(0, self.total_stages - self.completed_stages)
        return remaining_stages * avg_stage_time

    def get_status_summary(self) -> str:
        """Format an informative one-line status string with elapsed time and ETA."""
        elapsed_str = format_duration(self.get_elapsed_seconds())
        eta_str = format_duration(self.get_estimated_remaining_seconds())
        current = self.current_stage_name or "Idle"

        return (
            f"[{self.pipeline_name}] Stage {self.completed_stages}/{self.total_stages} ({current}) | "
            f"Progress: {self.get_progress_percentage()}% | Elapsed: {elapsed_str} | ETA: {eta_str}"
        )


class LayerProgressTracker:
    """Tracks sub-stage progress for layer-by-layer quantization (e.g. GPTQ / AWQ / FP8)."""

    def __init__(
        self,
        total_layers: int,
        stage_name: str = "quantization",
        use_rich_progress: bool = False,
        console: Console | None = None,
    ) -> None:
        self.total_layers = max(1, total_layers)
        self.stage_name = stage_name
        self.completed_layers = 0
        self.start_time = time.perf_counter()

        self.use_rich_progress = use_rich_progress
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        if self.use_rich_progress:
            self._progress = create_progress_bar(console=console)

    def __enter__(self) -> LayerProgressTracker:
        if self._progress is not None:
            self._progress.start()
            self._task_id = self._progress.add_task(
                f"[cyan]{self.stage_name}[/cyan] (layers)",
                total=self.total_layers,
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._progress is not None:
            self._progress.stop()

    def step(self, layer_idx: int | None = None, layer_name: str = "") -> None:
        """Advance layer progress by 1 or set to layer_idx."""
        if layer_idx is not None:
            self.completed_layers = layer_idx + 1
        else:
            self.completed_layers += 1

        eta = self.get_eta_seconds()
        rate = self.get_layers_per_second()
        desc = layer_name or f"Layer {self.completed_layers}"

        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=self.completed_layers,
                description=f"[cyan]{self.stage_name}[/cyan] -> {desc}",
            )
        else:
            logger.info(
                f"[{self.stage_name}] Layer {self.completed_layers}/{self.total_layers} "
                f"({desc}) | {rate:.2f} layers/s | ETA: {format_duration(eta)}"
            )

    def get_layers_per_second(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        return (self.completed_layers / elapsed) if elapsed > 0 else 0.0

    def get_progress_percentage(self) -> float:
        return round((self.completed_layers / self.total_layers) * 100.0, 1)

    def get_eta_seconds(self) -> float:
        rate = self.get_layers_per_second()
        if rate <= 0:
            return 0.0
        remaining = max(0, self.total_layers - self.completed_layers)
        return remaining / rate


class ExpertProgressTracker:
    """Tracks sub-stage progress for per-expert MoE pruning or merging."""

    def __init__(
        self,
        total_experts: int,
        stage_name: str = "moe_compression",
        use_rich_progress: bool = False,
        console: Console | None = None,
    ) -> None:
        self.total_experts = max(1, total_experts)
        self.stage_name = stage_name
        self.completed_experts = 0
        self.start_time = time.perf_counter()

        self.use_rich_progress = use_rich_progress
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        if self.use_rich_progress:
            self._progress = create_progress_bar(console=console)

    def __enter__(self) -> ExpertProgressTracker:
        if self._progress is not None:
            self._progress.start()
            self._task_id = self._progress.add_task(
                f"[magenta]{self.stage_name}[/magenta] (experts)",
                total=self.total_experts,
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._progress is not None:
            self._progress.stop()

    def step(self, expert_idx: int | None = None, expert_id: str = "") -> None:
        """Advance expert progress by 1 or set to expert_idx."""
        if expert_idx is not None:
            self.completed_experts = expert_idx + 1
        else:
            self.completed_experts += 1

        eta = self.get_eta_seconds()
        rate = self.get_experts_per_second()
        desc = expert_id or f"Exp_{self.completed_experts}"

        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=self.completed_experts,
                description=f"[magenta]{self.stage_name}[/magenta] -> {desc}",
            )
        else:
            logger.info(
                f"[{self.stage_name}] Expert {self.completed_experts}/{self.total_experts} "
                f"({desc}) | {rate:.1f} exp/s | ETA: {format_duration(eta)}"
            )

    def get_experts_per_second(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        return (self.completed_experts / elapsed) if elapsed > 0 else 0.0

    def get_progress_percentage(self) -> float:
        return round((self.completed_experts / self.total_experts) * 100.0, 1)

    def get_eta_seconds(self) -> float:
        rate = self.get_experts_per_second()
        if rate <= 0:
            return 0.0
        remaining = max(0, self.total_experts - self.completed_experts)
        return remaining / rate


class StepProgressTracker:
    """Tracks sub-stage progress for iterative distillation or fine-tuning steps."""

    def __init__(
        self,
        total_steps: int,
        stage_name: str = "distillation",
        use_rich_progress: bool = False,
        console: Console | None = None,
    ) -> None:
        self.total_steps = max(1, total_steps)
        self.stage_name = stage_name
        self.completed_steps = 0
        self.start_time = time.perf_counter()

        self.use_rich_progress = use_rich_progress
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        if self.use_rich_progress:
            self._progress = create_progress_bar(console=console)

    def __enter__(self) -> StepProgressTracker:
        if self._progress is not None:
            self._progress.start()
            self._task_id = self._progress.add_task(
                f"[green]{self.stage_name}[/green] (steps)",
                total=self.total_steps,
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._progress is not None:
            self._progress.stop()

    def step(
        self,
        step_idx: int | None = None,
        step_info: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Advance training step."""
        if step_idx is not None:
            self.completed_steps = step_idx + 1
        else:
            self.completed_steps += 1

        eta = self.get_eta_seconds()
        rate = self.get_steps_per_second()
        desc = step_info or f"Step {self.completed_steps}"

        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=self.completed_steps,
                description=f"[green]{self.stage_name}[/green] -> {desc}",
            )
        else:
            metric_str = f" | {metrics}" if metrics else ""
            logger.info(
                f"[{self.stage_name}] Step {self.completed_steps}/{self.total_steps} "
                f"({desc}) | {rate:.2f} steps/s | ETA: {format_duration(eta)}{metric_str}"
            )

    def get_steps_per_second(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        return (self.completed_steps / elapsed) if elapsed > 0 else 0.0

    def get_progress_percentage(self) -> float:
        return round((self.completed_steps / self.total_steps) * 100.0, 1)

    def get_eta_seconds(self) -> float:
        rate = self.get_steps_per_second()
        if rate <= 0:
            return 0.0
        remaining = max(0, self.total_steps - self.completed_steps)
        return remaining / rate


__all__ = [
    "ExpertProgressTracker",
    "LayerProgressTracker",
    "PipelineProgressTracker",
    "StepProgressTracker",
    "create_progress_bar",
    "format_duration",
]
