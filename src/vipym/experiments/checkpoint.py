"""Experiment Checkpointer for resumability after interruption or failure."""

import json
from pathlib import Path
from typing import Any

import pydantic

from vipym.core.logger import get_logger

logger = get_logger(__name__)


class ExperimentCheckpoint(pydantic.BaseModel):
    """Snapshot of completed stages, intermediate paths, and partial metrics."""

    experiment_id: str
    baseline_completed: bool = False
    baseline_score: float | None = None
    baseline_metrics: dict[str, Any] = pydantic.Field(default_factory=dict)
    compressed_artifact_path: str | None = None
    compressed_methods_applied: list[str] = pydantic.Field(default_factory=list)
    evaluation_completed: bool = False
    evaluation_results: dict[str, Any] = pydantic.Field(default_factory=dict)
    analysis_completed: bool = False
    pareto_points: list[dict[str, Any]] = pydantic.Field(default_factory=list)


class CheckpointManager:
    """Manages reading and writing stage checkpoints to disk."""

    def __init__(self, checkpoint_path: Path | str) -> None:
        self.checkpoint_path = Path(checkpoint_path)

    def load(self, experiment_id: str) -> ExperimentCheckpoint:
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, encoding="utf-8") as f:
                    data = json.load(f)
                return ExperimentCheckpoint(**data)
            except Exception as e:
                logger.warning(f"Failed to read checkpoint at {self.checkpoint_path}: {e}")
        return ExperimentCheckpoint(experiment_id=experiment_id)

    def save(self, checkpoint: ExperimentCheckpoint) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            f.write(checkpoint.model_dump_json(indent=2))
