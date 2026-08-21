"""Central ViPym Experiment Runner and Orchestration Engine (Thin Wrapper)."""

from pathlib import Path

from vipym.config.schema import ViPymExperimentConfig
from vipym.experiments.runner import (
    ExperimentRunSummary,
    ResumableExperimentRunner,
)

# For backward compatibility
ExperimentExecutionResult = ExperimentRunSummary


class ViPymRunner:
    """Thin convenience wrapper around ResumableExperimentRunner with checkpointing disabled."""

    def __init__(
        self, config: ViPymExperimentConfig, artifacts_dir: Path | str = "./artifacts"
    ) -> None:
        self._runner = ResumableExperimentRunner(
            config=config,
            artifacts_dir=artifacts_dir,
            checkpoint_enabled=False,
        )

    @property
    def config(self) -> ViPymExperimentConfig:
        return self._runner.config

    @property
    def artifacts_dir(self) -> Path:
        return self._runner.exp_dir

    @property
    def manifest(self):
        return self._runner.manifest

    @property
    def cost_model(self):
        return self._runner.cost_calculator

    def run(self) -> ExperimentRunSummary:
        return self._runner.run(resume=False)


__all__ = ["ExperimentExecutionResult", "ExperimentRunSummary", "ViPymRunner"]
