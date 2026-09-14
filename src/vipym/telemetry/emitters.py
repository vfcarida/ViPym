"""Enterprise Telemetry Emitters for Weights & Biases (WandB) and MLflow.

Provides plug-and-play tracking of hyperparameter configurations, benchmark curves,
Pareto frontiers, and model artifacts into enterprise MLOps platforms.
Includes zero-dependency graceful fallbacks and offline in-memory execution.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from vipym.core.logger import get_logger

logger = get_logger(__name__)


class TelemetryEmitter(ABC):
    """Abstract interface for enterprise experiment tracking and telemetry emitters."""

    @abstractmethod
    def start_run(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize and start a remote tracking run."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics series at current or specified step."""
        pass

    @abstractmethod
    def log_artifact(
        self,
        artifact_path: str | Path,
        artifact_name: str | None = None,
        artifact_type: str = "model",
    ) -> None:
        """Upload and register a model checkpoint, report, or dataset artifact."""
        pass

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """Finalize and close the tracking run."""
        pass

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Return True if a tracking run is currently active."""
        pass


class NoOpTelemetryEmitter(TelemetryEmitter):
    """Zero-overhead default emitter used when no remote tracking service is configured."""

    def __init__(self) -> None:
        self._active = False
        self.recorded_metrics: list[dict[str, Any]] = []

    def start_run(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self._active = True
        logger.debug("NoOpTelemetryEmitter: Started run '%s'", run_name)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._active:
            self.recorded_metrics.append({"metrics": metrics, "step": step})

    def log_artifact(
        self,
        artifact_path: str | Path,
        artifact_name: str | None = None,
        artifact_type: str = "model",
    ) -> None:
        logger.debug("NoOpTelemetryEmitter: Registered artifact '%s'", artifact_path)

    def end_run(self, status: str = "FINISHED") -> None:
        self._active = False
        logger.debug("NoOpTelemetryEmitter: Ended run with status %s", status)

    @property
    def is_active(self) -> bool:
        return self._active


class WandbTelemetryEmitter(TelemetryEmitter):
    """Weights & Biases telemetry emitter with offline resilience and artifact tracking."""

    def __init__(
        self,
        project: str = "vipym-compression",
        entity: str | None = None,
        offline: bool = False,
    ) -> None:
        self.project = project
        self.entity = entity
        self.offline = offline or (os.environ.get("WANDB_MODE") == "offline")
        self._run: Any = None
        self.in_memory_metrics: list[dict[str, Any]] = []

    def start_run(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        tags_list = list(dict.fromkeys(["vipym", *(tags or [])]))
        try:
            import wandb

            mode = "offline" if self.offline else None
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                config=config or {},
                tags=tags_list,
                mode=mode,
                reinit=True,
            )
            logger.info("WandbTelemetryEmitter: Initialized Weights & Biases run '%s'", run_name)
        except Exception as e:
            logger.warning(
                "WandB not available or failed to initialize: %s. Using in-memory fallback.", e
            )
            self._run = "in_memory"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.is_active:
            return

        self.in_memory_metrics.append({"metrics": metrics, "step": step})
        if self._run != "in_memory":
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception as e:
                logger.debug("Failed logging metrics to WandB: %s", e)

    def log_artifact(
        self,
        artifact_path: str | Path,
        artifact_name: str | None = None,
        artifact_type: str = "model",
    ) -> None:
        if not self.is_active:
            return

        path = Path(artifact_path)
        name = artifact_name or path.name

        if self._run != "in_memory":
            try:
                import wandb

                artifact = wandb.Artifact(name=name, type=artifact_type)
                if path.is_dir():
                    artifact.add_dir(str(path))
                elif path.is_file():
                    artifact.add_file(str(path))
                wandb.log_artifact(artifact)
                logger.info("WandbTelemetryEmitter: Uploaded artifact '%s'", name)
            except Exception as e:
                logger.debug("Failed logging artifact to WandB: %s", e)

    def end_run(self, status: str = "FINISHED") -> None:
        if self._run and self._run != "in_memory":
            try:
                import wandb

                wandb.finish(exit_code=0 if status == "FINISHED" else 1)
            except Exception as e:
                logger.debug("Failed finishing WandB run: %s", e)
        self._run = None

    @property
    def is_active(self) -> bool:
        return self._run is not None


class MLflowTelemetryEmitter(TelemetryEmitter):
    """MLflow telemetry emitter for tracking experiment runs, metrics, and models."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "vipym-compression",
    ) -> None:
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        self.experiment_name = experiment_name
        self._active_run: Any = None
        self.in_memory_metrics: list[dict[str, Any]] = []

    def start_run(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        try:
            import mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            mlflow.set_experiment(self.experiment_name)
            run = mlflow.start_run(run_name=run_name)
            self._active_run = run

            # Flatten config parameters for MLflow logging
            if config:
                flat_params = {}
                for k, v in config.items():
                    if isinstance(v, (str, int, float, bool)):
                        flat_params[k] = v
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, (str, int, float, bool)):
                                flat_params[f"{k}.{sub_k}"] = sub_v
                if flat_params:
                    mlflow.log_params(flat_params)

            if tags:
                mlflow.set_tags({f"tag_{i}": t for i, t in enumerate(tags)})

            logger.info("MLflowTelemetryEmitter: Started MLflow run '%s'", run_name)
        except Exception as e:
            logger.warning(
                "MLflow not available or failed to initialize: %s. Using in-memory fallback.", e
            )
            self._active_run = "in_memory"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.is_active:
            return

        self.in_memory_metrics.append({"metrics": metrics, "step": step})
        if self._active_run != "in_memory":
            try:
                import mlflow

                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.debug("Failed logging metrics to MLflow: %s", e)

    def log_artifact(
        self,
        artifact_path: str | Path,
        artifact_name: str | None = None,
        artifact_type: str = "model",
    ) -> None:
        if not self.is_active:
            return

        path = Path(artifact_path)
        if self._active_run != "in_memory":
            try:
                import mlflow

                if path.is_dir():
                    mlflow.log_artifacts(str(path), artifact_path=artifact_name)
                elif path.is_file():
                    mlflow.log_artifact(str(path), artifact_path=artifact_name)
                logger.info("MLflowTelemetryEmitter: Logged artifact '%s'", path)
            except Exception as e:
                logger.debug("Failed logging artifact to MLflow: %s", e)

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active_run and self._active_run != "in_memory":
            try:
                import mlflow

                mlflow.end_run(status=status)
            except Exception as e:
                logger.debug("Failed ending MLflow run: %s", e)
        self._active_run = None

    @property
    def is_active(self) -> bool:
        return self._active_run is not None


class CompositeTelemetryEmitter(TelemetryEmitter):
    """Dispatches telemetry concurrently to multiple configured tracking platforms."""

    def __init__(self, emitters: list[TelemetryEmitter]) -> None:
        self.emitters = emitters

    def start_run(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        for em in self.emitters:
            em.start_run(run_name, config, tags)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for em in self.emitters:
            em.log_metrics(metrics, step)

    def log_artifact(
        self,
        artifact_path: str | Path,
        artifact_name: str | None = None,
        artifact_type: str = "model",
    ) -> None:
        for em in self.emitters:
            em.log_artifact(artifact_path, artifact_name, artifact_type)

    def end_run(self, status: str = "FINISHED") -> None:
        for em in self.emitters:
            em.end_run(status)

    @property
    def is_active(self) -> bool:
        return any(em.is_active for em in self.emitters)


def get_telemetry_emitter(
    provider: str | None = None,
    **kwargs: Any,
) -> TelemetryEmitter:
    """Factory helper to obtain the configured telemetry emitter."""
    p = (provider or os.environ.get("VIPYM_TELEMETRY_PROVIDER", "noop")).lower()

    if p in ("wandb", "weights_and_biases"):
        return WandbTelemetryEmitter(**kwargs)
    elif p == "mlflow":
        return MLflowTelemetryEmitter(**kwargs)
    elif p in ("all", "composite"):
        return CompositeTelemetryEmitter(
            [WandbTelemetryEmitter(**kwargs), MLflowTelemetryEmitter(**kwargs)]
        )
    return NoOpTelemetryEmitter()
