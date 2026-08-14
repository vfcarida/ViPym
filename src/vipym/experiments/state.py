"""Experiment lifecycle state machine and transition validator."""

import json
from pathlib import Path
from typing import Optional
from vipym.config.constants import ExperimentState
from vipym.config.exceptions import StateTransitionError
from vipym.core.logger import get_logger

logger = get_logger(__name__)

VALID_TRANSITIONS = {
    ExperimentState.CREATED: {ExperimentState.VALIDATED, ExperimentState.FAILED},
    ExperimentState.VALIDATED: {ExperimentState.BASELINE_RUNNING, ExperimentState.FAILED},
    ExperimentState.BASELINE_RUNNING: {ExperimentState.BASELINE_COMPLETED, ExperimentState.FAILED},
    ExperimentState.BASELINE_COMPLETED: {ExperimentState.COMPRESSION_RUNNING, ExperimentState.EVALUATION_RUNNING, ExperimentState.FAILED},
    ExperimentState.COMPRESSION_RUNNING: {ExperimentState.COMPRESSION_COMPLETED, ExperimentState.FAILED},
    ExperimentState.COMPRESSION_COMPLETED: {ExperimentState.INFERENCE_VALIDATED, ExperimentState.EVALUATION_RUNNING, ExperimentState.FAILED},
    ExperimentState.INFERENCE_VALIDATED: {ExperimentState.EVALUATION_RUNNING, ExperimentState.FAILED},
    ExperimentState.EVALUATION_RUNNING: {ExperimentState.EVALUATION_COMPLETED, ExperimentState.FAILED},
    ExperimentState.EVALUATION_COMPLETED: {ExperimentState.ANALYSIS_COMPLETED, ExperimentState.FAILED},
    ExperimentState.ANALYSIS_COMPLETED: {ExperimentState.REPORT_COMPLETED, ExperimentState.FAILED},
    ExperimentState.REPORT_COMPLETED: set(),
    ExperimentState.FAILED: {ExperimentState.VALIDATED, ExperimentState.BASELINE_RUNNING, ExperimentState.COMPRESSION_RUNNING, ExperimentState.EVALUATION_RUNNING},
    ExperimentState.SKIPPED: set(),
}


class ExperimentStateManager:
    """Tracks and persists experiment lifecycle state to disk."""

    def __init__(self, experiment_id: str, state_file_path: Path) -> None:
        self.experiment_id = experiment_id
        self.state_file_path = Path(state_file_path)
        self.current_state: ExperimentState = ExperimentState.CREATED
        self.load_or_init()

    def load_or_init(self) -> None:
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.current_state = ExperimentState(data.get("state", ExperimentState.CREATED.value))
            except Exception as e:
                logger.warning(f"Could not load state file {self.state_file_path}: {e}")
        else:
            self.persist()

    def transition_to(self, next_state: ExperimentState, error_message: Optional[str] = None) -> None:
        allowed = VALID_TRANSITIONS.get(self.current_state, set())
        if next_state not in allowed:
            raise StateTransitionError(
                f"Illegal experiment state transition: '{self.current_state}' -> '{next_state}'. "
                f"Allowed transitions from '{self.current_state}': {[s.value for s in allowed]}"
            )
        old = self.current_state
        self.current_state = next_state
        logger.info(f"Experiment [{self.experiment_id}] state transition: [bold]{old.value}[/bold] -> [bold cyan]{next_state.value}[/bold cyan]")
        self.persist(error_message)

    def persist(self, error_message: Optional[str] = None) -> None:
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment_id": self.experiment_id,
            "state": self.current_state.value,
            "error_message": error_message,
        }
        with open(self.state_file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
