"""Unit tests for experiment state machine and lifecycle transitions."""

import tempfile
from pathlib import Path

import pytest

from vipym.config.constants import ExperimentState
from vipym.config.exceptions import StateTransitionError
from vipym.experiments.state import ExperimentStateManager


def test_valid_state_transitions():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "state.json"
        mgr = ExperimentStateManager("test_exp", state_file)
        assert mgr.current_state == ExperimentState.CREATED

        mgr.transition_to(ExperimentState.VALIDATED)
        assert mgr.current_state == ExperimentState.VALIDATED

        mgr.transition_to(ExperimentState.BASELINE_RUNNING)
        mgr.transition_to(ExperimentState.BASELINE_COMPLETED)
        mgr.transition_to(ExperimentState.COMPRESSION_RUNNING)
        mgr.transition_to(ExperimentState.COMPRESSION_COMPLETED)
        mgr.transition_to(ExperimentState.INFERENCE_VALIDATED)
        mgr.transition_to(ExperimentState.EVALUATION_RUNNING)
        mgr.transition_to(ExperimentState.EVALUATION_COMPLETED)
        mgr.transition_to(ExperimentState.ANALYSIS_COMPLETED)
        mgr.transition_to(ExperimentState.REPORT_COMPLETED)
        assert mgr.current_state == ExperimentState.REPORT_COMPLETED


def test_invalid_state_transition_raises_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "state.json"
        mgr = ExperimentStateManager("test_exp", state_file)

        # Illegal jump: CREATED -> REPORT_COMPLETED
        with pytest.raises(StateTransitionError):
            mgr.transition_to(ExperimentState.REPORT_COMPLETED)
