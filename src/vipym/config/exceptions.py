"""ViPym typed exception hierarchy.

Re-exports core exceptions from vipym.core.exceptions to provide a single source of truth.
"""

from vipym.core.exceptions import (
    BaselineMismatchError,
    BenchmarkEvaluationError,
    CloudOrchestrationError,
    CompressionPipelineError,
    ConfigurationError,
    ContaminationError,
    EvaluationSandboxError,
    IncompatibleArchitectureError,
    InferenceRuntimeError,
    InvalidPipelineDAGError,
    ModelAdapterError,
    SandboxUnavailableError,
    StateTransitionError,
    ViPymError,
)

__all__ = [
    "BaselineMismatchError",
    "BenchmarkEvaluationError",
    "CloudOrchestrationError",
    "CompressionPipelineError",
    "ConfigurationError",
    "ContaminationError",
    "EvaluationSandboxError",
    "IncompatibleArchitectureError",
    "InferenceRuntimeError",
    "InvalidPipelineDAGError",
    "ModelAdapterError",
    "SandboxUnavailableError",
    "StateTransitionError",
    "ViPymError",
]
