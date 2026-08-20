"""ViPym typed exception hierarchy."""


class ViPymError(Exception):
    """Base exception for all ViPym errors."""

    pass


class ConfigurationError(ViPymError):
    """Raised when an invalid or unparseable configuration is provided."""

    pass


class ModelAdapterError(ViPymError):
    """Raised when a model adapter fails to inspect, load, or tokenize a model."""

    pass


class IncompatibleArchitectureError(ModelAdapterError):
    """Raised when a compression method or runtime is incompatible with the model architecture."""

    pass


class CompressionPipelineError(ViPymError):
    """Raised when compression pipeline execution fails."""

    pass


class InvalidPipelineDAGError(CompressionPipelineError):
    """Raised when a compression pipeline DAG contains cycles or invalid stage dependencies."""

    pass


class InferenceRuntimeError(ViPymError):
    """Raised when the inference serving runtime fails to start or generate outputs."""

    pass


class EvaluationSandboxError(ViPymError):
    """Raised when untrusted code execution sandbox violates security constraints or fails."""

    pass


class SandboxUnavailableError(EvaluationSandboxError):
    """Raised when Docker / container sandbox is unavailable and unsafe execution is disallowed."""

    pass


class BenchmarkEvaluationError(ViPymError):
    """Raised when benchmark execution fails."""

    pass


class ContaminationError(BenchmarkEvaluationError):
    """Raised when high contamination is detected between training data and evaluation suite."""

    pass


class CloudOrchestrationError(ViPymError):
    """Raised when AWS provisioning, S3 transfer, or remote orchestration fails."""

    pass


class BaselineMismatchError(ViPymError):
    """Raised when an experiment attempts to compare against an invalid or mismatched baseline."""

    pass


class StateTransitionError(ViPymError):
    """Raised when an illegal state transition occurs in the experiment lifecycle."""

    pass
