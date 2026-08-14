"""Custom exceptions for the ViPym framework."""


class ViPymError(Exception):
    """Base exception for all ViPym errors."""
    pass


class ConfigurationError(ViPymError):
    """Raised when an invalid configuration is provided."""
    pass


class ModelAdapterError(ViPymError):
    """Raised when a model adapter fails to load or inspect a model."""
    pass


class IncompatibleArchitectureError(ModelAdapterError):
    """Raised when a compression method or runtime is incompatible with the model architecture."""
    pass


class CompressionPipelineError(ViPymError):
    """Raised when a compression pipeline execution fails."""
    pass


class InvalidPipelineDAGError(CompressionPipelineError):
    """Raised when a compression pipeline DAG contains cycles or invalid dependencies."""
    pass


class InferenceRuntimeError(ViPymError):
    """Raised when the inference serving runtime fails."""
    pass


class EvaluationSandboxError(ViPymError):
    """Raised when an untrusted code sandbox fails to initialize or security constraints are violated."""
    pass


class BenchmarkEvaluationError(ViPymError):
    """Raised when benchmark execution fails."""
    pass


class CloudOrchestrationError(ViPymError):
    """Raised when AWS provisioning, S3 transfer, or remote orchestration fails."""
    pass


class BaselineMismatchError(ViPymError):
    """Raised when an experiment attempts to compare against an invalid or mismatched baseline."""
    pass
