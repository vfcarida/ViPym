"""MoE-to-Dense Knowledge Distillation package.

Public API:

    from vipym.distillation import (
        DistillationConfig,
        StudentConfig,
        TrainingConfig,
        DataConfig,
        TrainingMetrics,
        DistillationTrainer,
        DistillationMethod,
        StudentInitializer,
        ProgressiveDistillationPipeline,
        ProgressiveStageSpec,
        SyntheticDataGenerator,
        ExecutionFilter,
        DistillationDataset,
        TeacherLogitCache,
        forward_kl_loss,
        reverse_kl_loss,
        js_divergence_loss,
        ce_loss,
        combined_loss,
    )

The ``DistillationMethod`` class is automatically registered in
``CompressionRegistry`` under ``"distillation"`` and ``"distill_moe_to_dense"``
upon first import of this package.
"""

from vipym.distillation.config import (
    DataConfig,
    DistillationConfig,
    StudentConfig,
    TrainingConfig,
    TrainingMetrics,
)
from vipym.distillation.data import (
    DistillationDataset,
    ExecutionFilter,
    SyntheticDataGenerator,
    TeacherLogitCache,
)
from vipym.distillation.losses import (
    ce_loss,
    combined_loss,
    forward_kl_loss,
    js_divergence_loss,
    reverse_kl_loss,
)
from vipym.distillation.method import DistillationMethod  # registers in CompressionRegistry
from vipym.distillation.progressive import ProgressiveDistillationPipeline, ProgressiveStageSpec
from vipym.distillation.student import StudentInitializer
from vipym.distillation.trainer import DistillationTrainer

__all__ = [
    # Config
    "DataConfig",
    "DistillationConfig",
    "StudentConfig",
    "TrainingConfig",
    "TrainingMetrics",
    # Data
    "DistillationDataset",
    "ExecutionFilter",
    "SyntheticDataGenerator",
    "TeacherLogitCache",
    # Losses
    "ce_loss",
    "combined_loss",
    "forward_kl_loss",
    "js_divergence_loss",
    "reverse_kl_loss",
    # Core
    "DistillationMethod",
    "DistillationTrainer",
    "StudentInitializer",
    # Progressive
    "ProgressiveDistillationPipeline",
    "ProgressiveStageSpec",
]
