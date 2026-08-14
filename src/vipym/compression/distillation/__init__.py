"""Distillation adapters."""

from vipym.compression.distillation.logit_distill import (
    LogitDistillationMethod,
    TeacherSyntheticDataPipeline,
)
from vipym.compression.distillation.response_distill import ResponseDistillationMethod

__all__ = [
    "LogitDistillationMethod",
    "ResponseDistillationMethod",
    "TeacherSyntheticDataPipeline",
]
