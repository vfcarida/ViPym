"""Logit Distillation & Teacher-Generated Synthetic Data pipeline.

.. deprecated::
    This module is a legacy stub.  For the full MoE-to-Dense distillation
    engine use ``vipym.distillation`` (P007).  The ``LogitDistillationMethod``
    class below is kept for backwards compatibility only.
"""
import warnings as _warnings

_warnings.warn(
    "vipym.compression.distillation.logit_distill is deprecated. "
    "Use vipym.distillation (DistillationMethod) instead.",
    DeprecationWarning,
    stacklevel=2,
)


from pathlib import Path
from typing import Any

import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


class LogitDistillationMethod(CompressionMethod):
    """KL-Divergence Logit-level Distillation (Requires vocabulary alignment).

    .. deprecated::
        Use ``vipym.distillation.DistillationMethod`` instead.
    """

    def __init__(
        self,
        student_model_id: str = "Qwen/Qwen2.5-Coder-7B",
        temperature: float = 2.0,
        alpha_ce: float = 0.5,
    ) -> None:
        self.student_model_id = student_model_id
        self.temperature = temperature
        self.alpha_ce = alpha_ce

    @property
    def name(self) -> str:
        return f"distill_logit_T{self.temperature}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
            },
            supported_dtypes={SupportedDtype.BF16, SupportedDtype.FP16},
            supports_moe=True,
            requires_training=True,
            requires_calibration=False,
            supported_runtimes={"vllm", "hf"},
        )

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        pass

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        out = Path(output_dir or "./logit_distilled_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Executing Logit Distillation (T={self.temperature}, alpha_ce={self.alpha_ce})"
        )

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out)

        total_bytes = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())

        return CompressionArtifact(
            output_path=out,
            format="safetensors",
            compressed_size_bytes=total_bytes,
            applied_methods=[self.name],
            metadata={
                "student_model_id": self.student_model_id,
                "temperature": self.temperature,
                "alpha_ce": self.alpha_ce,
            },
        )


class TeacherSyntheticDataPipeline:
    """Generates high-quality synthetic task solutions using frontier teacher LLM."""

    def __init__(self, teacher_backend: Any, prompt_templates: list[str]) -> None:
        self.teacher_backend = teacher_backend
        self.prompt_templates = prompt_templates

    def generate_dataset(self, num_samples: int = 1000) -> list[dict]:
        logger.info(f"Generating {num_samples} synthetic reasoning samples from teacher")
        return [
            {"instruction": f"Solve task {i}", "response": f"# Solution {i}"}
            for i in range(num_samples)
        ]


CompressionRegistry.register("distill_logit", LogitDistillationMethod)
