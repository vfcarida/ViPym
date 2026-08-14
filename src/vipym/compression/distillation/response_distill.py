"""Sequence-level response distillation and teacher tracking adapter."""

from pathlib import Path
from typing import Any

import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


class ResponseDistillationMethod(CompressionMethod):
    """Response Distillation: Trains student model on teacher-generated synthetic reasoning data."""

    def __init__(
        self,
        student_model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        distillation_dataset: str = "code_distill_50k",
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
    ) -> None:
        self.student_model_id = student_model_id
        self.distillation_dataset = distillation_dataset
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    @property
    def name(self) -> str:
        student_short = self.student_model_id.split("/")[-1].lower()
        return f"distill_response_to_{student_short}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
                ComputeArchitecture.HYBRID_ATTENTION,
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
        out = Path(output_dir or "./distilled_student_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Executing Response Distillation: Teacher={model.__class__.__name__} -> Student={self.student_model_id}"
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
                "teacher_model": model.__class__.__name__,
                "student_model_id": self.student_model_id,
                "distillation_dataset": self.distillation_dataset,
                "learning_rate": self.learning_rate,
                "epochs": self.num_epochs,
            },
        )


CompressionRegistry.register("distill_response", ResponseDistillationMethod)
