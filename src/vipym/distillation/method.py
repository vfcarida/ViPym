"""DAG-compatible CompressionMethod wrapper for MoE-to-Dense distillation.

Registers as ``"distillation"`` and ``"distill_moe_to_dense"`` in
``CompressionRegistry`` so it can be chained in a DAG pipeline after
expert pruning/merging stages.

Example YAML stage:

  - id: distill_32b
    method: distillation
    dependencies: [merge_experts]
    parameters:
      teacher_model: moonshotai/kimi-k3
      student:
        architecture: qwen2
        size: 32b
        init_from: Qwen/Qwen2.5-32B
      training:
        epochs: 3
        max_steps: 10000
        temperature: 2.0
        alpha: 0.7
        deepspeed_stage: 3
      data:
        synthetic_samples: 500000
        code_ratio: 0.8
        execution_filter: true
        sandbox_timeout: 30
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.distillation.config import DistillationConfig
from vipym.distillation.data import DistillationDataset, ExecutionFilter, SyntheticDataGenerator
from vipym.distillation.student import StudentInitializer
from vipym.distillation.trainer import DistillationTrainer
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


class DistillationMethod(CompressionMethod):
    """MoE-to-Dense knowledge distillation as a ``CompressionMethod``.

    In the DAG pipeline ``compress()`` receives the **teacher** model
    (the previous stage's output).  It constructs the student, runs the
    training loop, and returns a ``CompressionArtifact`` pointing at the
    distilled student checkpoint.
    """

    def __init__(
        self,
        teacher_model: str = "",
        student_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        data_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._teacher_model = teacher_model
        self._student_cfg = student_config or {}
        self._training_cfg = training_config or {}
        self._data_cfg = data_config or {}
        self._extra = kwargs

    @property
    def name(self) -> str:
        size = self._student_cfg.get("size", "unknown")
        return f"distill_moe_to_dense_{size}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.MOE,
                ComputeArchitecture.DENSE,
                ComputeArchitecture.HYBRID_ATTENTION,
            },
            supported_dtypes={SupportedDtype.BF16, SupportedDtype.FP16, SupportedDtype.FP32},
            supports_moe=True,
            requires_training=True,
            requires_calibration=False,
            supported_runtimes={"vllm", "sglang", "hf"},
        )

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        # Distillation is universally applicable to any teacher architecture
        pass

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        """Run distillation.  ``model`` is used as the teacher."""
        out = Path(output_dir or "./distillation_output")
        out.mkdir(parents=True, exist_ok=True)

        # Merge constructor defaults with runtime kwargs
        combined_params: dict[str, Any] = {
            "teacher_model": kwargs.get("teacher_model", self._teacher_model),
            "student": kwargs.get("student", self._student_cfg),
            "training": kwargs.get("training", self._training_cfg),
            "data": kwargs.get("data", self._data_cfg),
        }
        cfg = DistillationConfig.from_dict(combined_params)

        # Override max_steps for fast-path if explicitly provided
        if "max_steps" in kwargs:
            cfg.training.max_steps = int(kwargs["max_steps"])

        logger.info(
            f"DistillationMethod: teacher={cfg.teacher_model or model.__class__.__name__} "
            f"→ student size={cfg.student.size}, "
            f"T={cfg.training.temperature}, α={cfg.training.alpha}"
        )

        # Initialise student
        vocab_size = getattr(tokenizer, "vocab_size", 32000) if tokenizer is not None else 32000
        student = StudentInitializer(cfg.student, vocab_size=vocab_size).initialize(teacher=model)

        # Build dataset
        train_dataset: DistillationDataset | None = None
        if cfg.data.synthetic_samples > 0:
            gen = SyntheticDataGenerator(
                teacher=model,
                tokenizer=tokenizer,
                num_samples=cfg.data.synthetic_samples,
                code_ratio=cfg.data.code_ratio,
            )
            samples = gen.generate()

            if cfg.data.execution_filter:
                flt = ExecutionFilter(timeout=cfg.data.sandbox_timeout)
                samples = flt.filter(samples)

            train_dataset = DistillationDataset(
                samples=samples,
                tokenizer=tokenizer,
                max_seq_len=cfg.data.max_seq_len,
            )

        # Train
        trainer = DistillationTrainer(
            teacher=model,
            student=student,
            config=cfg,
            train_dataset=train_dataset,
            output_dir=out,
        )
        metrics_history = trainer.train()

        # Save student
        if hasattr(student, "save_pretrained"):
            student.save_pretrained(str(out))

        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(out))

        total_bytes = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())
        final_loss = metrics_history[-1].loss if metrics_history else 0.0

        logger.info(f"Distillation complete — final loss: {final_loss:.4f}")

        return CompressionArtifact(
            output_path=out,
            format="safetensors",
            compressed_size_bytes=total_bytes,
            applied_methods=[self.name],
            metadata={
                "teacher_model": cfg.teacher_model,
                "student_size": cfg.student.size,
                "student_architecture": cfg.student.architecture,
                "init_from": cfg.student.init_from,
                "epochs": cfg.training.epochs,
                "temperature": cfg.training.temperature,
                "alpha": cfg.training.alpha,
                "steps_trained": len(metrics_history),
                "final_loss": final_loss,
                "deepspeed_stage": cfg.training.deepspeed_stage,
            },
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CompressionRegistry.register("distillation", DistillationMethod)
CompressionRegistry.register("distill_moe_to_dense", DistillationMethod)
