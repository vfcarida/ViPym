"""Typed configuration dataclasses for MoE-to-Dense Knowledge Distillation (P007).

All config objects are plain dataclasses (not Pydantic) for lightweight testability.
The top-level ``DistillationConfig.from_dict()`` accepts the YAML stage ``parameters``
dict and constructs the full nested config tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Leaf configs
# ---------------------------------------------------------------------------


@dataclass
class StudentConfig:
    """Student model architecture and initialisation strategy.

    Args:
        architecture: HF model family name (``"llama"``, ``"qwen2"``,
            ``"mistral"``).  Used to select the right
            ``AutoModelForCausalLM`` config class.
        size: Target size string (``"7b"``, ``"14b"``, ``"32b"``, ``"70b"``).
            Informational only — the actual architecture is determined by
            ``init_from``.
        init_from: HF model-id for warm-start (e.g.
            ``"Qwen/Qwen2.5-32B"``).  Set to ``None`` for random
            initialisation or ``"teacher_subset"`` to copy top-K teacher layers.
        num_layers_from_teacher: Number of transformer layers to copy from the
            teacher when ``init_from == "teacher_subset"``.  0 means all.
    """

    architecture: str = "llama"
    size: str = "7b"
    init_from: str | None = None
    num_layers_from_teacher: int = 0


@dataclass
class TrainingConfig:
    """Optimiser and training-loop hyper-parameters.

    Args:
        epochs: Total training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation: Gradient accumulation steps (effective batch =
            ``batch_size * gradient_accumulation * num_gpus``).
        learning_rate: Peak learning rate.
        warmup_ratio: Fraction of total steps used for linear LR warm-up.
        temperature: Softmax temperature τ for teacher/student logits
            (higher = softer targets).
        alpha: Mixing weight.  ``loss = α·L_KL + (1-α)·L_CE``.
        deepspeed_stage: ZeRO optimisation stage (0 = disabled, 2 or 3).
        max_steps: Hard cap on training steps (``None`` = run all epochs).
        save_every_steps: Checkpoint interval in steps.
        eval_every_steps: Evaluation interval in steps.
        eval_benchmarks: List of benchmark IDs to evaluate at checkpoints.
        gradient_checkpointing: Enable gradient checkpointing to trade
            compute for memory.
        loss_type: Which distillation loss to use (``"forward_kl"``,
            ``"reverse_kl"``, ``"js"``).
    """

    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    temperature: float = 2.0
    alpha: float = 0.7
    deepspeed_stage: int = 3
    max_steps: int | None = None
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    eval_benchmarks: list[str] = field(default_factory=lambda: ["humaneval"])
    gradient_checkpointing: bool = True
    loss_type: str = "forward_kl"


@dataclass
class DataConfig:
    """Data generation and filtering configuration.

    Args:
        synthetic_samples: Number of teacher-generated synthetic samples.
        code_ratio: Fraction of training data that is code (vs. general text).
        execution_filter: If ``True``, run generated code through sandbox and
            keep only samples that pass.
        sandbox_timeout: Per-sample sandbox timeout in seconds.
        extra_corpora: Additional HF dataset IDs to mix in
            (e.g. ``"bigcode/the-stack-v2-train-smol-ids"``).
        cache_teacher_logits: Pre-compute and cache teacher logits to disk
            (``True``) vs. recompute on-the-fly (``False``).
        cache_dir: Directory for teacher logit shards.  Defaults to
            ``./teacher_logit_cache``.
        max_seq_len: Maximum token sequence length per sample.
    """

    synthetic_samples: int = 500_000
    code_ratio: float = 0.8
    execution_filter: bool = True
    sandbox_timeout: int = 30
    extra_corpora: list[str] = field(default_factory=list)
    cache_teacher_logits: bool = True
    cache_dir: str = "./teacher_logit_cache"
    max_seq_len: int = 2048


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass
class DistillationConfig:
    """Root configuration for the MoE-to-Dense distillation pipeline.

    Build via ``DistillationConfig.from_dict(yaml_parameters_dict)``.
    """

    teacher_model: str
    student: StudentConfig = field(default_factory=StudentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DistillationConfig":
        """Parse from a plain dict (e.g. the ``parameters`` block in a YAML stage).

        Ignores unknown top-level keys so forward-compat additions don't crash.
        """
        teacher_model = d.get("teacher_model", "")
        student = StudentConfig(**_filter_fields(StudentConfig, d.get("student", {})))
        training = TrainingConfig(**_filter_fields(TrainingConfig, d.get("training", {})))
        data = DataConfig(**_filter_fields(DataConfig, d.get("data", {})))
        return cls(teacher_model=teacher_model, student=student, training=training, data=data)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON/YAML export."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Training metrics (emitted to training_log.jsonl)
# ---------------------------------------------------------------------------


@dataclass
class TrainingMetrics:
    """Snapshot of training state at a given step."""

    step: int
    epoch: int
    loss: float
    kl_loss: float
    ce_loss: float
    perplexity: float
    learning_rate: float
    eval_scores: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_fields(dc_cls: type, d: dict[str, Any]) -> dict[str, Any]:
    """Return only the keys that exist as fields on *dc_cls*."""
    import dataclasses
    known = {f.name for f in dataclasses.fields(dc_cls)}
    return {k: v for k, v in d.items() if k in known}
