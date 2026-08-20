"""Progressive multi-stage distillation pipeline.

Implements chained distillation:

  Stage 1: Teacher (2.8T MoE)  ──► Student 1 (256B MoE, fewer experts)
  Stage 2: Student 1            ──► Student 2 (32B Dense)
  Stage 3 (optional):
           Student 2            ──► Student 3 (14B Dense)

Each stage is a ``DistillationTrainer`` run.  The previous stage's final
checkpoint becomes the *teacher* for the next stage.

Example YAML:
  progressive_stages:
    - teacher_model: moonshotai/kimi-k3
      student:
        architecture: qwen2
        size: 32b
        init_from: Qwen/Qwen2.5-32B
      training:
        epochs: 1
        max_steps: 5000
        temperature: 3.0
    - teacher_model: ./stage0_out   # auto-filled
      student:
        architecture: qwen2
        size: 14b
        init_from: Qwen/Qwen2.5-14B
      training:
        epochs: 2
        temperature: 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn

from vipym.core.logger import get_logger
from vipym.distillation.config import DistillationConfig, _filter_fields
from vipym.distillation.data import DistillationDataset, ExecutionFilter, SyntheticDataGenerator
from vipym.distillation.student import StudentInitializer
from vipym.distillation.trainer import DistillationTrainer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-stage spec
# ---------------------------------------------------------------------------


@dataclass
class ProgressiveStageSpec:
    """Configuration for a single progressive distillation stage."""

    teacher_model: str
    student: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProgressiveStageSpec":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ProgressiveDistillationPipeline:
    """Orchestrate multi-stage MoE → Dense distillation.

    Args:
        stages: List of ``ProgressiveStageSpec`` dicts (or objects).
        base_output_dir: Root directory; each stage writes to
            ``base_output_dir/stage_{i}/``.
        load_model_fn: Callable ``(model_id: str) -> nn.Module`` to load a
            teacher model.  If ``None``, a stub is used (for tests).
        load_tokenizer_fn: Callable ``(model_id: str) -> Any`` for tokenizer.
    """

    def __init__(
        self,
        stages: list[dict[str, Any]],
        base_output_dir: str | Path = "./progressive_distillation",
        load_model_fn: Any = None,
        load_tokenizer_fn: Any = None,
    ) -> None:
        self.stage_specs = [
            s if isinstance(s, ProgressiveStageSpec) else ProgressiveStageSpec.from_dict(s)
            for s in stages
        ]
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self._load_model = load_model_fn
        self._load_tokenizer = load_tokenizer_fn

    def run(self) -> list[Path]:
        """Execute all stages sequentially.

        Returns:
            List of output directory paths, one per stage.
        """
        stage_outputs: list[Path] = []
        prev_output: Path | None = None

        for stage_idx, spec in enumerate(self.stage_specs):
            stage_out = self.base_output_dir / f"stage_{stage_idx}"
            stage_out.mkdir(parents=True, exist_ok=True)

            # If this is not the first stage, the teacher is the previous output
            teacher_id = spec.teacher_model if prev_output is None else str(prev_output)

            logger.info(f"[Progressive] Stage {stage_idx}: teacher={teacher_id} → {stage_out}")

            teacher = self._load_teacher(teacher_id)
            tokenizer = self._load_tok(teacher_id)

            # Build DistillationConfig for this stage
            cfg_dict = {
                "teacher_model": teacher_id,
                "student": spec.student,
                "training": spec.training,
                "data": spec.data,
            }
            cfg = DistillationConfig.from_dict(cfg_dict)

            # Build student
            vocab_size = getattr(tokenizer, "vocab_size", 32000) if tokenizer is not None else 32000
            student = StudentInitializer(cfg.student, vocab_size=vocab_size).initialize(teacher=teacher)

            # Build dataset
            dataset = self._build_dataset(teacher, tokenizer, cfg)

            # Train
            trainer = DistillationTrainer(
                teacher=teacher,
                student=student,
                config=cfg,
                train_dataset=dataset,
                output_dir=stage_out,
            )
            metrics = trainer.train()

            n_steps = len(metrics)
            final_loss = metrics[-1].loss if metrics else float("nan")
            logger.info(f"[Progressive] Stage {stage_idx} done — {n_steps} steps, final loss={final_loss:.4f}")

            prev_output = stage_out
            stage_outputs.append(stage_out)

        return stage_outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_teacher(self, model_id: str) -> nn.Module:
        if self._load_model is not None:
            return self._load_model(model_id)
        # Fallback stub for tests
        from vipym.distillation.student import _SimpleDenseModel
        return _SimpleDenseModel(vocab_size=256, hidden_size=64, num_layers=2)

    def _load_tok(self, model_id: str) -> Any:
        if self._load_tokenizer is not None:
            return self._load_tokenizer(model_id)
        return None  # Data module handles None tokenizer gracefully

    def _build_dataset(
        self,
        teacher: nn.Module,
        tokenizer: Any,
        cfg: DistillationConfig,
    ) -> DistillationDataset | None:
        n = cfg.data.synthetic_samples
        if n <= 0:
            return None

        gen = SyntheticDataGenerator(
            teacher=teacher,
            tokenizer=tokenizer,
            num_samples=n,
            code_ratio=cfg.data.code_ratio,
        )
        samples = gen.generate()

        if cfg.data.execution_filter:
            flt = ExecutionFilter(timeout=cfg.data.sandbox_timeout)
            samples = flt.filter(samples)

        return DistillationDataset(
            samples=samples,
            tokenizer=tokenizer,
            max_seq_len=cfg.data.max_seq_len,
        )
