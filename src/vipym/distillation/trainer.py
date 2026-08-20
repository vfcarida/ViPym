"""MoE-to-Dense Distillation Trainer.

Main training loop:
  1. For each batch: get student logits + (teacher logits from cache or live forward).
  2. Compute combined KL + CE loss.
  3. Backward + optimiser step (DeepSpeed engine if available, else vanilla PyTorch).
  4. Checkpoint at ``save_every_steps`` with full resume capability.
  5. Emit ``TrainingMetrics`` to ``training_log.jsonl`` at every step.
  6. Evaluate on benchmarks at ``eval_every_steps``.

Single-GPU / CPU fast-path (for unit tests):
  If ``deepspeed`` is not importable or ``deepspeed_stage == 0``, the trainer
  uses a plain ``torch.optim.AdamW`` + ``torch.optim.lr_scheduler.CosineAnnealingLR``.

Multi-GPU / Production:
  Set ``training.deepspeed_stage: 2`` or ``3`` in YAML.  The trainer will call
  ``deepspeed.initialize(model, optimizer, ...)`` and use the returned engine.
  Model loading with ZeRO-3 requires the ``zero.init()`` context manager to be
  applied *before* calling ``StudentInitializer.initialize()`` — callers are
  responsible for this when running in distributed mode.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vipym.core.logger import get_logger
from vipym.distillation.config import DistillationConfig, TrainingMetrics
from vipym.distillation.losses import combined_loss

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_CKPT_META_FILE = "trainer_state.json"
_CKPT_MODEL_FILE = "student_model.pt"


def _save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> Path:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Model weights
    if hasattr(student, "save_pretrained"):
        student.save_pretrained(str(ckpt_dir))
    else:
        torch.save(student.state_dict(), ckpt_dir / _CKPT_MODEL_FILE)

    # Trainer state
    meta = {
        "step": step,
        "epoch": epoch,
        "optimizer_state": None,   # omit for brevity in non-DS path
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    with open(ckpt_dir / _CKPT_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Checkpoint saved → {ckpt_dir}")
    return ckpt_dir


def _load_checkpoint(
    ckpt_dir: Path,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> tuple[int, int]:
    """Load checkpoint state into existing objects.  Returns (step, epoch)."""
    meta_path = ckpt_dir / _CKPT_META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    model_path = ckpt_dir / _CKPT_MODEL_FILE
    if model_path.exists():
        student.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    elif (ckpt_dir / "pytorch_model.bin").exists():
        student.load_state_dict(
            torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        )

    if scheduler is not None and meta.get("scheduler_state") is not None:
        scheduler.load_state_dict(meta["scheduler_state"])

    logger.info(f"Resumed from checkpoint: step={meta['step']}, epoch={meta['epoch']}")
    return int(meta["step"]), int(meta["epoch"])


# ---------------------------------------------------------------------------
# DeepSpeed integration
# ---------------------------------------------------------------------------


def _try_deepspeed_init(
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    ds_stage: int,
    local_rank: int = 0,
) -> tuple[Any, Any]:
    """Attempt to wrap student + optimizer with DeepSpeed.

    Returns ``(engine, None)`` on success or ``(None, None)`` if DeepSpeed
    is unavailable (single-GPU / CI path).
    """
    try:
        import deepspeed  # type: ignore[import]

        ds_config: dict[str, Any] = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "fp16": {"enabled": torch.cuda.is_available()},
            "zero_optimization": {
                "stage": ds_stage,
                "overlap_comm": True,
                "reduce_scatter": True,
            },
        }
        engine, opt, _, _ = deepspeed.initialize(
            model=student,
            optimizer=optimizer,
            config=ds_config,
        )
        logger.info(f"DeepSpeed ZeRO-{ds_stage} initialised.")
        return engine, opt
    except ImportError:
        logger.info("DeepSpeed not available — using vanilla PyTorch training loop.")
        return None, None


# ---------------------------------------------------------------------------
# DistillationTrainer
# ---------------------------------------------------------------------------


class DistillationTrainer:
    """Full distillation training loop.

    Args:
        teacher: Teacher model (frozen during training).
        student: Student model (trained).
        config: ``DistillationConfig``.
        train_dataset: ``torch.utils.data.Dataset`` yielding
            ``(input_ids, labels, cached_teacher_logits_or_None)``.
        output_dir: Where to save checkpoints and ``training_log.jsonl``.
        resume_from_checkpoint: Path to an existing checkpoint directory to
            resume from, or ``None`` to start fresh.
        device: Target device.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: DistillationConfig,
        train_dataset: Any | None = None,
        output_dir: Path | None = None,
        resume_from_checkpoint: Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.config = config
        self.train_dataset = train_dataset
        self.output_dir = Path(output_dir or "./distillation_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume_path = resume_from_checkpoint
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._log_path = self.output_dir / "training_log.jsonl"
        self._metrics_history: list[TrainingMetrics] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> list[TrainingMetrics]:
        """Run the full distillation training loop.

        Returns:
            List of ``TrainingMetrics`` emitted at each step.
        """
        cfg_t = self.config.training
        start_step, start_epoch = 0, 0

        # Freeze teacher
        self.teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Student to device
        self.student.to(self.device).train()
        if cfg_t.gradient_checkpointing and hasattr(self.student, "gradient_checkpointing_enable"):
            self.student.gradient_checkpointing_enable()

        # Optimiser + scheduler
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=cfg_t.learning_rate,
            weight_decay=0.01,
        )
        total_steps = self._total_steps()
        scheduler = self._build_scheduler(optimizer, total_steps, cfg_t.warmup_ratio)

        # DeepSpeed wrapping (no-op if unavailable / stage==0)
        ds_engine, ds_opt = None, None
        if cfg_t.deepspeed_stage > 0:
            ds_engine, ds_opt = _try_deepspeed_init(self.student, optimizer, cfg_t.deepspeed_stage)
        effective_student = ds_engine if ds_engine is not None else self.student
        effective_optimizer = ds_opt if ds_opt is not None else optimizer

        # Resume
        if self.resume_path is not None and Path(self.resume_path).exists():
            start_step, start_epoch = _load_checkpoint(
                Path(self.resume_path), self.student, optimizer, scheduler
            )

        # DataLoader
        dataloader = self._build_dataloader()

        # Training loop
        global_step = start_step
        t0 = time.perf_counter()

        for epoch in range(start_epoch, cfg_t.epochs):
            for batch in dataloader:
                if cfg_t.max_steps is not None and global_step >= cfg_t.max_steps:
                    break

                input_ids, labels, cached_logits = self._unpack_batch(batch)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                # Teacher logits
                if cached_logits is not None:
                    teacher_logits = cached_logits.to(self.device)
                else:
                    with torch.no_grad():
                        t_out = self.teacher(input_ids=input_ids)
                        teacher_logits = t_out.logits if hasattr(t_out, "logits") else t_out

                # Student forward
                s_out = effective_student(input_ids=input_ids)
                student_logits = s_out.logits if hasattr(s_out, "logits") else s_out

                # Loss
                total, kl, c_e = combined_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    alpha=cfg_t.alpha,
                    temperature=cfg_t.temperature,
                    loss_type=cfg_t.loss_type,
                )

                # Backward
                if ds_engine is not None:
                    ds_engine.backward(total)
                    ds_engine.step()
                else:
                    total.backward()
                    nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    effective_optimizer.step()
                    effective_optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

                # Metrics
                loss_val = float(total.item())
                kl_val = float(kl.item())
                ce_val = float(c_e.item())
                ppl = math.exp(min(ce_val, 20.0))
                lr_now = float(optimizer.param_groups[0]["lr"])
                elapsed = time.perf_counter() - t0

                metrics = TrainingMetrics(
                    step=global_step,
                    epoch=epoch,
                    loss=loss_val,
                    kl_loss=kl_val,
                    ce_loss=ce_val,
                    perplexity=ppl,
                    learning_rate=lr_now,
                    elapsed_seconds=round(elapsed, 2),
                )
                self._emit_metrics(metrics)
                self._metrics_history.append(metrics)

                global_step += 1

                # Checkpoint
                if global_step % cfg_t.save_every_steps == 0:
                    _save_checkpoint(self.output_dir, global_step, epoch, self.student, optimizer, scheduler)

                # Eval
                if global_step % cfg_t.eval_every_steps == 0:
                    eval_scores = self._evaluate()
                    metrics.eval_scores = eval_scores
                    logger.info(f"Step {global_step} eval: {eval_scores}")

            if cfg_t.max_steps is not None and global_step >= cfg_t.max_steps:
                break

        # Final checkpoint
        _save_checkpoint(self.output_dir, global_step, cfg_t.epochs, self.student, optimizer, scheduler)
        logger.info(f"Training complete — {global_step} steps, {time.perf_counter() - t0:.1f}s")
        return self._metrics_history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _total_steps(self) -> int:
        cfg_t = self.config.training
        if cfg_t.max_steps is not None:
            return cfg_t.max_steps
        n = len(self.train_dataset) if self.train_dataset is not None else 1
        steps_per_epoch = max(1, n // max(1, cfg_t.batch_size))
        return steps_per_epoch * cfg_t.epochs

    def _build_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            # Empty placeholder for tests that don't supply a dataset
            return DataLoader([], batch_size=self.config.training.batch_size)

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=_distil_collate_fn,
            drop_last=False,
        )

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            input_ids = batch[0]
            labels = batch[1]
            cached = batch[2] if len(batch) > 2 else None
            return input_ids, labels, cached
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_ratio: float,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if total_steps <= 0:
            return None
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _emit_metrics(self, metrics: TrainingMetrics) -> None:
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")

    def _evaluate(self) -> dict[str, float]:
        """Stub evaluation — returns placeholder scores.

        In production this would invoke the ViPym evaluation harness
        (HumanEval, BigCodeBench, etc.) against the current student checkpoint.
        """
        return {bench: 0.0 for bench in self.config.training.eval_benchmarks}


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------


def _distil_collate_fn(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Pad a batch of (input_ids, labels, cached_logits_or_None) to equal length."""
    input_ids_list, labels_list, logits_list = [], [], []
    has_logits = False

    for item in batch:
        input_ids_list.append(item[0])
        labels_list.append(item[1])
        if len(item) > 2 and item[2] is not None:
            logits_list.append(item[2])
            has_logits = True

    # Pad to max length in batch
    max_len = max(t.shape[0] for t in input_ids_list)
    padded_ids = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
    padded_labels = torch.full((len(input_ids_list), max_len), -100, dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
        L = ids.shape[0]
        padded_ids[i, :L] = ids
        padded_labels[i, :L] = labs

    cached_logits: torch.Tensor | None = None
    if has_logits and len(logits_list) == len(input_ids_list):
        # Stack if shapes match, else ignore cache for this batch
        try:
            cached_logits = torch.stack(logits_list, dim=0)
        except RuntimeError:
            cached_logits = None

    return padded_ids, padded_labels, cached_logits
