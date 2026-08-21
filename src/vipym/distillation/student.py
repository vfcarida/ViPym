"""Student model initialisation for MoE-to-Dense distillation.

Three initialisation strategies:

``"random"``
    Create a fresh ``AutoModelForCausalLM`` from an HF config.  The model
    architecture is specified by ``StudentConfig.architecture`` and the
    hidden size / layer count is approximated from ``StudentConfig.size``.

``"pretrained"``
    Warm-start from a smaller pre-trained model via
    ``AutoModelForCausalLM.from_pretrained(StudentConfig.init_from)``.
    This is the recommended production path.

``"teacher_subset"``
    Copy the first ``N`` transformer layers from the teacher into a fresh
    student architecture.  For MoE→Dense, shared-attention layers are copied
    directly; FFN layers in the student are initialised from the *first expert*
    weight block of the corresponding teacher MoE block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vipym.core.logger import get_logger
from vipym.distillation.config import StudentConfig

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dense architecture size estimates
# ---------------------------------------------------------------------------
# Maps size-string → approximate (hidden_dim, num_layers, num_heads)
# These are only used for *random* init when no pretrained model is specified.
_SIZE_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
    },
    "small": {
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
    },
    "7b": {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "intermediate_size": 14336,
    },
    "14b": {
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "intermediate_size": 13824,
    },
    "32b": {
        "hidden_size": 6656,
        "num_hidden_layers": 60,
        "num_attention_heads": 52,
        "intermediate_size": 17920,
    },
    "70b": {
        "hidden_size": 8192,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "intermediate_size": 28672,
    },
}


# ---------------------------------------------------------------------------
# StudentInitializer
# ---------------------------------------------------------------------------


class StudentInitializer:
    """Initialise a dense student model for distillation.

    Args:
        config: ``StudentConfig`` controlling architecture and init strategy.
        vocab_size: Vocabulary size (must match teacher tokenizer).
    """

    def __init__(self, config: StudentConfig, vocab_size: int = 32000) -> None:
        self.config = config
        self.vocab_size = vocab_size

    def initialize(self, teacher: nn.Module | None = None) -> nn.Module:
        """Create and return the student ``nn.Module``.

        Args:
            teacher: Required when ``config.init_from == "teacher_subset"``.

        Returns:
            Initialised student model in training mode.
        """
        init_from = (self.config.init_from or "").strip().lower()

        if init_from == "" or init_from == "random":
            logger.info(f"StudentInitializer: random init (size={self.config.size})")
            return self._random_init()

        if init_from == "teacher_subset":
            if teacher is None:
                raise ValueError("'teacher_subset' init requires passing teacher model.")
            logger.info(
                f"StudentInitializer: teacher_subset init from teacher ({type(teacher).__name__})"
            )
            return self._teacher_subset_init(teacher)

        # Pretrained warm-start
        logger.info(f"StudentInitializer: pretrained init from '{self.config.init_from}'")
        return self._pretrained_init()

    # ------------------------------------------------------------------
    # Random init
    # ------------------------------------------------------------------

    def _random_init(self) -> nn.Module:
        """Try ``transformers.AutoModelForCausalLM`` with a generated config.

        Falls back to a tiny ``_SimpleDenseModel`` if transformers is not installed
        (e.g. in minimal CI environments).
        """
        size_key = self.config.size.lower()
        preset = _SIZE_PRESETS.get(size_key, _SIZE_PRESETS["tiny"])

        try:
            from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import]

            hf_cfg = AutoConfig.for_model(
                self.config.architecture,
                vocab_size=self.vocab_size,
                **preset,
            )
            model = AutoModelForCausalLM.from_config(hf_cfg)
            logger.info(f"Random student: {sum(p.numel() for p in model.parameters()):,} params")
            return model.train()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"transformers not available ({exc}); using _SimpleDenseModel fallback.")
            return _SimpleDenseModel(
                vocab_size=self.vocab_size,
                hidden_size=preset["hidden_size"],
                num_layers=preset["num_hidden_layers"],
            )

    # ------------------------------------------------------------------
    # Pretrained warm-start
    # ------------------------------------------------------------------

    def _pretrained_init(self) -> nn.Module:
        try:
            from transformers import AutoModelForCausalLM  # type: ignore[import]

            model = AutoModelForCausalLM.from_pretrained(
                self.config.init_from,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            logger.info(
                f"Pretrained student loaded: {sum(p.numel() for p in model.parameters()):,} params"
            )
            return model.train()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Could not load pretrained '{self.config.init_from}': {exc}. Falling back to random."
            )
            return self._random_init()

    # ------------------------------------------------------------------
    # Teacher-subset init (MoE → Dense)
    # ------------------------------------------------------------------

    def _teacher_subset_init(self, teacher: nn.Module) -> nn.Module:
        """Copy teacher layers into a fresh dense student skeleton.

        For MoE blocks: the student's FFN is initialised from the *first*
        expert in the teacher's MoE block (the most-used expert by default).
        """
        # Build skeleton
        student = self._random_init()

        n_copy = self.config.num_layers_from_teacher
        teacher_layers = _get_transformer_layers(teacher)
        student_layers = _get_transformer_layers(student)

        if not teacher_layers or not student_layers:
            logger.warning(
                "Could not locate transformer layers for teacher_subset init; returning random student."
            )
            return student

        if n_copy <= 0:
            n_copy = min(len(teacher_layers), len(student_layers))

        logger.info(
            f"Copying {n_copy} layers from teacher ({len(teacher_layers)} available) into student ({len(student_layers)} layers)"
        )

        for s_idx in range(min(n_copy, len(student_layers))):
            t_idx = int(s_idx * len(teacher_layers) / n_copy)
            t_layer = teacher_layers[t_idx]
            s_layer = student_layers[s_idx]
            _copy_layer(t_layer, s_layer)

        return student


# ---------------------------------------------------------------------------
# Helpers for layer discovery and copying
# ---------------------------------------------------------------------------


def _get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Heuristically locate the list of transformer decoder layers."""
    # Common attribute names used by HF models
    for attr in ("layers", "model.layers", "transformer.h", "model.decoder.layers"):
        parts = attr.split(".")
        obj: Any = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue
    return []


def _copy_layer(src: nn.Module, dst: nn.Module) -> None:
    """Best-effort copy of compatible parameters from src → dst.

    For MoE src layers: first expert's FFN weights are used for the dense dst FFN.
    Incompatible shape pairs are silently skipped.
    """
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    updated: dict[str, torch.Tensor] = {}

    for key, dst_param in dst_sd.items():
        # Direct match
        if key in src_sd and src_sd[key].shape == dst_param.shape:
            updated[key] = src_sd[key].clone()
            continue

        # MoE → Dense: try key with expert prefix  (e.g. "mlp.experts.0.*")
        for expert_prefix in ("experts.0.", "mlp.experts.0.", "block_sparse_moe.experts.0."):
            candidate_key = expert_prefix + key.replace("mlp.", "")
            if candidate_key in src_sd and src_sd[candidate_key].shape == dst_param.shape:
                updated[key] = src_sd[candidate_key].clone()
                break

    if updated:
        dst_sd.update(updated)
        dst.load_state_dict(dst_sd, strict=False)


# ---------------------------------------------------------------------------
# Fallback minimal dense model for testing without transformers installed
# ---------------------------------------------------------------------------


class _SimpleDenseModel(nn.Module):
    """Tiny GPT-style causal LM for unit testing without HF transformers."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=max(1, hidden_size // 64),
                    dim_feedforward=hidden_size * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        # Clamp to valid vocab range (protects against character-level tokeniser producing ids > vocab_size)
        input_ids = input_ids.clamp(0, self.vocab_size - 1)
        x = self.embed(input_ids)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        memory = torch.zeros_like(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask)
        x = self.norm(x)
        logits = self.lm_head(x)  # [B, L, V]

        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1).clamp(min=0),
            )
            # Return HF-style namespace
            return _ModelOutput(loss=loss, logits=logits)
        return _ModelOutput(loss=None, logits=logits)

    def save_pretrained(self, path: str | Path) -> None:
        from pathlib import Path as _Path

        import torch

        _Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), _Path(path) / "pytorch_model.bin")

    def gradient_checkpointing_enable(self) -> None:
        pass  # No-op for simple model


class _ModelOutput:
    """Minimal stand-in for HF ModelOutput."""

    __slots__ = ("loss", "logits")

    def __init__(self, loss: torch.Tensor | None, logits: torch.Tensor) -> None:
        self.loss = loss
        self.logits = logits
