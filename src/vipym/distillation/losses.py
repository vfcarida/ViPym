"""Pure-PyTorch loss functions for knowledge distillation.

All functions operate on raw logit tensors and are framework-agnostic
(no HuggingFace, no DeepSpeed imports).  Vocabulary-size mismatch between
teacher and student is handled transparently via truncation + re-normalisation.

Loss taxonomy
─────────────
  forward_kl  : KL(teacher ‖ student)  — standard distillation; student fits teacher modes.
  reverse_kl  : KL(student ‖ teacher)  — mode-seeking; sharpens student distribution.
  js           : JS(teacher, student)  — symmetric; compromise between forward/reverse KL.
  combined     : α · L_distil + (1-α) · L_CE  — blends distillation with ground-truth CE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vocabulary alignment
# ---------------------------------------------------------------------------


def align_vocab(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate the larger vocabulary to the smaller one and re-normalise teacher probs.

    Args:
        student_logits: ``[B, L, V_s]`` or ``[B, V_s]``.
        teacher_logits: ``[B, L, V_t]`` or ``[B, V_t]``.

    Returns:
        ``(student_logits, teacher_logits)`` with matching last dimension.
    """
    v_s = student_logits.shape[-1]
    v_t = teacher_logits.shape[-1]
    if v_s == v_t:
        return student_logits, teacher_logits
    v_min = min(v_s, v_t)
    return student_logits[..., :v_min], teacher_logits[..., :v_min]


# ---------------------------------------------------------------------------
# Core distillation losses
# ---------------------------------------------------------------------------


def forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """KL(softmax(teacher/τ) ‖ log_softmax(student/τ)) scaled by τ².

    Standard Hinton et al. distillation loss.  The τ² factor compensates for
    the gradient magnitude reduction introduced by temperature scaling.

    Args:
        student_logits: Float tensor ``[B, L, V]`` or ``[B*L, V]``.
        teacher_logits: Float tensor matching ``student_logits`` shape.
        temperature: Softmax temperature τ (> 1 softens targets).
        reduction: ``"batchmean"`` (default) or ``"sum"`` / ``"none"``.

    Returns:
        Scalar loss (or tensor if ``reduction="none"``).
    """
    student_logits, teacher_logits = align_vocab(student_logits, teacher_logits)

    # Flatten to 2-D if 3-D
    orig_shape = student_logits.shape
    if student_logits.dim() == 3:
        b, l, v = orig_shape
        student_logits = student_logits.reshape(b * l, v)
        teacher_logits = teacher_logits.reshape(b * l, v)

    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)

    loss = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * loss


def reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """KL(softmax(student/τ) ‖ log_softmax(teacher/τ)) scaled by τ².

    Mode-seeking: penalises student for placing mass where teacher has none.
    Useful for sharpening distributions.
    """
    student_logits, teacher_logits = align_vocab(student_logits, teacher_logits)

    if student_logits.dim() == 3:
        b, l, v = student_logits.shape
        student_logits = student_logits.reshape(b * l, v)
        teacher_logits = teacher_logits.reshape(b * l, v)

    student_probs = F.softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)

    loss = F.kl_div(teacher_log_probs, student_probs, reduction=reduction)
    return (temperature ** 2) * loss


def js_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Jensen-Shannon divergence: 0.5 · KL(t‖m) + 0.5 · KL(s‖m)  where m = (t+s)/2."""
    student_logits, teacher_logits = align_vocab(student_logits, teacher_logits)

    if student_logits.dim() == 3:
        b, l, v = student_logits.shape
        student_logits = student_logits.reshape(b * l, v)
        teacher_logits = teacher_logits.reshape(b * l, v)

    t_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    s_probs = F.softmax(student_logits.float() / temperature, dim=-1)
    m_probs = 0.5 * (t_probs + s_probs)
    m_log = m_probs.log().clamp(min=-100.0)

    kl_tm = F.kl_div(m_log, t_probs, reduction=reduction)
    kl_sm = F.kl_div(m_log, s_probs, reduction=reduction)
    return (temperature ** 2) * 0.5 * (kl_tm + kl_sm)


def ce_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard cross-entropy on ground-truth labels.

    Args:
        student_logits: ``[B, L, V]`` or ``[B*L, V]``.
        labels: Long tensor ``[B, L]`` or ``[B*L]`` with ``ignore_index`` for padding.
        ignore_index: Token ID to ignore in loss (default -100 = HF convention).

    Returns:
        Scalar loss.
    """
    if student_logits.dim() == 3:
        b, l, v = student_logits.shape
        student_logits = student_logits.reshape(b * l, v)
        labels = labels.reshape(b * l)

    # Clamp valid (non-ignored) label indices to the model's vocab size.
    # This protects against character-level fallback tokenisers that may
    # produce ids up to 255 while the test model only has vocab_size=64.
    num_classes = student_logits.shape[-1]
    valid_mask = labels != ignore_index
    labels = labels.clone()
    labels[valid_mask] = labels[valid_mask].clamp(0, num_classes - 1)

    return F.cross_entropy(
        student_logits.float(),
        labels.long(),
        ignore_index=ignore_index,
    )


def combined_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.7,
    temperature: float = 2.0,
    loss_type: str = "forward_kl",
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """α · L_distil + (1-α) · L_CE

    Args:
        student_logits: ``[B, L, V_s]``.
        teacher_logits: ``[B, L, V_t]``.
        labels: Long tensor ``[B, L]`` with ``ignore_index`` padding.
        alpha: Weight on the distillation loss.
        temperature: Softmax temperature τ.
        loss_type: ``"forward_kl"``, ``"reverse_kl"``, or ``"js"``.
        ignore_index: Padding token for CE loss.

    Returns:
        ``(total_loss, kl_loss, ce_loss_val)`` — all scalars.
    """
    _LOSS_FNS = {
        "forward_kl": forward_kl_loss,
        "reverse_kl": reverse_kl_loss,
        "js": js_divergence_loss,
    }
    if loss_type not in _LOSS_FNS:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Choose from {list(_LOSS_FNS)}")

    distil_fn = _LOSS_FNS[loss_type]
    kl = distil_fn(student_logits, teacher_logits, temperature=temperature)
    c_e = ce_loss(student_logits, labels, ignore_index=ignore_index)
    total = alpha * kl + (1.0 - alpha) * c_e
    return total, kl, c_e
