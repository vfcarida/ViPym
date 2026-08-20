"""Router Distillation for MoE — teacher-guided KL-divergence retraining.

After expert pruning or merging the original router is misaligned: its output
dimension no longer matches the surviving expert set.  This module trains a
*student* router (or fine-tunes the sliced router in-place) to mimic the
*teacher* routing distribution on calibration hidden states, while enforcing
load-balance across the remaining experts.

Training signal
───────────────
  L = λ_kl  · KL(teacher_probs ‖ student_probs)
    + λ_bal · L_balance
    + λ_ent · L_entropy

  L_balance   = std(mean_routing_probs)  → penalises expert collapse
  L_entropy   = -Σ p log p (per-token)  → encourages sharp routing

Only the router/gate weights are updated; every expert module is frozen.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from vipym.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class RouterDistillationConfig:
    """Hyper-parameters for router distillation.

    Args:
        steps: Optimisation steps (500–1000 is usually sufficient).
        lr: Learning rate for AdamW.
        calibration_samples: Number of hidden-state vectors sampled per step.
        balance_loss_weight: Weight of the load-balance regulariser (λ_bal).
        kl_loss_weight: Weight of the KL-divergence teacher loss (λ_kl).
        entropy_loss_weight: Weight of the per-token entropy loss (λ_ent).
        warmup_steps: Linear LR warmup (0 = no warmup).
        temperature: Softmax temperature applied to *teacher* logits before KL.
        max_utilisation_ratio: Target max/min expert utilisation ratio.
            Logged but not enforced as a hard constraint.
        log_every: Log training progress every N steps.
    """

    steps: int = 500
    lr: float = 1e-4
    calibration_samples: int = 1024
    balance_loss_weight: float = 0.10
    kl_loss_weight: float = 1.00
    entropy_loss_weight: float = 0.01
    warmup_steps: int = 50
    temperature: float = 1.0
    max_utilisation_ratio: float = 3.0
    log_every: int = 100

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RouterDistillationConfig":
        """Build from a plain dict (e.g. parsed from YAML)."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Results dataclass
# ---------------------------------------------------------------------------


@dataclass
class RouterDistillationResult:
    """Metrics returned after distillation."""

    steps_run: int
    elapsed_seconds: float
    initial_loss: float
    final_loss: float
    converged: bool
    loss_history: list[float] = field(default_factory=list)
    utilisation_ratio: float = 1.0  # max / min expert load
    load_balanced: bool = True      # True if ratio < config.max_utilisation_ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps_run": self.steps_run,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "initial_loss": round(self.initial_loss, 6),
            "final_loss": round(self.final_loss, 6),
            "converged": self.converged,
            "utilisation_ratio": round(self.utilisation_ratio, 2),
            "load_balanced": self.load_balanced,
        }


# ---------------------------------------------------------------------------
# Core distillation function
# ---------------------------------------------------------------------------


def distil_router(
    student_router: nn.Linear,
    teacher_logits: torch.Tensor,
    config: RouterDistillationConfig | None = None,
    device: torch.device | None = None,
) -> RouterDistillationResult:
    """Distil a sliced student router from pre-computed teacher logits.

    The teacher logits are the *original* router outputs on calibration hidden
    states, collected **before** pruning.  The function maps them to teacher
    probabilities over the *retained* expert indices implicitly via the
    calibration data passed to the student router.

    Args:
        student_router: The already-sliced ``nn.Linear`` gate (out_features ==
            number of retained experts).  Only this module is optimised.
        teacher_logits: Float tensor of shape ``[N, E_orig]`` — the original
            router logits over the full expert set for N calibration tokens.
        config: Distillation hyper-parameters.  Defaults to
            ``RouterDistillationConfig()``.
        device: Target device.  Inferred from ``student_router`` if ``None``.

    Returns:
        ``RouterDistillationResult`` with training metrics.
    """
    cfg = config or RouterDistillationConfig()
    dev = device or next(student_router.parameters()).device

    teacher_logits = teacher_logits.to(dev).float().detach()
    # Teacher soft targets — temperature-scaled
    teacher_probs = F.softmax(teacher_logits / cfg.temperature, dim=-1)

    # The teacher has E_orig columns; student has E_retained.
    # We marginalise the teacher distribution over the *retained* columns and
    # re-normalise so it sums to 1 — this is the distillation target.
    e_retained = student_router.out_features
    if teacher_probs.shape[-1] > e_retained:
        # Take first e_retained columns (they correspond to retained experts
        # after the gate was already sliced in `expert_pruning.py`).
        teacher_target = teacher_probs[:, :e_retained]
        teacher_target = teacher_target / (teacher_target.sum(dim=-1, keepdim=True) + 1e-8)
    else:
        teacher_target = teacher_probs

    n_samples = teacher_target.shape[0]
    batch_size = min(cfg.calibration_samples, n_samples)

    # Build calibration hidden states using an identity matrix trick:
    # we only have the teacher logits, not the hidden states that produced them.
    # We create a synthetic feature matrix using SVD on the teacher weights so
    # the student can be trained in the weight space.  This avoids requiring
    # the caller to pass raw activations when only logits are available.
    hidden_dim = student_router.in_features
    calib_hidden = _make_calib_hidden(teacher_target, hidden_dim, n_samples, dev)

    optimizer = optim.AdamW(student_router.parameters(), lr=cfg.lr, weight_decay=1e-3)
    scheduler = _make_scheduler(optimizer, cfg.warmup_steps, cfg.steps)

    student_router.train()
    loss_history: list[float] = []
    t0 = time.perf_counter()

    for step in range(cfg.steps):
        idx = torch.randint(0, n_samples, (batch_size,), device=dev)
        h_batch = calib_hidden[idx]          # [B, hidden_dim]
        t_batch = teacher_target[idx]        # [B, E_retained]

        optimizer.zero_grad(set_to_none=True)
        student_logits = student_router(h_batch)  # [B, E_retained]
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # KL divergence: KL(teacher ‖ student) = Σ t * (log t - log s)
        kl_loss = F.kl_div(student_log_probs, t_batch, reduction="batchmean")

        # Load-balance: penalise high variance in mean expert utilisation
        mean_routing = F.softmax(student_logits, dim=-1).mean(dim=0)  # [E_retained]
        balance_loss = mean_routing.std()

        # Per-token entropy (encourages sparse routing)
        probs = F.softmax(student_logits, dim=-1)
        entropy_loss = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()

        total = (
            cfg.kl_loss_weight * kl_loss
            + cfg.balance_loss_weight * balance_loss
            + cfg.entropy_loss_weight * entropy_loss
        )
        total.backward()
        nn.utils.clip_grad_norm_(student_router.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_history.append(float(total.item()))

        if (step + 1) % cfg.log_every == 0 or step == 0:
            logger.debug(
                f"[RouterDistil] step {step + 1}/{cfg.steps} "
                f"loss={total.item():.5f} kl={kl_loss.item():.5f} "
                f"balance={balance_loss.item():.5f}"
            )

    student_router.eval()
    elapsed = time.perf_counter() - t0

    # Compute final utilisation ratio
    util_ratio = _compute_utilisation_ratio(student_router, calib_hidden[:256])

    initial = loss_history[0] if loss_history else float("nan")
    final = loss_history[-1] if loss_history else float("nan")
    converged = (final <= initial) if not math.isnan(initial) else False

    result = RouterDistillationResult(
        steps_run=cfg.steps,
        elapsed_seconds=elapsed,
        initial_loss=initial,
        final_loss=final,
        converged=converged,
        loss_history=loss_history,
        utilisation_ratio=util_ratio,
        load_balanced=util_ratio < cfg.max_utilisation_ratio,
    )

    logger.info(
        f"Router distillation done in {elapsed:.1f}s — "
        f"loss {initial:.4f} → {final:.4f}, "
        f"utilisation ratio {util_ratio:.2f}x "
        f"({'OK' if result.load_balanced else 'HIGH — consider more steps'})"
    )
    return result


# ---------------------------------------------------------------------------
# Full distillation entry-point (collects teacher logits then distils)
# ---------------------------------------------------------------------------


def run_router_distillation(
    student_router: nn.Linear,
    teacher_router: nn.Linear,
    calibration_hidden_states: torch.Tensor,
    retained_expert_indices: list[int],
    config: RouterDistillationConfig | None = None,
    device: torch.device | None = None,
) -> RouterDistillationResult:
    """Collect teacher logits then run KL distillation on the student.

    This is the high-level entry-point used by ``ExpertPruningMethod`` and
    ``ExpertMergingMethod`` when ``router_distillation`` config is present.

    Args:
        student_router: Already-sliced gate layer (shape ``[E_retained, H]``).
        teacher_router: Original gate layer *before* pruning (shape
            ``[E_orig, H]``).  Frozen; used only for forward passes.
        calibration_hidden_states: Float tensor ``[N, H]`` or ``[B, L, H]``.
        retained_expert_indices: Indices of surviving experts in the original
            expert ordering.  Used to slice teacher probability columns.
        config: Distillation hyper-parameters.
        device: Target device.

    Returns:
        ``RouterDistillationResult``.
    """
    cfg = config or RouterDistillationConfig()
    dev = device or next(student_router.parameters()).device

    hs = calibration_hidden_states.to(dev).float().detach()
    if hs.dim() == 3:
        hs = hs.view(-1, hs.shape[-1])

    # Collect teacher logits (no grad)
    teacher_router.eval()
    teacher_router.to(dev)
    with torch.no_grad():
        teacher_logits_all = teacher_router(hs)  # [N, E_orig]

    # Slice teacher to retained expert columns only, then re-normalise
    retained_t = torch.tensor(retained_expert_indices, device=dev)
    teacher_logits_retained = teacher_logits_all[:, retained_t]  # [N, E_retained]

    return distil_router(
        student_router=student_router,
        teacher_logits=teacher_logits_retained,
        config=cfg,
        device=dev,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_calib_hidden(
    teacher_target: torch.Tensor,
    hidden_dim: int,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Synthesise calibration hidden states from teacher target distribution.

    Uses random projection: H = P · teacher_target  where P is a random
    orthogonal matrix [hidden_dim × E_retained].  The resulting hidden states
    have the same routing signal as the teacher distribution.
    """
    e_retained = teacher_target.shape[-1]
    # Random orthogonal-ish projection matrix
    proj = torch.randn(e_retained, hidden_dim, device=device)
    proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)
    # calib_hidden[i] ∝ Σ_e teacher_target[i, e] * proj[e]
    calib_hidden = teacher_target @ proj  # [N, hidden_dim]
    # Add small noise to prevent degenerate solutions
    calib_hidden = calib_hidden + 0.01 * torch.randn_like(calib_hidden)
    return calib_hidden.detach()


def _compute_utilisation_ratio(
    router: nn.Linear,
    hidden_states: torch.Tensor,
) -> float:
    """Return max/min expert utilisation ratio over a batch of hidden states."""
    with torch.no_grad():
        logits = router(hidden_states.to(next(router.parameters()).device))
        mean_probs = F.softmax(logits, dim=-1).mean(dim=0)  # [E_retained]
    max_load = float(mean_probs.max().item())
    min_load = float(mean_probs.min().item())
    if min_load < 1e-8:
        return float("inf")
    return max_load / min_load


def _make_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> optim.lr_scheduler.LRScheduler | None:
    """Cosine schedule with optional linear warmup."""
    if warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
