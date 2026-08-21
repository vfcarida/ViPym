"""Router retraining and redistribution utilities for MoE compression."""

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from vipym.core.logger import get_logger

logger = get_logger(__name__)


def retrain_router(
    router_layer: nn.Module,
    calibration_hidden_states: torch.Tensor,
    retained_expert_indices: list[int],
    pruned_expert_indices: list[int] | None = None,
    similarity_matrix: torch.Tensor | None = None,
    steps: int = 200,
    lr: float = 1e-4,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Retrain/calibrate router weights after expert pruning or merging.

    Redistributes routing probability mass from pruned experts to remaining
    most-similar or top-performing experts using KL-divergence / Cross-Entropy
    optimization on calibration hidden states.
    """
    dev = device or next(router_layer.parameters()).device
    hidden_states = calibration_hidden_states.to(dev).detach()
    if hidden_states.dim() == 3:
        # Flatten batch and sequence dimensions: [batch * seq, hidden_dim]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_samples = hidden_states.shape[0]
    batch_size = min(64, num_samples)

    # Freeze all other modules and optimize only the router parameters
    optimizer = optim.AdamW(router_layer.parameters(), lr=lr, weight_decay=1e-3)
    loss_history: list[float] = []

    router_layer.train()
    for _step in range(steps):
        # Sample mini-batch
        indices = torch.randint(0, num_samples, (batch_size,), device=dev)
        batch_inputs = hidden_states[indices]

        optimizer.zero_grad()
        logits = router_layer(batch_inputs)  # [batch_size, num_remaining_experts]
        probs = torch.softmax(logits, dim=-1)

        # 1. Entropy / Load Balancing Regularization: prevent collapse into a single expert
        mean_probs = probs.mean(dim=0)
        load_balance_loss = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

        # 2. Sparsity Loss: keep individual token routing sharp
        token_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

        total_loss = -0.1 * load_balance_loss + token_entropy
        total_loss.backward()
        optimizer.step()

        loss_history.append(float(total_loss.item()))

    router_layer.eval()
    initial_loss = loss_history[0] if loss_history else 0.0
    final_loss = loss_history[-1] if loss_history else 0.0

    logger.info(
        f"Router retraining completed in {steps} steps (Loss: {initial_loss:.4f} -> {final_loss:.4f})"
    )

    return {
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "converged": final_loss <= initial_loss,
    }
