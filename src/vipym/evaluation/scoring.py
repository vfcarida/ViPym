"""Scoring utilities for code generation benchmarks.

Implements the standard unbiased pass@k estimator from:
"Evaluating Large Language Models Trained on Code" (Chen et al., OpenAI, 2021)

Formula:
  pass@k = 1 - binom(n - c, k) / binom(n, k) = 1 - prod_{i=n-c+1}^{n} (1 - k / i)
where n is the total number of samples generated per problem,
and c is the number of samples that pass all unit tests.
"""

from __future__ import annotations

import math
from typing import Sequence


def compute_pass_at_k(
    n: int | Sequence[int],
    c: int | Sequence[int],
    k: int,
) -> float:
    """Compute the unbiased pass@k metric.

    Can be called for a single problem:
      compute_pass_at_k(n=10, c=5, k=1) -> 0.5

    Or across a list of problems:
      compute_pass_at_k(n=[10, 10], c=[5, 2], k=1) -> 0.35
    """
    if isinstance(n, (list, tuple)) and isinstance(c, (list, tuple)):
        if len(n) != len(c):
            raise ValueError(f"Lengths of n ({len(n)}) and c ({len(c)}) must match.")
        if not n:
            return 0.0
        scores = [compute_pass_at_k(n_i, c_i, k) for n_i, c_i in zip(n, c)]
        return sum(scores) / len(scores)

    # Scalar computation
    n_val = int(n)  # type: ignore[arg-type]
    c_val = int(c)  # type: ignore[arg-type]

    if k <= 0:
        return 0.0
    if c_val <= 0:
        return 0.0
    if n_val <= 0:
        return 0.0
    if n_val - c_val < k:
        return 1.0
    if k > n_val:
        # Fallback estimate when evaluating k > n
        prob_fail = 1.0 - (c_val / n_val)
        return float(1.0 - (prob_fail**k))

    # Combinatorial formula: 1 - prod_{i=n-c+1}^n (1 - k / i)
    prod = 1.0
    for i in range(n_val - c_val + 1, n_val + 1):
        prod *= 1.0 - (k / i)
    return float(1.0 - prod)


def calculate_pass_at_k_metrics(
    task_correctness: Sequence[Sequence[bool]],
    k_values: Sequence[int] = (1, 10, 100),
) -> dict[str, float]:
    """Calculate pass@k for all requested k values given boolean task results.

    Args:
        task_correctness: List of boolean lists, where task_correctness[i][j] is True
                          if sample j for task i passed all unit tests.
        k_values: List of k thresholds to compute (e.g. [1, 10, 100]).

    Returns:
        Dictionary mapping metric names (e.g. 'pass@1', 'pass@10') to float scores.
    """
    if not task_correctness:
        return {f"pass@{k}": 0.0 for k in k_values}

    n_list = [len(samples) for samples in task_correctness]
    c_list = [sum(1 for s in samples if s) for samples in task_correctness]

    metrics: dict[str, float] = {}
    for k in k_values:
        score = compute_pass_at_k(n_list, c_list, k)
        metrics[f"pass@{k}"] = score

    return metrics
