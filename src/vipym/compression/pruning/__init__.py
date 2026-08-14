"""Pruning adapters."""

from vipym.compression.pruning.magnitude import (
    MagnitudePruningMethod,
    NMSparsityMethod,
    WandaPruningMethod,
)

__all__ = ["MagnitudePruningMethod", "NMSparsityMethod", "WandaPruningMethod"]
