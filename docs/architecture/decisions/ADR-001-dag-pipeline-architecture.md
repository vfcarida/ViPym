# ADR-001: Pipeline-as-a-DAG Execution Engine for Model Compression

## Status
Accepted

## Context
Traditional LLM compression tooling executes single-pass sequential quantization (e.g. baseline $\to$ AWQ $\to$ model). Modern compression techniques, however, require multi-stage, non-linear composition:
1. Orthogonal transforms (QuaRot / SpinQuant) to suppress activation outliers.
2. Semi-structured (2:4) sparsity or magnitude pruning.
3. Quantization (AWQ W4A16 or AutoRound).
4. KV-Cache quantization (FP8 / INT4).
5. Sequence or token distillation from a teacher model.

A linear array cannot express branched dependencies or stage caching.

## Decision
Implement `DirectedAcyclicCompressionPipeline` using Kahn's algorithm for topological sorting and cycle detection. Each node defines explicit prerequisite dependencies.

## Consequences
- Enables arbitrary composition of orthogonal transforms, pruning, quantization, and distillation.
- Automatically validates graph acyclicity before allocating GPU memory.
- Stage-level artifacts can be cached and reused across parallel experiment branches.
