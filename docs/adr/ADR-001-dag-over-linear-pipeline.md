# ADR-001: Directed Acyclic Graph (DAG) Pipeline over Linear Pipeline

## Status
Accepted

## Context
Traditional LLM compression tools (e.g. standard AutoGPTQ or basic quantization scripts) execute a single linear sequence of operations on a model (e.g. `load -> quantize -> save`). However, state-of-the-art compression strategies on modern architectures require complex multi-stage pipelines:
1. Orthogonal basis rotations (QuaRot / SpinQuant) before weight/activation quantization.
2. Structured N:M sparsity before quantization.
3. MoE expert surgical pruning followed by mixed-precision quantization on remaining experts.
4. Parallel branches generating multiple compressed candidates for multi-objective Pareto analysis from a single base model.

A strictly linear pipeline representation cannot express shared dependencies, topological ordering, or branch-and-evaluate workflows.

## Decision
ViPym models all compression workflows as Directed Acyclic Graphs (`DAGCompressionPipeline` / `DirectedAcyclicCompressionPipeline`):
- Each stage declares explicit prerequisite `dependencies: [stage_id_1, ...]`.
- Cycle detection and topological ordering are enforced at configuration compile time using **Kahn's Algorithm**.
- Intermediate stage outputs produce standardized `CompressionArtifact` objects that can be consumed by downstream sibling or child stages.

## Consequences
### Positive
- **Composability**: Researchers can combine rotation transforms, pruning, quantization, and KV-cache compression in arbitrary valid orders.
- **Reproducibility**: The complete pipeline topology is declaratively captured in JSON/YAML manifests.
- **Parallelism**: Independent branches in the DAG can be executed concurrently on multi-GPU nodes.

### Negative
- Slightly higher configuration complexity compared to single-command quantizers.
