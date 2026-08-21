# ADR-002: Mixture-of-Experts (MoE) First Architecture Design

## Status
Accepted

## Context
Frontier open-weight architectures (e.g. Moonshot AI Kimi K3 with 2.8 Trillion total parameters / 896 routed + 2 shared experts, DeepSeek-V3 with 671 Billion total parameters, and Mixtral 8x7B) are dominated by sparse Mixture-of-Experts topologies. 

Standard dense compression methods suffer significant limitations on MoE models:
- Dense quantizers treat all weights uniformly, ignoring the stark contrast between critical shared attention layers and sparsely activated domain experts.
- Traditional pruning removes random weights across all experts rather than surgically pruning entire inactive experts or merging expert centroids based on semantic similarity.
- Calibration datasets drawn from general English prose fail to activate domain-specialized programming and mathematical experts.

## Decision
ViPym is architected from the ground up with **MoE-First Introspection**:
1. **First-Class MoE Support**: `ModelAdapter` inspects `num_experts`, `num_selected_experts`, `shared_experts`, and routing matrices.
2. **MoE-Specific Surgical Operations**:
   - `ExpertProfiler`: Measures routing frequency, token routing entropy, and importance per expert under target benchmark workloads.
   - `ExpertPruner`: Prunes low-activation expert networks and reshapes router projection weights.
   - `ExpertMerger`: Merges expert weights via cosine similarity clustering.
3. **Asymmetric Mixed-Precision Quantization**: Allows higher bit-width for shared attention blocks (e.g. 8-bit) and aggressive quantization for sparse routed experts (e.g. 4-bit / MXFP4).

## Consequences
### Positive
- Enables compression of multi-trillion parameter MoEs (Kimi K3, DeepSeek-V3) from 64 GPUs down to 8–16 GPUs without catastrophic degradation.
- Preserves specialized coding and reasoning experts.

### Negative
- Requires architecture-aware tensor inspection routines for novel MoE router definitions.
