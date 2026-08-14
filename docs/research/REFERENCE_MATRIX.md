# ViPym — Authoritative Technical Reference Matrix

| Ref ID | Topic | Source Title | Authors / Organization | Venue / Year | Key Insights & Implementation Impact |
|---|---|---|---|---|---|
| `R-001` | Weight Quantization | *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* | Lin et al. (MIT) | MLSys 2024 | Salient weight protection based on activation magnitude; implemented in `vipym.compression.quantization.awq`. |
| `R-002` | Second-Order Quantization | *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* | Frantar et al. (IST Austria) | ICLR 2023 | Optimal Brain Surgeon inverse Hessian error compensation; implemented in `vipym.compression.quantization.gptq`. |
| `R-003` | Activation Quantization | *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* | Xiao et al. (MIT) | ICML 2023 | Per-channel activation outlier migration to weights; implemented in `vipym.compression.quantization.smoothquant`. |
| `R-004` | Rotational Transforms | *QuaRot: Outlier-Free 4-Bit Post-Training Quantization of LLMs* | Ashkboos et al. (ETH Zurich) | arXiv 2024 | Random orthogonal Hadamard rotations to eliminate outliers; implemented in `vipym.compression.transforms.spinquant`. |
| `R-005` | Microscaling Formats | *OCP Microscaling Formats (MX) Specification v1.0* | Open Compute Project (AMD, Arm, Intel, Meta, NVIDIA, Qualcomm) | OCP 2023 | Sub-byte MXFP4 / MXFP8 block floating point; implemented in `vipym.compression.quantization.mxfp`. |
| `R-006` | Sparsity | *A Simple and Effective Pruning Approach for Large Language Models (Wanda)* | Sun et al. (CMU) | ICLR 2024 | Weight $\times$ activation magnitude pruning without retraining; implemented in `vipym.compression.pruning.magnitude`. |
| `R-007` | MoE Architecture | *Kimi K3 Technical Report & Architecture Specification* | Moonshot AI | 2025/2026 | 2.8T MoE with 896 routed experts, 104B active per token, Hybrid KDA+MLA attention; modeled in `vipym.models.architectures.kimi_k3`. |
| `R-008` | Sandboxed Code Execution | *gVisor: Application Kernel for Containers* | Google Cloud | 2024 | User-space virtualization (`runsc`) for untrusted benchmark code; implemented in `vipym.evaluation.sandbox`. |
| `R-009` | Benchmark Methodology | *HumanEval / Pass@k Estimator* | Chen et al. (OpenAI) | arXiv 2021 | Unbiased hypergeometric $Pass@k$ formulation; implemented in `vipym.metrics.quality`. |
| `R-010` | Multi-Objective Optimization | *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II* | Deb et al. | IEEE TEC 2002 | Non-dominated sorting Pareto frontier algorithm; implemented in `vipym.analysis.pareto`. |
