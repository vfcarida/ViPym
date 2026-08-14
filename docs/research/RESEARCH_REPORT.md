# ViPym — External Technical Research & Literature Synthesis

**Research Scope:** LLM Compression, MoE Architectures, Quantization, Sparsity, Sandboxed Evaluation, Cloud Cost Modeling.

---

## 1. Executive Summary of Authoritative Literature

Modern frontier LLMs (e.g. Moonshot AI Kimi K3, DeepSeek-V3, LLaMA-3) contain hundreds of billions to trillions of parameters, making inference economics and memory bandwidth the primary bottleneck for real-world deployment. 

Key research paradigms addressed in ViPym:
1. **Outlier Suppression & Rotational Invariance (QuaRot / SpinQuant):** Pre-multiplying activations and weights by randomized orthogonal Walsh-Hadamard matrices eliminates activation outliers across intermediate layers without altering model outputs, enabling 4-bit weight and activation quantization ($W4A4$) without catastrophic degradation.
2. **MoE Specialization & Microscaling (MXFP4 / MXFP8):** Massive mixture-of-experts models (such as Kimi K3 with 896 routed experts and 104B active parameters) benefit from sub-byte microscaling formats (OCP MXFP4/MXFP8) where 32-element scaling blocks share an E8M0 exponent.
3. **Hardware-Accelerated 2:4 Semi-Structured Sparsity (NVIDIA Ampere/Hopper/Blackwell):** Restricting weight matrices to 2 non-zero values per 4 consecutive elements provides $2\times$ theoretical tensor core compute throughput while preserving over 98% baseline capability when paired with magnitude pruning or Wanda (Weight and Activation).
4. **Sandboxed Evaluation Security (gVisor / seccomp):** Untrusted LLM code execution on benchmarks (HumanEval, MBPP, SWE-bench) requires hypervisor/user-space kernel isolation (`runsc`), network isolation, and dropped Linux capabilities (`CAP_NET_RAW`, `CAP_SYS_ADMIN`).

---

## 2. Research Questions & Findings

### Q1: How to compress MoE models with hundreds of sparse routed experts without cross-expert degradation?
* **Finding:** Experts have varying activation frequencies and token routing distributions. Applying Activation-Aware Weight Quantization (AWQ) or AutoRound with domain-specific calibration data prevents degradation in rarely-routed specialized experts.

### Q2: How should KV-cache memory be compressed for 1M context window workloads?
* **Finding:** FP8 (E4M3) KV-cache quantization reduces memory bandwidth and VRAM by 50% with near-zero perplexity impact. For extreme context windows, INT4 per-head asymmetric group quantization yields $4\times$ memory reduction.

### Q3: What statistical methods should be used for evaluating LLM compression trade-offs?
* **Finding:** A single scalar accuracy metric is misleading. Non-dominated Pareto frontier optimization across $\text{Pass@1}$, $\text{Peak VRAM}$, $\text{Latency (P50/P95)}$, and $\text{Cloud Cost (\$/hr)}$ gives a complete multi-objective view.
