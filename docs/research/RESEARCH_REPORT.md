# ViPym — External Technical Research & Literature Synthesis

**Research Scope:** LLM Compression, MoE Surgery, Quantization Dynamics, Sparsity, Sandboxed Evaluation, Multi-Objective Pareto Optimization.

---

## 1. Authoritative Literature & Methodological Taxonomy

| ID | Paper / Source | Authors & Venue | Key Scientific Contribution | ViPym Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **S001** | **AWQ: Activation-aware Weight Quantization** | Lin et al. (*MLSys 2024*) | Salient channel protection (top 1% activation magnitude preserved in FP16/INT8). | [`src/vipym/compression/methods/quantization/awq.py`](../../src/vipym/compression/methods/quantization/awq.py) |
| **S002** | **GPTQ: Accurate Post-Training Quantization** | Frantar et al. (*ICLR 2023*) | Second-order inverse Hessian error compensation for layer-by-layer weight rounding. | [`src/vipym/compression/methods/quantization/gptq.py`](../../src/vipym/compression/methods/quantization/gptq.py) |
| **S003** | **Wanda: Pruning by Weights and Activations** | Sun et al. (*ICML 2024*) | Weight removal scoring via $S_{ij} = \|W_{ij}\| \cdot \|X_j\|_2$ without expensive retraining. | [`src/vipym/compression/methods/pruning/wanda.py`](../../src/vipym/compression/methods/pruning/wanda.py) |
| **S004** | **QuaRot & SpinQuant** | Ashkboos et al. (*NeurIPS 2024*) | Orthogonal randomized Walsh-Hadamard rotations eliminating cross-channel activation outliers. | [`src/vipym/compression/transforms/spinquant.py`](../../src/vipym/compression/transforms/spinquant.py) |
| **S005** | **SWE-bench** | Jimenez et al. (*ICLR 2024*) | Repository-level software engineering benchmarks with Docker execution of generated Git diffs. | [`src/vipym/evaluation/suites/swebench.py`](../../src/vipym/evaluation/suites/swebench.py) |
| **S006** | **HumanEval / EvalPlus** | Chen et al. (2021) / Liu et al. (2024) | Functional correctness with rigorous expanded test suites and Pass@1 computation. | [`src/vipym/evaluation/suites/humaneval.py`](../../src/vipym/evaluation/suites/humaneval.py) |
| **S007** | **BigCodeBench** | Zhuo et al. (*2024*) | Real-world library utilization and multi-dependency code synthesis evaluation. | [`src/vipym/evaluation/suites/bigcodebench.py`](../../src/vipym/evaluation/suites/bigcodebench.py) |
| **S009** | **vLLM: PagedAttention** | Kwon et al. (*SOSP 2023*) | Continuous batching, virtual memory KV-cache allocation, and tensor parallel serving. | [`src/vipym/inference/vllm_engine.py`](../../src/vipym/inference/vllm_engine.py) |
| **S011** | **MC-SMoE / MoE Compression** | Li et al. (*2024*) | Expert activation profiling, cosine similarity clustering, and router knowledge distillation. | [`src/vipym/compression/moe/`](../../src/vipym/compression/moe/) |

---

## 2. In-Depth Answers to Core Research Questions

### Q1: What is the current best practice for MoE expert pruning with router retraining?
* **Best Practice Protocol**:
  1. **Domain-Specific Expert Profiling**: Measure token assignment frequency across all routed experts using target domain calibration data (e.g. Python code corpora via [`CalibrationDatasetManager`](../../src/vipym/data/calibration.py)).
  2. **Expert Similarity Clustering**: Compute pairwise cosine similarity between expert FFN weight matrices ($W_{gate}, W_{up}, W_{down}$). Highly redundant experts ($\text{sim} > 0.85$) are merged via weighted averaging; low-utility experts ($\text{routing frequency} < 0.5\%$) are pruned.
  3. **Router Distillation**: Freeze all remaining expert weights and train only the router gating projection layers using KL-divergence loss against the uncompressed teacher model's soft routing probabilities:
     $$\mathcal{L}_{router} = \tau^2 \cdot D_{KL}\left(\text{Softmax}\left(\frac{z_{student}}{\tau}\right) \;\Big\|\; \text{Softmax}\left(\frac{z_{teacher}}{\tau}\right)\right)$$
* **ViPym Implementation**: Implemented in [`src/vipym/compression/moe/router_distillation.py`](../../src/vipym/compression/moe/router_distillation.py) and [`src/vipym/compression/methods/expert_pruning.py`](../../src/vipym/compression/methods/expert_pruning.py).

---

### Q2: How do production evaluation harnesses load HumanEval/MBPP datasets at scale?
* **Best Practice Protocol**:
  - Direct integration with official Hugging Face datasets (`openai/openai_humaneval`, `google-research-datasets/mbpp`) supporting offline fallback caching.
  - Test harness wrapping: Prompts are formatted with exact function signatures, and generated completions are concatenated with canonical test cases into sandboxed isolated sub-processes.
  - Multi-worker execution: Running benchmark tasks concurrently with a worker pool ([`ThreadPoolExecutor`](../../src/vipym/evaluation/runner.py)) achieves 4x–8x throughput speedup without benchmark execution bottlenecks.

---

### Q3: What are the failure modes of post-training quantization on MoE models?
* **Failure Modes & Mitigations**:
  1. **Rare Expert Starvation**: Calibration datasets with general text fail to activate specialized code/math experts, leading to inaccurate quantization scale factors. *Mitigation:* Ingest balanced code corpora with anti-contamination filters.
  2. **Shared Expert Outlier Contamination**: Shared experts process every token, accumulating severe kurtosis and activation outliers. *Mitigation:* Mixed-precision quantization (keep shared experts in FP8 or INT8, quantize routed experts to W4A16).
  3. **Routing Instability**: Quantizing router weights shifts top-$k$ expert selection. *Mitigation:* Keep router linear projection heads in FP16/BF16.

---

### Q4: How does SWE-bench agent evaluation actually work (scaffolding & sandbox)?
* **Architecture**:
  1. **Task Instantiation**: Ingest issue description, repo name, and base commit SHA.
  2. **Agent Scaffolding**: Provide iterative tool actions (search files, view code, execute tests, produce unified diff).
  3. **Evaluation Sandbox**: Apply the candidate git patch (`git apply patch.diff`) in a secure Docker container, execute test suites (`eval_patch`), and verify whether failing test cases pass without regressions.
* **ViPym Implementation**: [`src/vipym/evaluation/suites/swebench.py`](../../src/vipym/evaluation/suites/swebench.py) and [`src/vipym/evaluation/agents/swe_agent.py`](../../src/vipym/evaluation/agents/swe_agent.py).

---

### Q5: What Pareto optimization algorithms are appropriate for < 100 data points?
* **Best Practice**:
  - For $N < 100$ candidate models, evolutionary algorithms (NSGA-II) introduce unnecessary stochastic approximations.
  - **Exact Non-Dominated Sorting (Kung's Algorithm / Direct Dominance)** computes the exact global Pareto frontier in $O(M \cdot N^2)$ time ($< 1\text{ ms}$).
  - Multi-objective dominance criterion: Candidate $A$ dominates $B$ ($A \succ B$) if $A$ is better or equal in all objectives (Pass@1 $\uparrow$, VRAM $\downarrow$, Latency $\downarrow$, Cost $\downarrow$) and strictly better in at least one.
* **ViPym Implementation**: [`src/vipym/analysis/pareto.py`](../../src/vipym/analysis/pareto.py).

---

### Q6: How to measure real inference latency/VRAM without interfering with the model?
* **Best Practice**:
  1. **Warmup Request Exclusion**: Discard the first 3–5 requests to avoid CUDA initialization, JIT compilation, and kernel loading latency artifacts.
  2. **High-Resolution Monotonic Timing**: Measure Time-To-First-Token (TTFT) and Inter-Token Latency (ITL) using `time.perf_counter()`.
  3. **Peak Memory Statistics**: Query `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` with `torch.cuda.reset_peak_memory_stats()` after each evaluation pass.
* **ViPym Implementation**: [`src/vipym/telemetry/profiler.py`](../../src/vipym/telemetry/profiler.py).
