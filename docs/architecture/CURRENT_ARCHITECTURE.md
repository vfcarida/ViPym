# ViPym — Current System Architecture

```mermaid
graph TD
    subgraph CLI Entrypoint
        CLI[vipym CLI - Typer] --> DOC[Doctor Diagnostics]
        CLI --> RUN[Resumable Runner]
        CLI --> BASE[Baseline Engine]
        CLI --> COMP[Compression Engine]
        CLI --> EVAL[Evaluation Engine]
        CLI --> ANALYZE[Pareto Analyzer]
    end

    subgraph Core & Experiments
        RUN --> STATE[ExperimentStateManager: 12 States]
        RUN --> CHECK[CheckpointManager: JSON Snapshots]
        RUN --> MAN[ReproducibilityManifest: Provenance & Hashes]
    end

    subgraph Compression Pipeline
        RUN --> DAG[DAG Engine: Kahn's Topological Sort]
        DAG --> Q1[AWQ / GPTQ / SmoothQuant / AutoRound]
        DAG --> Q2[FP8 / MXFP4 / MXFP8 Microscaling]
        DAG --> T1[QuaRot / SpinQuant Orthogonal Transforms]
        DAG --> P1[2:4 Semi-Structured & Wanda Sparsity]
        DAG --> D1[Sequence & Logit Distillation]
        DAG --> KV1[FP8 & INT4 KV-Cache Quantization]
    end

    subgraph Serving & Evaluation
        RUN --> INFER[Inference Backends: vLLM / SGLang / HF]
        INFER --> BENCH[BenchmarkRunner]
        BENCH --> SUITES[HumanEval / MBPP / LiveCodeBench / SWE-bench]
        SUITES --> SANDBOX[gVisor / Docker Sandbox Isolation]
    end

    subgraph Analysis & Reporting
        BENCH --> TEL[Telemetry & Quality Metrics Collector]
        TEL --> PARETO[ParetoFrontierOptimizer]
        PARETO --> REP[ReportGenerator: HTML / Markdown / LaTeX / Plotly]
    end
```

## Control & Data Flow

1. **Config Ingestion:** Pydantic v2 loads and strictly validates YAML configurations against schema definitions.
2. **Model Introspection:** Model adapters extract parameter counts, MoE expert counts, active routing, and context limits.
3. **Immutable Baseline:** Serves target model uncompressed; computes baseline Pass@1, TTFT, ITL, and peak VRAM.
4. **DAG Execution:** Topologically traverses and executes discrete compression stages, serializing artifacts.
5. **Sandboxed Evaluation:** Generates completions and executes unit tests in gVisor/Docker containers with zero network access and scrubbed environments.
6. **Multi-Objective Optimization:** Evaluates non-dominated Pareto frontier points across capability, memory, latency, and cost.
7. **Report Synthesis:** Generates interactive Plotly dashboards, Markdown summaries, and publication-ready LaTeX tables.
