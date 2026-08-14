# ViPym System Architecture

ViPym decouples compression, serving, and evaluation to ensure domain-agnostic reusability and statistical reproducibility.

```mermaid
flowchart TD
    subgraph Control Plane
        CFG[Configuration Loader] --> MAN[Manifest Generator]
        MAN --> RUNNER[Resumable Experiment Runner]
    end

    subgraph Compute Plane
        RUNNER --> DAG[Compression DAG Engine]
        DAG --> QUANT[AWQ / GPTQ / SmoothQuant / QuaRot]
        QUANT --> SERVE[Inference Runtime: vLLM / SGLang]
        SERVE --> EVAL[Evaluation Engine in gVisor Sandbox]
    end

    subgraph Analysis Plane
        EVAL --> METRICS[Telemetry & Quality Collector]
        METRICS --> PARETO[Multi-Objective Pareto Frontier Engine]
        PARETO --> REPORT[Report Generator: HTML / LaTeX / Plotly]
    end
```

## Lifecycle States & Resumability

Experiments persist state at every boundary:
`CREATED -> VALIDATED -> BASELINE_RUNNING -> BASELINE_COMPLETED -> COMPRESSION_RUNNING -> COMPRESSION_COMPLETED -> INFERENCE_VALIDATED -> EVALUATION_RUNNING -> EVALUATION_COMPLETED -> ANALYSIS_COMPLETED -> REPORT_COMPLETED`

If interrupted or stopped due to an out-of-memory error or hardware preemption, re-running `vipym run --config <path>` will resume directly from the last completed stage.
