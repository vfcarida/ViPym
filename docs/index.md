# Getting Started with ViPym

**ViPym — Shrinking LLMs, Preserving Intelligence**  
A modular, reproducible, production-grade LLM compression benchmarking and evaluation framework.

---

## 1. Quick Installation

```bash
# Clone repository
git clone https://github.com/vfcarida/ViPym.git
cd ViPym

# Install in editable mode with all optional dependencies
pip install -e ".[all]"
```

## 2. Health & Diagnostic Check

Run `vipym doctor` to verify system readiness across Python, CUDA, PyTorch, vLLM, Docker, VRAM, and Disk space:

```bash
vipym doctor
```

## 3. Running Your First Experiment

Execute a fast, local CPU/GPU smoke test:

```bash
vipym run --config configs/experiments/smoke_test.yaml
```

The run will:
1. Validate configuration schemas.
2. Establish an immutable baseline.
3. Apply the compression pipeline DAG (e.g. AWQ W4A16).
4. Serve the compressed model via vLLM / SGLang.
5. Evaluate tasks in an isolated gVisor/Docker sandbox.
6. Calculate multi-objective Pareto frontiers.
7. Generate interactive Plotly HTML dashboards, Markdown summaries, and publication-ready LaTeX tables.

---

## 4. Key CLI Commands

| Command | Description |
|---|---|
| `vipym run --config <path>` | Execute full end-to-end experiment |
| `vipym validate --config <path>` | Validate configuration schema |
| `vipym baseline --model <id>` | Establish uncompressed baseline |
| `vipym compress --model <id> --method awq` | Run single compression algorithm |
| `vipym evaluate --model <path> --suite humaneval` | Run benchmark evaluation suite |
| `vipym analyze --dir ./artifacts` | Calculate Pareto frontiers on results |
| `vipym report --dir ./artifacts --format html` | View synthesized reports |
| `vipym inspect-model --model moonshotai/Kimi-K3` | Inspect MoE architecture and parameter topology |
| `vipym list-compressors` | List all registered compression algorithms |
| `vipym list-evaluators` | List all registered benchmark suites |
