# 5-Minute Quick-Start Guide (CPU / Laptop Demo)

Run a complete end-to-end model compression, benchmark evaluation, and Pareto analysis in under 5 minutes on any standard laptop without requiring a GPU or cloud account.

---

## 1. Quick Installation

```bash
git clone https://github.com/vfcarida/ViPym.git
cd ViPym
pip install -e ".[dev]"
```

Verify your environment readiness:
```bash
vipym doctor
```

---

## 2. Run the 5-Minute Demo

Execute the pre-configured lightweight demo recipe:

```bash
# Allow local test execution if Docker is not currently running
export VIPYM_ALLOW_UNSAFE=1

# Run the end-to-end experiment
vipym run recipes/se-lifecycle-demo.yaml --output results/
```

### What Happens During Execution:
1. **Model Loading**: Downloads and profiles the lightweight GPT-2 model.
2. **Baseline Evaluation**: Runs baseline zero-shot inference on HumanEval tasks.
3. **Compression Pipeline**: Applies Wanda 50% pruning followed by second-order GPTQ 4-bit quantization.
4. **Compressed Evaluation**: Evaluates the compressed model in sandbox mode.
5. **Pareto Frontier Generation**: Computes cost vs quality tradeoffs.
6. **Recommendation Synthesis**: Produces deployment recommendations and cost projections.

---

## 3. View Interactive Results

Launch ViPym Studio to explore the interactive Pareto chart, stage progress, and benchmark scores:

```bash
vipym studio --artifacts-dir results/
```

Open `http://127.0.0.1:8080` in your web browser.

---

## 4. Explore Generated Files

All generated reports and models are saved to `results/se-lifecycle-demo-gpt2/`:

- `report.html` — Standalone HTML dashboard with plots and tables.
- `analysis/pareto.html` — Interactive Plotly chart of latency, memory, cost, and quality.
- `analysis/recommendation.md` — Human-readable executive summary & recommendations.
- `evaluations/humaneval.json` — Per-problem test results and pass rates.
- `models/artifact_info.json` — Quantized and pruned model metadata.
