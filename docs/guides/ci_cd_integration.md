# Automated CI/CD Model Evaluation Quality Gates

ViPym provides native CI/CD evaluation gates designed to prevent code-generation accuracy regressions and compilation failures in continuous integration pipelines.

Whenever a new checkpoint, pruned model, or quantized variant is committed or proposed in a Pull Request, ViPym evaluates the artifact against standardized benchmark suites (such as HumanEval, MBPP, or SWE-bench Lite) and automatically blocks merging if predetermined quality thresholds are breached.

---

## 1. GitHub Actions Integration

### Using the Reusable ViPym Quality Gate Action

You can drop the ViPym action directly into any GitHub Actions workflow:

```yaml
name: Model Quality Gate

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  evaluate-checkpoint:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code & Models
        uses: actions/checkout@v4

      - name: Run ViPym Quality Gate
        uses: vfcarida/ViPym/.github/actions/eval-gate@main
        with:
          model-path: "checkpoints/stage_0_compressed"
          suite: "humaneval"
          limit: "25" # Quick gate on 25 tasks for fast PR feedback
          backend: "hf"
          min-pass1: "0.60" # Enforce at least 60% Pass@1
          min-compile-rate: "0.85" # Enforce at least 85% compile rate
          output-json: "eval-gate-results.json"

      - name: Upload Gate Metrics Artifact
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: vipym-gate-results
          path: eval-gate-results.json
```

### Generated Step Summary
The action automatically appends a rich Markdown table into GitHub Actions' `$GITHUB_STEP_SUMMARY`:

| Metric | Measured | Required Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Pass@1** | 0.6400 | 0.60 | ✅ PASS |
| **Compile Rate** | 0.9200 | 0.85 | ✅ PASS |

**Suite**: `humaneval` | **Model**: `checkpoints/stage_0_compressed` | **Overall Result**: **PASSED**

---

## 2. CLI Quality Gating (`vipym evaluate --gate`)

You can run the quality gate directly from any shell, terminal, or build script:

```bash
vipym evaluate \
    --model meta-llama/Meta-Llama-3-8B \
    --suite humaneval \
    --backend hf \
    --limit 50 \
    --gate \
    --min-pass1 0.65 \
    --min-compile-rate 0.90 \
    --output-json ./reports/gate_report.json
```

### Exit Codes & Failure Behavior
- **Exit Code 0**: All threshold criteria satisfied.
- **Exit Code 1**: Quality gate violation detected (e.g. `Pass@1 (0.580) fell below minimum required threshold (0.650)`).

The JSON output contains a complete machine-readable manifest:
```json
{
  "model": "meta-llama/Meta-Llama-3-8B",
  "suite": "humaneval",
  "total_tasks": 50,
  "passed_tasks": 34,
  "pass_at_1": 0.68,
  "compile_rate": 0.94,
  "runtime_error_rate": 0.06,
  "duration_seconds": 32.4,
  "gate_enabled": true,
  "gate_passed": true,
  "violations": [],
  "thresholds": {
    "min_pass1": 0.65,
    "min_compile_rate": 0.90
  }
}
```

---

## 3. GitLab CI Configuration

For GitLab CI, configure a dedicated model validation job in `.gitlab-ci.yml`:

```yaml
model_quality_gate:
  stage: test
  image: python:3.12-slim
  script:
    - pip install -U pip
    - pip install "vipym[eval,torch]"
    - vipym evaluate --model "$CI_PROJECT_DIR/checkpoints/latest" --suite humaneval --limit 20 --gate --min-pass1 0.60 --output-json gate-results.json
  artifacts:
    when: always
    paths:
      - gate-results.json
```
