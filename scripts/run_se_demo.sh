#!/usr/bin/env bash
# ==============================================================================
# ViPym: Software Engineering Lifecycle Compression & Pareto Analysis Demo
# Runs in ~2-5 minutes on standard CPU without requiring GPU hardware
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "========================================================================"
echo " Starting ViPym SE Lifecycle Compression & Pareto Analysis Demo"
echo "========================================================================"

# Allow degraded bare-subprocess execution if Docker is not available locally
export VIPYM_ALLOW_UNSAFE=1

# 1. Run the end-to-end experiment
echo "[1/2] Executing recipe: recipes/se-lifecycle-demo.yaml..."
python -m vipym.cli.main run recipes/se-lifecycle-demo.yaml --output results/ --no-resume

# 2. Check generated artifacts
RESULTS_DIR="results/se-lifecycle-demo-gpt2"
echo ""
echo "[2/2] Verifying generated analysis & report artifacts..."
echo "✓ Root Dashboard:      ${RESULTS_DIR}/report.html"
echo "✓ Interactive Pareto:  ${RESULTS_DIR}/analysis/pareto.html"
echo "✓ Recommendation:      ${RESULTS_DIR}/analysis/recommendation.md"
echo "✓ Evaluation Results:  ${RESULTS_DIR}/evaluations/humaneval.json"
echo "✓ Model Metadata:      ${RESULTS_DIR}/models/artifact_info.json"

echo ""
echo "========================================================================"
echo " Demo completed successfully! To explore interactive reports in Studio:"
echo "   vipym studio --artifacts-dir results/"
echo "========================================================================"
