# ViPym Operational Runbook

Operational guide for executing large-scale LLM compression, evaluating massive Mixture-of-Experts (MoE) models, and managing cloud infrastructure.

---

## 1. Multi-Stage Experiment Execution & Resumption

### Running a Production Recipe
```bash
vipym run recipes/kimi-k3-full.yaml --output-dir /opt/vipym/results/kimi_k3_run1
```

### Resuming an Interrupted / Evicted Experiment
ViPym automatically saves checkpoint state machines in `manifest.json`. If a spot instance is terminated or a process killed:
```bash
# Resume from the exact stage that was running (no redundant compute)
vipym run recipes/kimi-k3-full.yaml --output-dir /opt/vipym/results/kimi_k3_run1 --resume
```

---

## 2. Launching ViPym Studio in Cloud / Headless Environments

To monitor experiment DAG progress, Pareto frontiers, and live token throughput on a remote server:

```bash
# Start Studio API & SPA bound to localhost with token authentication
vipym studio --port 8000
```
On your local machine, establish an SSH tunnel:
```bash
ssh -N -L 8000:localhost:8000 user@gpu-instance.ec2.internal
```
Access the dashboard at `http://localhost:8000`. The token is stored at `~/.vipym/studio-token`.

---

## 3. Spot Instance Cost Optimization & ROI Validation

To calculate the multi-candidate Pareto frontier and generate enterprise cost projections:
```bash
vipym run recipes/cost-optimized-se.yaml -o results/cost_opt/
```
Inspect the deployment recommendation report:
```bash
cat results/cost_opt/analysis/recommendation.md
```
Or compare two independent compression runs:
```bash
vipym compare results/cost_opt/ results/quality_first/ -o comparison.html
```
