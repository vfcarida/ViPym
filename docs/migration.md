# ViPym Migration & Upgrade Guide

This document assists users and developers in transitioning between ViPym versions and migrating legacy compression/evaluation scripts to modern DAG pipelines.

---

## Migrating from Legacy Scripts to ViPym 0.1.0+

### Legacy Pattern (Single-shot Imperative Scripts)
```python
# DEPRECATED: Manual single-stage script without telemetry or checkpoints
from transformers import AutoModelForCausalLM
import auto_gptq

model = AutoModelForCausalLM.from_pretrained("gpt2")
# ... manual quantization ...
# ... manual benchmark scoring ...
```

### Modern ViPym Pattern (Declarative Resumable DAG)
```yaml
# recipe.yaml
experiment_id: modern-gpt2-compression
model:
  id: openai-community/gpt2

compression_pipeline:
  - stage_id: wanda_sparsity
    method: prune_wanda
    parameters: { sparsity: 0.5 }
  - stage_id: awq_4bit
    method: quantize_awq
    parameters: { bits: 4 }

evaluation:
  suites: [humaneval, mbpp]
  max_workers: 4

cost_assumptions:
  aws_ec2_hourly_rate: 0.50
```

Execute via CLI or Python API:
```bash
vipym run recipe.yaml -o results/
```
```python
from vipym.experiments.runner import ResumableExperimentRunner
from vipym.config.schema import ViPymExperimentConfig

config = ViPymExperimentConfig.from_yaml("recipe.yaml")
runner = ResumableExperimentRunner(config=config)
summary = runner.run(resume=True)
```

---

## Deprecation Schedule

| Deprecated Feature | Replacement | Target Removal |
| :--- | :--- | :--- |
| `vipym.compression.distillation.logit_distill` | `vipym.distillation.DistillationMethod` | v0.2.0 |
| `ViPymRunner` | `vipym.experiments.runner.ResumableExperimentRunner` | v0.2.0 |
| `ExecutionStatus` | `vipym.experiments.state.ExperimentState` | v0.2.0 |
