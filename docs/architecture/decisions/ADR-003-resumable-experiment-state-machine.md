# ADR-003: Resumable Experiment Lifecycle State Machine and Stage Checkpointing

## Status
Accepted

## Context
Large-scale LLM compression and benchmark evaluation runs (e.g. 2.8T Kimi K3 on multi-node GPU clusters) can take several hours and are subject to hardware interruptions, Spot instance preemptions, or out-of-memory errors. Re-running the entire pipeline from scratch on failure is cost-prohibitive.

## Decision
Implement a 12-state explicit finite state machine (`ExperimentStateManager`) and a JSON checkpoint manager (`CheckpointManager`):
`CREATED -> VALIDATED -> BASELINE_RUNNING -> BASELINE_COMPLETED -> COMPRESSION_RUNNING -> COMPRESSION_COMPLETED -> INFERENCE_VALIDATED -> EVALUATION_RUNNING -> EVALUATION_COMPLETED -> ANALYSIS_COMPLETED -> REPORT_COMPLETED`

Upon reinvocation (`vipym run --config <path>`), completed stages (such as Baseline Evaluation or Compression DAG execution) are read directly from disk checkpoints.

## Consequences
- Zero redundant computation on re-runs.
- Clear state visibility and auditability.
