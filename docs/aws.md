# AWS Ephemeral Orchestration & Cost Methodology

ViPym provides ephemeral cloud orchestration for large-scale GPU experiments (such as **Kimi K3** 2.8T MoE on multi-node p5.48xlarge clusters) without requiring permanent standing infrastructure.

## Ephemeral Lifecycle Flow

```mermaid
sequenceDiagram
    participant User as CLI / GitHub Action
    participant EC2 as Ephemeral EC2 (Spot / On-Demand)
    participant S3 as S3 Artifact Bucket
    participant CW as CloudWatch

    User->>EC2: Launch Ephemeral Node with Watchdog User-Data
    EC2->>S3: Pull Model Checkpoint & Configs
    Note over EC2: Execute Baseline -> Quantize -> Serve -> Eval
    EC2->>S3: Push Checkpoints, Reports & Manifests
    EC2->>CW: Emit Telemetry & Logs
    EC2->>EC2: Self-Terminating Poweroff (sudo poweroff)
```

## Traceable Cost Formula

$$\text{Total Cost} = (T_{\text{baseline}} + T_{\text{compress}} + T_{\text{eval}}) \times R_{\text{EC2}} + (S_{\text{weights}} \times R_{\text{S3}}) + (D_{\text{egress}} \times R_{\text{transfer}})$$

All calculations are traceable to explicit parameters in `CostAssumptionConfig`.
