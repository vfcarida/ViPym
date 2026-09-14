# Hugging Face Hub Publishing & Model Card Generation

ViPym provides native, one-command publishing of compressed model checkpoints directly to the [Hugging Face Hub](https://huggingface.co).

Each upload automatically synthesizes an evaluation-backed, transparent **Model Card** (`README.md`) detailing:
- Compression method and reduction ratio (e.g. AWQ 4-bit, SmoothQuant W8A8).
- Quality retention across benchmark suites (HumanEval, MBPP, SWE-bench Lite).
- Serving efficiency profiles (TTFT, P50/P95 latency, peak VRAM, cost per 1M tokens).
- Anti-contamination cryptographic audit status.
- Drop-in deployment snippets for `vLLM` and Hugging Face `transformers`.

---

## 1. Quickstart via CLI

Publish any local checkpoint directory created by ViPym:

```bash
# Authenticate with Hugging Face (or set HF_TOKEN environment variable)
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Publish with automatic model card generation
vipym publish checkpoints/stage_0_compressed \
    --repo-id my-org/Llama-3-8B-ViPym-AWQ \
    --message "Publish production AWQ 4-bit compressed weights"
```

### Dry-Run Mode
Validate your checkpoint and generate the model card locally without touching the network:

```bash
vipym publish checkpoints/stage_0_compressed \
    --repo-id my-org/Llama-3-8B-ViPym-AWQ \
    --dry-run
```

### CLI Reference

| Option | Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| `checkpoint_dir` | Positional | Path to local model checkpoint directory | Required |
| `--repo-id` | `-r` | Hugging Face Hub repository ID (`username/repo`) | Required |
| `--token` | `-t` | Hugging Face API access token | `$HF_TOKEN` |
| `--private` | — | Make the Hugging Face repository private | `False` |
| `--dry-run` | — | Validate files and synthesize README without uploading | `False` |
| `--message` | `-m` | Git commit message for the Hub upload | `"Upload compressed model via ViPym"` |

---

## 2. Python API

You can also integrate model publishing directly into your Python pipelines and scripts:

```python
from vipym.artifacts.publisher import HFModelPublisher, ModelCardMetadata

publisher = HFModelPublisher(token="hf_your_token")

# Optional: Customize evaluation metrics and metadata
metadata = ModelCardMetadata(
    model_name="Llama-3-8B-ViPym-AWQ",
    base_model="meta-llama/Meta-Llama-3-8B",
    compression_method="AWQ 4-Bit (W4A16)",
    original_size_gb=16.0,
    compressed_size_gb=4.8,
    benchmark_scores={
        "HumanEval Pass@1": 0.642,
        "MBPP Pass@1": 0.685,
    },
    baseline_scores={
        "HumanEval Pass@1": 0.655,
        "MBPP Pass@1": 0.690,
    },
    latency_p50_ms=18.4,
    vram_peak_gb=5.4,
    contamination_audit={
        "clean": True,
        "overlap_rate": 0.0,
        "sha256_digest": "3a87f1b2c...",
    },
)

result = publisher.publish(
    checkpoint_dir="./checkpoints/stage_0_compressed",
    repo_id="my-org/Llama-3-8B-ViPym-AWQ",
    model_card_metadata=metadata,
    dry_run=False,
)

print(f"Model successfully published: {result.repo_url}")
print(f"Total uploaded: {result.total_bytes / (1024**2):.1f} MB")
```

---

## 3. Automated Model Card Structure

Generated model cards adhere strictly to the Hugging Face standard and include:

1. **YAML Frontmatter**: Includes `base_model`, `license`, `tags` (`vipym`, `quantization`, `vllm`), and `pipeline_tag: text-generation`.
2. **Provenance Badges**: Visual status indicators for compression ratio, licensing, and anti-contamination validation.
3. **Quality Retention Table**: Side-by-side benchmark comparison between baseline and compressed checkpoints with calculated retention percentages.
4. **Instant Serving Commands**:
   - `vllm serve my-org/Llama-3-8B-ViPym-AWQ --gpu-memory-utilization 0.90`
   - Python code snippet for Hugging Face `transformers` and `vipym.models.loader`.
