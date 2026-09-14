"""Unit tests for Hugging Face Hub Publisher and Model Card Generator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vipym.artifacts.publisher import (
    HFModelPublisher,
    ModelCardMetadata,
    generate_model_card,
    infer_metadata_from_directory,
)
from vipym.cli.main import app

runner = CliRunner()


def test_generate_model_card_full() -> None:
    """Verify model card markdown generation with complete metadata."""
    meta = ModelCardMetadata(
        model_name="Llama-3-8B-ViPym-AWQ-W4A16",
        base_model="meta-llama/Meta-Llama-3-8B-Instruct",
        compression_method="AWQ 4-Bit (W4A16)",
        original_size_gb=16.0,
        compressed_size_gb=4.8,
        compression_ratio=3.33,
        benchmark_scores={"HumanEval Pass@1": 0.642, "MBPP Pass@1": 0.685},
        baseline_scores={"HumanEval Pass@1": 0.655, "MBPP Pass@1": 0.690},
        latency_p50_ms=18.4,
        latency_p95_ms=24.1,
        throughput_tok_s=98.2,
        vram_peak_gb=5.4,
        cost_per_1m_tokens=0.0012,
        contamination_audit={
            "clean": True,
            "sha256_digest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "overlap_rate": 0.0,
        },
    )

    card = generate_model_card(meta)

    # Validate YAML Frontmatter
    assert "---" in card
    assert "base_model: meta-llama/Meta-Llama-3-8B-Instruct" in card
    assert "library_name: vipym" in card
    assert "pipeline_tag: text-generation" in card

    # Validate Badges & Sections
    assert "# Llama-3-8B-ViPym-AWQ-W4A16" in card
    assert "Anti--Contamination" in card
    assert "PASSED (0.0% Leakage)" in card
    assert "AWQ 4-Bit (W4A16)" in card
    assert "3.3x reduction" in card

    # Validate Evaluation Table & Retention
    assert "| **HumanEval Pass@1** | 0.655 | **0.642** | 98.0% |" in card
    assert "| **MBPP Pass@1** | 0.690 | **0.685** | 99.3% |" in card

    # Validate Serving Snippets
    assert "vllm serve Llama-3-8B-ViPym-AWQ-W4A16" in card
    assert "from vipym.models.loader import load_model" in card
    assert "AutoModelForCausalLM" in card


def test_generate_model_card_minimal_dict() -> None:
    """Verify model card generation from a dictionary with minimal fields."""
    data = {
        "model_name": "Minimal-Model",
        "base_model": "gpt2",
    }
    card = generate_model_card(data)
    assert "# Minimal-Model" in card
    assert "base_model: gpt2" in card
    assert "Evaluation in progress" in card


def test_infer_metadata_from_directory(tmp_path: Path) -> None:
    """Verify metadata inference from local directory files."""
    ckpt_dir = tmp_path / "checkpoint_awq"
    ckpt_dir.mkdir()

    # Create dummy config.json
    config_data = {
        "_name_or_path": "deepseek-ai/DeepSeek-Coder-6.7B",
        "quantization_config": {"quant_method": "awq"},
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")

    # Create dummy metrics.json
    metrics_data = {"HumanEval Pass@1": 0.72}
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics_data), encoding="utf-8")

    # Create dummy weight file
    weight_file = ckpt_dir / "model.safetensors"
    weight_file.write_bytes(b"0" * 1024 * 1024)  # 1 MB

    meta = infer_metadata_from_directory(ckpt_dir)
    assert meta.model_name == "checkpoint_awq"
    assert meta.base_model == "deepseek-ai/DeepSeek-Coder-6.7B"
    assert meta.compression_method == "AWQ"
    assert meta.benchmark_scores["HumanEval Pass@1"] == 0.72
    assert meta.compressed_size_gb is not None


def test_validate_checkpoint_errors(tmp_path: Path) -> None:
    """Verify validation errors for nonexistent, empty, or weightless directories."""
    pub = HFModelPublisher()

    # Non-existent
    with pytest.raises(FileNotFoundError):
        pub.validate_checkpoint(tmp_path / "does_not_exist")

    # Empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="is empty"):
        pub.validate_checkpoint(empty_dir)

    # Directory with only unrelated text file
    unrelated_dir = tmp_path / "unrelated"
    unrelated_dir.mkdir()
    (unrelated_dir / "notes.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain model weights"):
        pub.validate_checkpoint(unrelated_dir)


def test_publish_dry_run(tmp_path: Path) -> None:
    """Verify dry_run execution prepares README without calling external HF API."""
    ckpt_dir = tmp_path / "my_model"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.safetensors").write_bytes(b"fake-weights")
    (ckpt_dir / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")

    pub = HFModelPublisher(token="hf_mock_token")
    result = pub.publish(
        checkpoint_dir=ckpt_dir,
        repo_id="testuser/my-awesome-compressed-llama",
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.repo_id == "testuser/my-awesome-compressed-llama"
    assert result.repo_url == "https://huggingface.co/testuser/my-awesome-compressed-llama"
    assert (ckpt_dir / "README.md").exists()
    assert "README.md" in result.uploaded_files
    assert "model.safetensors" in result.uploaded_files


def test_publish_active_mock(tmp_path: Path) -> None:
    """Verify active upload workflow calls create_repo and upload_folder on HfApi."""
    ckpt_dir = tmp_path / "ready_model"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.safetensors").write_bytes(b"fake-weights")
    (ckpt_dir / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")

    with patch("huggingface_hub.HfApi") as mock_hf_api:
        api_instance = MagicMock()
        mock_hf_api.return_value = api_instance

        pub = HFModelPublisher(token="hf_test_123")
        res = pub.publish(
            checkpoint_dir=ckpt_dir,
            repo_id="org/compressed-model",
            private=True,
            commit_message="Initial ViPym upload",
            dry_run=False,
        )

        assert res.dry_run is False
        mock_hf_api.assert_called_once_with(token="hf_test_123")
        api_instance.create_repo.assert_called_once_with(
            repo_id="org/compressed-model", private=True, exist_ok=True
        )
        api_instance.upload_folder.assert_called_once()
        call_kwargs = api_instance.upload_folder.call_args[1]
        assert call_kwargs["repo_id"] == "org/compressed-model"
        assert call_kwargs["commit_message"] == "Initial ViPym upload"


def test_publish_missing_huggingface_hub(tmp_path: Path) -> None:
    """Verify clear error when huggingface_hub is not available in non-dry-run mode."""
    ckpt_dir = tmp_path / "model_dir"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.safetensors").write_bytes(b"weights")

    pub = HFModelPublisher()
    with patch.dict("sys.modules", {"huggingface_hub": None}):
        with pytest.raises(ImportError, match="huggingface_hub"):
            pub.publish(
                checkpoint_dir=ckpt_dir,
                repo_id="org/model",
                dry_run=False,
            )


def test_cli_publish_command_dry_run(tmp_path: Path) -> None:
    """Verify CLI publish invocation with --dry-run flag."""
    ckpt_dir = tmp_path / "cli_model"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.safetensors").write_bytes(b"data")
    (ckpt_dir / "config.json").write_text('{"model_type": "gpt2"}', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "publish",
            str(ckpt_dir),
            "--repo-id",
            "vipym-team/test-cli-model",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "ViPym Model Publisher" in result.stdout
    assert "DRY-RUN" in result.stdout
    assert "prepared successfully" in result.stdout
    assert "https://huggingface.co/vipym-team/test-cli-model" in result.stdout


def test_cli_publish_command_invalid_path(tmp_path: Path) -> None:
    """Verify CLI publish failure exit code on non-existent checkpoint path."""
    result = runner.invoke(
        app,
        [
            "publish",
            str(tmp_path / "non_existent"),
            "--repo-id",
            "vipym-team/invalid",
        ],
    )
    assert result.exit_code != 0
    assert "Publishing failed" in result.stdout
