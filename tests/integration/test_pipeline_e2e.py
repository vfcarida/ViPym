"""End-to-end integration test for full compression pipeline: DAG prune -> quantize -> evaluate on HumanEval.

Verifies:
1. DAG pipeline with multiple stages (Wanda Pruning -> GPTQ Quantization)
2. Model executes DAG and saves composite compressed artifact
3. Compressed model is loaded into an InferenceBackend
4. BenchmarkRunner evaluates compressed model on 5 HumanEval tasks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from vipym.compression.methods.gptq import GPTQCompressionMethod
from vipym.compression.methods.pruning import WandaPruningMethod
from vipym.evaluation.runner import BenchmarkRunner
from vipym.evaluation.sandbox.docker_sandbox import SandboxedCodeRunner
from vipym.evaluation.sandbox.security_profile import SandboxSecurityConfig
from vipym.inference.hf_engine import HuggingFaceInferenceBackend
from vipym.interfaces.model import ModelAdapter, ModelMetadata
from vipym.pipelines.dag import DirectedAcyclicCompressionPipeline


def create_small_gpt2_model(output_dir: Path) -> tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    """Create and save a small GPT-2 model and tokenizer for fast pipeline tests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=512,
        n_embd=128,
        n_layer=4,
        n_head=4,
        bos_token_id=tokenizer.bos_token_id or 0,
        eos_token_id=tokenizer.eos_token_id or 0,
    )
    torch.manual_seed(42)
    model = GPT2LMHeadModel(config)
    model.eval()

    model.save_pretrained(output_dir)

    return model, tokenizer


class LocalModelAdapter(ModelAdapter):
    """Adapter wrapping the local temporary GPT-2 model."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir

    def get_capabilities(self):
        return MagicMock()

    def inspect_metadata(self, model_id_or_path: str, revision: str = "main") -> ModelMetadata:
        return ModelMetadata(
            model_id=str(self.model_dir),
            revision=revision,
            total_parameters=1_000_000,
            active_parameters=1_000_000,
            architecture_type="dense",
            native_dtypes=["fp32"],
            context_window=512,
            num_layers=4,
            hidden_size=128,
            num_attention_heads=4,
        )

    def load_for_compression(
        self, model_id_or_path: str, revision: str = "main", **kwargs
    ) -> torch.nn.Module:
        return GPT2LMHeadModel.from_pretrained(self.model_dir)

    def get_tokenizer(self, model_id_or_path: str, revision: str = "main") -> Any:
        return GPT2TokenizerFast.from_pretrained(self.model_dir)


@pytest.mark.integration
class TestPipelineE2E:
    def test_full_dag_prune_quantize_and_evaluate_e2e(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """End-to-end test: Prune -> Quantize -> Evaluate on 5 HumanEval tasks."""
        monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")
        source_dir = tmp_path / "pipeline_gpt2_source"
        model, tokenizer = create_small_gpt2_model(source_dir)
        adapter = LocalModelAdapter(source_dir)

        # 1. Build DAG Pipeline: Stage 1 = Wanda Pruning, Stage 2 = GPTQ Quantization
        dag = DirectedAcyclicCompressionPipeline()

        stage1_wanda = WandaPruningMethod(sparsity=0.25, prune_type="unstructured")
        stage2_gptq = GPTQCompressionMethod(bits=4, group_size=64, sym=True)

        dag.add_stage(
            stage_id="prune_stage",
            method=stage1_wanda,
        )
        dag.add_stage(
            stage_id="quant_stage",
            method=stage2_gptq,
            dependencies=["prune_stage"],
        )

        # 2. Execute DAG Compression Pipeline
        pipeline_out_dir = tmp_path / "dag_output"
        artifact = dag.execute(
            model_adapter=adapter,
            model_id=str(source_dir),
            output_dir=pipeline_out_dir,
        )

        assert artifact is not None
        assert artifact.output_path.exists()
        assert len(artifact.applied_methods) == 2
        assert "prune_stage" in [node.stage_id for node in dag.nodes.values()]

        # 3. Start Inference Backend with Compressed Model
        backend = HuggingFaceInferenceBackend()
        backend.start(
            model_path_or_id=artifact.output_path,
            max_model_len=512,
        )

        # 4. Run Evaluation on 5 HumanEval Benchmark Tasks
        sec_cfg = SandboxSecurityConfig(allow_unsafe_execution=True)
        sandbox = SandboxedCodeRunner(config=sec_cfg)
        eval_runner = BenchmarkRunner(sandbox_runner=sandbox)

        eval_result = eval_runner.run_suite(
            suite_name="humaneval",
            backend=backend,
            temperature=0.0,
            max_new_tokens=32,
            task_limit=5,
        )

        # 5. Verify Evaluation Results
        assert eval_result is not None
        assert eval_result.suite_name.lower() == "humaneval"
        assert eval_result.total_tasks == 5
        assert len(eval_result.task_results) == 5
        assert 0.0 <= eval_result.pass_at_1 <= 1.0
        assert "throughput_tok_s" in eval_result.summary_metrics
        assert "latency_p50_ms" in eval_result.summary_metrics
