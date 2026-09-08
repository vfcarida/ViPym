"""Unit tests for Branching DAG Execution Engine and Multi-Candidate Artifact Isolation."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelAdapter, ModelMetadata, PluginCapability
from vipym.pipelines.dag import DirectedAcyclicCompressionPipeline


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

    def save_pretrained(self, path: Path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "model.pt")


class DummyModelAdapter(ModelAdapter):
    def __init__(self, initial_model: nn.Module):
        self.initial_model = initial_model

    def get_capabilities(self) -> PluginCapability:
        return MagicMock()

    def inspect_metadata(self, model_id_or_path: str, revision: str = "main") -> ModelMetadata:
        return MagicMock()

    def load_for_compression(
        self, model_id_or_path: str, revision: str = "main", **kwargs
    ) -> nn.Module:
        p = Path(model_id_or_path)
        if (p / "model.pt").exists():
            model = DummyModel()
            model.load_state_dict(torch.load(p / "model.pt", weights_only=True))
            return model
        # Otherwise return deepcopy of initial model
        import copy

        return copy.deepcopy(self.initial_model)

    def get_tokenizer(self, model_id_or_path: str, revision: str = "main"):
        return None


class AddBiasMethod(CompressionMethod):
    """Simple test compression method that adds a constant offset to all weights."""

    def __init__(self, name: str, offset: float):
        self._name = name
        self.offset = offset

    @property
    def name(self) -> str:
        return self._name

    def get_capabilities(self) -> PluginCapability:
        return MagicMock()

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        pass

    def compress(
        self,
        model: nn.Module,
        tokenizer,
        calibration_data=None,
        output_dir=None,
        **kwargs,
    ) -> CompressionArtifact:
        out = Path(output_dir or "./test_out")
        out.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(self.offset)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out)

        return CompressionArtifact(
            output_path=out,
            format="test-format",
            compressed_size_bytes=1000,
            applied_methods=[self.name],
            metadata={"offset": self.offset},
        )


def test_branching_dag_topological_order_and_leaves():
    dag = DirectedAcyclicCompressionPipeline()

    m_root = AddBiasMethod("prune", 1.0)
    m_b1 = AddBiasMethod("awq_candidate", 10.0)
    m_b2 = AddBiasMethod("autoround_candidate", 20.0)

    dag.add_stage("prune_base", m_root, dependencies=[])
    dag.add_stage("branch_awq", m_b1, dependencies=["prune_base"])
    dag.add_stage("branch_autoround", m_b2, dependencies=["prune_base"])

    order = dag.get_topological_order()
    assert order[0] == "prune_base"
    assert set(order[1:]) == {"branch_awq", "branch_autoround"}


def test_branching_dag_execution_with_artifact_isolation():
    """Verify that parallel branches do not mutate each other's weights."""
    dag = DirectedAcyclicCompressionPipeline()

    m_root = AddBiasMethod("prune", 1.0)
    m_b1 = AddBiasMethod("branch_1", 10.0)
    m_b2 = AddBiasMethod("branch_2", 20.0)

    dag.add_stage("prune_base", m_root, dependencies=[])
    dag.add_stage("branch_1", m_b1, dependencies=["prune_base"])
    dag.add_stage("branch_2", m_b2, dependencies=["prune_base"])

    torch.manual_seed(42)
    base_model = DummyModel()
    orig_fc1_weight = base_model.fc1.weight.clone()

    adapter = DummyModelAdapter(base_model)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_root = Path(tmpdir) / "dag_output"
        primary_artifact = dag.execute(
            model_adapter=adapter,
            model_id="test-base-model",
            output_dir=out_root,
        )

        assert isinstance(primary_artifact, CompressionArtifact)
        assert "branches" in primary_artifact.metadata

        branches = dag.get_branch_artifacts()
        assert len(branches) == 2
        assert "branch_1" in branches
        assert "branch_2" in branches

        # Verify artifacts exist on disk
        art_b1 = branches["branch_1"]
        art_b2 = branches["branch_2"]
        assert (art_b1.output_path / "model.pt").exists()
        assert (art_b2.output_path / "model.pt").exists()

        # Load weights from both branches to verify isolation:
        # Branch 1 should have: orig + 1.0 (prune) + 10.0 (branch_1) = orig + 11.0
        # Branch 2 should have: orig + 1.0 (prune) + 20.0 (branch_2) = orig + 21.0
        # If there were leakage/shared mutation, Branch 2 would have orig + 1.0 + 10.0 + 20.0 = orig + 31.0
        w_b1 = torch.load(art_b1.output_path / "model.pt", weights_only=True)["fc1.weight"]
        w_b2 = torch.load(art_b2.output_path / "model.pt", weights_only=True)["fc1.weight"]

        assert torch.allclose(w_b1, orig_fc1_weight + 11.0, atol=1e-5)
        assert torch.allclose(w_b2, orig_fc1_weight + 21.0, atol=1e-5)
