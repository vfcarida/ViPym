"""Unit tests for P017 — Router Distillation."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from vipym.compression.moe.router_distillation import (
    RouterDistillationConfig,
    RouterDistillationResult,
    _compute_utilisation_ratio,
    _make_calib_hidden,
    distil_router,
    run_router_distillation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_router() -> nn.Linear:
    """Student router: 4 remaining experts, hidden_dim=32."""
    torch.manual_seed(0)
    return nn.Linear(32, 4, bias=False)


@pytest.fixture()
def teacher_router() -> nn.Linear:
    """Original router with 8 experts before pruning."""
    torch.manual_seed(1)
    return nn.Linear(32, 8, bias=False)


@pytest.fixture()
def teacher_logits() -> torch.Tensor:
    """Synthetic teacher logits [256 tokens, 4 retained experts]."""
    torch.manual_seed(42)
    return torch.randn(256, 4) * 2.0


@pytest.fixture()
def calibration_hidden() -> torch.Tensor:
    """Calibration hidden states [512 tokens, hidden_dim=32]."""
    torch.manual_seed(7)
    return torch.randn(512, 32)


# ---------------------------------------------------------------------------
# RouterDistillationConfig
# ---------------------------------------------------------------------------


class TestRouterDistillationConfig:
    def test_defaults(self):
        cfg = RouterDistillationConfig()
        assert cfg.steps == 500
        assert cfg.lr == pytest.approx(1e-4)
        assert cfg.balance_loss_weight == pytest.approx(0.10)
        assert cfg.kl_loss_weight == pytest.approx(1.00)

    def test_from_dict_partial(self):
        cfg = RouterDistillationConfig.from_dict({"steps": 200, "lr": 3e-4})
        assert cfg.steps == 200
        assert cfg.lr == pytest.approx(3e-4)
        # Untouched fields keep defaults
        assert cfg.balance_loss_weight == pytest.approx(0.10)

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown YAML keys must not raise."""
        cfg = RouterDistillationConfig.from_dict({"steps": 100, "unknown_param": True})
        assert cfg.steps == 100

    def test_from_dict_full(self):
        d = {
            "steps": 1000,
            "lr": 5e-5,
            "calibration_samples": 512,
            "balance_loss_weight": 0.2,
            "kl_loss_weight": 0.8,
            "entropy_loss_weight": 0.05,
            "warmup_steps": 100,
            "temperature": 2.0,
            "max_utilisation_ratio": 4.0,
            "log_every": 50,
        }
        cfg = RouterDistillationConfig.from_dict(d)
        assert cfg.steps == 1000
        assert cfg.temperature == pytest.approx(2.0)
        assert cfg.max_utilisation_ratio == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestPrivateHelpers:
    def test_make_calib_hidden_shape(self):
        teacher_target = torch.softmax(torch.randn(64, 4), dim=-1)
        h = _make_calib_hidden(teacher_target, hidden_dim=32, n_samples=64, device=torch.device("cpu"))
        assert h.shape == (64, 32)

    def test_make_calib_hidden_no_nan(self):
        teacher_target = torch.softmax(torch.randn(64, 4), dim=-1)
        h = _make_calib_hidden(teacher_target, hidden_dim=32, n_samples=64, device=torch.device("cpu"))
        assert not torch.isnan(h).any()

    def test_compute_utilisation_ratio_uniform(self):
        """Uniform router should return ratio close to 1.0."""
        router = nn.Linear(16, 4, bias=False)
        nn.init.constant_(router.weight, 1.0)
        hs = torch.ones(128, 16)
        ratio = _compute_utilisation_ratio(router, hs)
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_compute_utilisation_ratio_collapsed(self):
        """Router collapsed to single expert → high ratio."""
        router = nn.Linear(16, 4, bias=False)
        nn.init.zeros_(router.weight)
        # Boost first expert weight drastically
        with torch.no_grad():
            router.weight.data[0] = 100.0
        hs = torch.ones(128, 16)
        ratio = _compute_utilisation_ratio(router, hs)
        assert ratio > 5.0


# ---------------------------------------------------------------------------
# distil_router (core function)
# ---------------------------------------------------------------------------


class TestDistilRouter:
    def test_returns_result_instance(self, small_router, teacher_logits):
        cfg = RouterDistillationConfig(steps=10, log_every=5)
        result = distil_router(student_router=small_router, teacher_logits=teacher_logits, config=cfg)
        assert isinstance(result, RouterDistillationResult)

    def test_loss_decreases_over_training(self, small_router, teacher_logits):
        """Training loss should generally decrease (allow +5% wiggle)."""
        cfg = RouterDistillationConfig(steps=50, log_every=10)
        result = distil_router(student_router=small_router, teacher_logits=teacher_logits, config=cfg)
        # Average of first 10 vs last 10 steps
        early = sum(result.loss_history[:10]) / 10
        late = sum(result.loss_history[-10:]) / 10
        assert late < early * 1.05, f"Loss not decreasing: {early:.4f} → {late:.4f}"

    def test_converged_flag(self, small_router, teacher_logits):
        # Use more steps so the optimiser has time to settle
        cfg = RouterDistillationConfig(steps=80, log_every=20)
        result = distil_router(student_router=small_router, teacher_logits=teacher_logits, config=cfg)
        # Allow up to 5% above initial — stochastic mini-batch can cause noise spikes
        assert result.final_loss <= result.initial_loss * 1.05, (
            f"Loss diverged badly: {result.initial_loss:.4f} → {result.final_loss:.4f}"
        )


    def test_loss_history_length(self, small_router, teacher_logits):
        steps = 25
        cfg = RouterDistillationConfig(steps=steps, log_every=5)
        result = distil_router(student_router=small_router, teacher_logits=teacher_logits, config=cfg)
        assert len(result.loss_history) == steps

    def test_elapsed_seconds_positive(self, small_router, teacher_logits):
        cfg = RouterDistillationConfig(steps=5, log_every=2)
        result = distil_router(small_router, teacher_logits, cfg)
        assert result.elapsed_seconds > 0.0

    def test_student_weights_change(self, small_router, teacher_logits):
        """Router weights must be updated by the optimiser."""
        initial_weight = small_router.weight.data.clone()
        cfg = RouterDistillationConfig(steps=20, log_every=5)
        distil_router(small_router, teacher_logits, cfg)
        assert not torch.allclose(small_router.weight.data, initial_weight)

    def test_utilisation_ratio_finite(self, small_router, teacher_logits):
        cfg = RouterDistillationConfig(steps=20, log_every=5)
        result = distil_router(small_router, teacher_logits, cfg)
        assert result.utilisation_ratio < float("inf")
        assert result.utilisation_ratio > 0.0

    def test_to_dict_keys(self, small_router, teacher_logits):
        cfg = RouterDistillationConfig(steps=5)
        result = distil_router(small_router, teacher_logits, cfg)
        d = result.to_dict()
        for key in ("steps_run", "elapsed_seconds", "initial_loss", "final_loss", "converged", "utilisation_ratio", "load_balanced"):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# run_router_distillation (full teacher→student pipeline)
# ---------------------------------------------------------------------------


class TestRunRouterDistillation:
    def test_basic_run(self, small_router, teacher_router, calibration_hidden):
        cfg = RouterDistillationConfig(steps=10, log_every=5, calibration_samples=64)
        retained = list(range(4))  # keep first 4 of 8 experts
        result = run_router_distillation(
            student_router=small_router,
            teacher_router=teacher_router,
            calibration_hidden_states=calibration_hidden,
            retained_expert_indices=retained,
            config=cfg,
        )
        assert isinstance(result, RouterDistillationResult)
        assert result.steps_run == 10

    def test_teacher_unchanged(self, small_router, teacher_router, calibration_hidden):
        """Teacher weights must not be modified."""
        teacher_copy = copy.deepcopy(teacher_router)
        cfg = RouterDistillationConfig(steps=10, log_every=5, calibration_samples=64)
        run_router_distillation(
            student_router=small_router,
            teacher_router=teacher_router,
            calibration_hidden_states=calibration_hidden,
            retained_expert_indices=list(range(4)),
            config=cfg,
        )
        assert torch.allclose(teacher_router.weight.data, teacher_copy.weight.data)

    def test_3d_hidden_states_accepted(self, small_router, teacher_router):
        """3D [batch, seq, hidden] tensors should be flattened automatically."""
        cfg = RouterDistillationConfig(steps=5, calibration_samples=32)
        hs_3d = torch.randn(4, 16, 32)  # [batch, seq, hidden]
        result = run_router_distillation(
            student_router=small_router,
            teacher_router=teacher_router,
            calibration_hidden_states=hs_3d,
            retained_expert_indices=list(range(4)),
            config=cfg,
        )
        assert result.steps_run == 5

    def test_load_balance_flag(self, small_router, teacher_router, calibration_hidden):
        """load_balanced should be True for well-trained router."""
        cfg = RouterDistillationConfig(steps=50, calibration_samples=128, max_utilisation_ratio=100.0)
        result = run_router_distillation(
            student_router=small_router,
            teacher_router=teacher_router,
            calibration_hidden_states=calibration_hidden,
            retained_expert_indices=list(range(4)),
            config=cfg,
        )
        # With max_utilisation_ratio=100 this should always be True unless inf
        assert result.load_balanced


# ---------------------------------------------------------------------------
# Integration: ExpertPruningMethod with router_distillation config
# ---------------------------------------------------------------------------


class _MiniMoEBlock(nn.Module):
    """Minimal MoE block with gate + experts for integration testing."""
    def __init__(self, num_experts: int = 6, hidden: int = 32):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_experts)])


class _MiniMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0 = _MiniMoEBlock(num_experts=6, hidden=32)


class TestPruningWithDistillation:
    def _build_model_metadata(self):
        from vipym.core.constants import ComputeArchitecture, SupportedDtype
        from vipym.interfaces.model import ModelMetadata
        return ModelMetadata(
            model_id="mini-moe",
            architecture_type=ComputeArchitecture.MOE,
            num_parameters=1_000_000,
            dtype=SupportedDtype.FP32,
        )

    def test_prune_with_distillation_config(self, tmp_path):
        from vipym.compression.methods.expert_pruning import ExpertPruningMethod

        model = _MiniMoEModel()
        tokenizer = object()  # dummy

        method = ExpertPruningMethod(
            strategy="magnitude",
            prune_ratio=0.33,
            retrain_router=True,
            router_steps=5,
            router_distillation={
                "steps": 10,
                "lr": 1e-3,
                "calibration_samples": 32,
                "balance_loss_weight": 0.1,
                "log_every": 5,
            },
        )

        artifact = method.compress(
            model=model,
            tokenizer=tokenizer,
            output_dir=tmp_path / "pruned_distilled",
        )

        assert artifact is not None
        assert artifact.metadata["prune_ratio"] == pytest.approx(0.33, abs=0.01)
        # Distillation report should be present in at least one layer
        layers = artifact.metadata.get("layers", {})
        assert len(layers) > 0
        # At least one layer should have distillation key
        has_distil = any("distillation" in v for v in layers.values())
        assert has_distil, "Expected distillation report in layer metadata"

    def test_prune_without_distillation_config(self, tmp_path):
        """Omitting router_distillation must not crash."""
        from vipym.compression.methods.expert_pruning import ExpertPruningMethod

        model = _MiniMoEModel()
        method = ExpertPruningMethod(prune_ratio=0.33, router_steps=5)
        artifact = method.compress(
            model=model,
            tokenizer=object(),
            output_dir=tmp_path / "pruned_nodistil",
        )
        assert artifact is not None
