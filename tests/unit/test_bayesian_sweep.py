"""Unit tests for Bayesian Multi-Objective Compression Sweeps."""

from pathlib import Path

from vipym.experiments.sweep import (
    BayesianSweepOptimizer,
    SweepGridConfig,
    SweepRunner,
)


def test_bayesian_optimizer_lifecycle() -> None:
    """Verify suggest/tell cycle, exploration/exploitation, and exhaustion in BayesianSweepOptimizer."""
    grid = {
        "quantization": ["awq", "gptq"],
        "bits": [4, 8],
        "kv_cache": ["fp16", "fp8_e4m3"],
    }
    # 2 * 2 * 2 = 8 total combinations
    opt = BayesianSweepOptimizer(grid=grid, seed=42)

    assert len(opt.all_combinations) == 8
    assert len(opt.unevaluated_combinations) == 8

    # Warmup suggestions
    s1 = opt.suggest_next()
    assert s1 is not None
    assert len(opt.unevaluated_combinations) == 7

    opt.tell(
        s1,
        {
            "quality_score": 0.85,
            "compression_ratio": 3.2,
            "peak_vram_gb": 16.0,
            "latency_p50_ms": 22.0,
        },
    )

    # Subsequent suggestions
    seen = [s1]
    for _ in range(7):
        s = opt.suggest_next()
        assert s is not None
        assert s not in seen
        seen.append(s)
        opt.tell(
            s,
            {
                "quality_score": 0.80,
                "compression_ratio": 4.0,
                "peak_vram_gb": 12.0,
                "latency_p50_ms": 18.0,
            },
        )

    # Exhausted
    assert opt.suggest_next() is None
    assert len(opt.unevaluated_combinations) == 0


def test_bayesian_optimizer_reproducibility() -> None:
    """Verify that identical seeds produce identical sequence of suggested points."""
    grid = {
        "bits": [2, 4, 8],
        "pruning_ratio": [0.0, 0.1, 0.25, 0.5],
    }
    opt1 = BayesianSweepOptimizer(grid=grid, seed=123)
    opt2 = BayesianSweepOptimizer(grid=grid, seed=123)

    p1_first = opt1.suggest_next()
    p2_first = opt2.suggest_next()
    assert p1_first == p2_first


def test_sweep_grid_config_yaml(tmp_path: Path) -> None:
    """Verify YAML configuration loading with strategy and n_trials."""
    yaml_content = """
sweep_id: bayes_demo_01
model_id: deepseek-ai/DeepSeek-Coder-6.7B
strategy: bayesian
n_trials: 6
seed: 99
grid:
  quantization: [awq, gptq]
  bits: [4, 8]
  kv_cache: [fp16, fp8_e4m3]
"""
    cfg_file = tmp_path / "sweep.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = SweepGridConfig.from_yaml(cfg_file)
    assert cfg.sweep_id == "bayes_demo_01"
    assert cfg.strategy == "bayesian"
    assert cfg.n_trials == 6
    assert cfg.seed == 99
    assert len(cfg.grid["bits"]) == 2


def test_sweep_runner_bayesian(tmp_path: Path) -> None:
    """Verify end-to-end execution of a Bayesian compression sweep."""
    cfg = SweepGridConfig(
        sweep_id="test_bayesian_run",
        model_id="meta-llama/Llama-3-8B",
        strategy="bayesian",
        n_trials=4,
        seed=42,
        grid={
            "quantization": ["awq", "gptq"],
            "bits": [4, 8],
            "kv_cache": ["fp16", "fp8_e4m3"],
        },
        artifacts_dir=str(tmp_path),
    )

    runner = SweepRunner(config=cfg, artifacts_dir=tmp_path)
    res = runner.run(resume=False)

    assert res.completed_points == 4
    assert len(res.all_points) == 4
    assert len(res.pareto_optimal_points) >= 1
    assert (tmp_path / "sweep_test_bayesian_run" / "sweep_report.md").exists()
    assert (tmp_path / "sweep_test_bayesian_run" / "pareto_summary.json").exists()


def test_sweep_runner_random_and_resumption(tmp_path: Path) -> None:
    """Verify random strategy execution and state resumption without re-evaluating."""
    cfg = SweepGridConfig(
        sweep_id="test_random_run",
        model_id="meta-llama/Llama-3-8B",
        strategy="random",
        n_trials=3,
        seed=10,
        grid={
            "bits": [4, 8],
            "kv_cache": ["fp16", "fp8_e4m3"],
        },
        artifacts_dir=str(tmp_path),
    )

    runner = SweepRunner(config=cfg, artifacts_dir=tmp_path)
    res1 = runner.run(resume=False)
    assert res1.completed_points == 3

    # Run again with resume=True
    runner_resume = SweepRunner(config=cfg, artifacts_dir=tmp_path)
    res2 = runner_resume.run(resume=True)
    assert res2.completed_points == 3
