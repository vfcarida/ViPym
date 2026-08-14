"""ViPym Command-Line Interface."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vipym.__version__ import __version__
from vipym.cli.doctor import run_doctor_checks
from vipym.compression.registry import CompressionRegistry
from vipym.config.schema import ViPymExperimentConfig
from vipym.core.logger import get_logger
from vipym.evaluation.registry import EvaluationRegistry
from vipym.experiments.runner import ResumableExperimentRunner
from vipym.models.registry import ModelRegistry

app = typer.Typer(
    name="vipym",
    help="ViPym: Shrinking LLMs, Preserving Intelligence — Modular LLM Compression & Evaluation Framework",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


@app.callback(invoke_without_command=True)
def version_callback(
    version: bool | None = typer.Option(
        None, "--version", "-v", help="Show ViPym version and exit"
    ),
) -> None:
    if version:
        console.print(
            f"[bold cyan]ViPym[/bold cyan] version [bold green]{__version__}[/bold green]"
        )
        raise typer.Exit()


@app.command("doctor")
def doctor_cmd() -> None:
    """Validate system readiness across Python, CUDA, GPU, Docker, VRAM, and Disk."""
    run_doctor_checks()


@app.command("validate")
def validate_config(
    config_path: Path = typer.Option(
        ..., "--config", "-c", help="Path to experiment YAML configuration"
    ),
) -> None:
    """Validate an experiment YAML configuration file against Pydantic schema."""
    try:
        cfg = ViPymExperimentConfig.from_yaml(config_path)
        console.print(
            f"[bold green][VALID] Configuration is valid:[/bold green] [cyan]{cfg.experiment_id}[/cyan]"
        )
        console.print(f"Model: [magenta]{cfg.model.id}[/magenta]")
        console.print(f"Stages: [yellow]{len(cfg.compression_pipeline)}[/yellow]")
        console.print(f"Suites: [blue]{cfg.evaluation.suites}[/blue]")
    except Exception as e:
        console.print(f"[bold red][ERROR] Configuration Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command("run")
def run_experiment(
    config_path: Path = typer.Option(
        ..., "--config", "-c", help="Path to experiment YAML configuration"
    ),
    artifacts_dir: Path = typer.Option(
        Path("./artifacts"), "--artifacts-dir", "-a", help="Directory to save artifacts"
    ),
    no_resume: bool = typer.Option(
        False, "--no-resume", help="Force restart from scratch ignoring checkpoints"
    ),
) -> None:
    """Run an end-to-end compression, benchmark evaluation, and reporting experiment."""
    console.print(f"[bold green]Starting ViPym Experiment:[/bold green] {config_path}")
    cfg = ViPymExperimentConfig.from_yaml(config_path)
    runner = ResumableExperimentRunner(config=cfg, artifacts_dir=artifacts_dir)
    res = runner.run(resume=not no_resume)
    console.print(
        f"\n[bold green][SUCCESS] Experiment [{res.experiment_id}] Completed Successfully![/bold green]"
    )
    console.print(f"Manifest ID: [cyan]{res.manifest_id}[/cyan]")
    console.print(f"Total Duration: [yellow]{res.total_duration_sec:.2f}s[/yellow]")
    console.print(f"Total Est. Cost: [yellow]${res.total_cost_usd:.4f}[/yellow]")
    console.print(f"Report Dashboard: [magenta]{res.generated_report_files.get('html')}[/magenta]")


@app.command("baseline")
def baseline_cmd(
    model_id: str = typer.Option(..., "--model", "-m", help="Target model ID"),
    suite: str = typer.Option("humaneval", "--suite", "-s", help="Evaluation suite"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of tasks"),
) -> None:
    """Establish and evaluate an uncompressed baseline."""
    from vipym.evaluation.runner import BenchmarkRunner
    from vipym.inference.registry import InferenceRegistry

    console.print(
        f"Establishing baseline for [cyan]{model_id}[/cyan] on suite [magenta]{suite}[/magenta]"
    )
    backend = InferenceRegistry.get("vllm")
    backend.start(model_id)
    runner = BenchmarkRunner()
    res = runner.run_suite(suite, backend, task_limit=limit)
    backend.stop()
    console.print(
        f"Baseline Score for [magenta]{res.suite_name}[/magenta]: Pass@1 = [bold green]{res.pass_at_1 * 100:.1f}%[/bold green]"
    )


@app.command("compress")
def compress_cmd(
    model_id: str = typer.Option(..., "--model", "-m", help="Model to compress"),
    method: str = typer.Option("awq", "--method", help="Compression algorithm"),
    output_dir: Path = typer.Option(Path("./compressed_output"), "--output-dir", "-o"),
) -> None:
    """Execute a standalone compression algorithm on a model."""
    try:
        adapter = ModelRegistry.get(model_id)
    except Exception:
        adapter = ModelRegistry.get("hf")
    compressor = CompressionRegistry.get(method)
    console.print(
        f"Compressing [cyan]{model_id}[/cyan] using [magenta]{method}[/magenta] -> {output_dir}"
    )
    model = adapter.load_for_compression(model_id)
    tokenizer = adapter.get_tokenizer(model_id)
    art = compressor.compress(model, tokenizer, output_dir=output_dir)
    console.print(
        f"[bold green][SUCCESS] Compression completed:[/bold green] format={art.format}, size={art.compressed_size_bytes} bytes"
    )


@app.command("evaluate")
def evaluate_cmd(
    model_path_or_id: str = typer.Option(..., "--model", "-m", help="Model path or HuggingFace ID"),
    suite: str = typer.Option("humaneval", "--suite", "-s", help="Evaluation suite name"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Cap number of tasks"),
) -> None:
    """Evaluate a model on a benchmark suite without compression."""
    from vipym.evaluation.runner import BenchmarkRunner
    from vipym.inference.registry import InferenceRegistry

    console.print(
        f"Evaluating model [cyan]{model_path_or_id}[/cyan] on suite [magenta]{suite}[/magenta]"
    )
    backend = InferenceRegistry.get("vllm")
    backend.start(model_path_or_id)
    runner = BenchmarkRunner()
    res = runner.run_suite(suite, backend, task_limit=limit)
    backend.stop()
    console.print(
        f"Results for [magenta]{res.suite_name}[/magenta]: Pass@1 = [bold green]{res.pass_at_1 * 100:.1f}%[/bold green]"
    )


@app.command("benchmark")
def benchmark_cmd(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to evaluation config YAML"),
) -> None:
    """Run benchmark suites defined in an evaluation config."""
    run_experiment(config_path=config_path)


@app.command("analyze")
def analyze_cmd(
    experiment_dir: Path = typer.Option(
        Path("./artifacts"), "--dir", "-d", help="Path to experiment artifacts directory"
    ),
) -> None:
    """Perform multi-objective Pareto analysis on results.json."""
    from vipym.analysis.pareto import ParetoFrontierOptimizer, ParetoPoint

    results_file = experiment_dir / "results.json"
    if not results_file.exists():
        console.print(f"[bold red]results.json not found in {experiment_dir}[/bold red]")
        raise typer.Exit(code=1)

    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)
    points = [ParetoPoint(**d) for d in data]

    opt = ParetoFrontierOptimizer()
    pareto_pts = opt.compute_pareto_frontier(points)

    table = Table(title="Pareto Optimal Configurations", header_style="bold green")
    table.add_column("Configuration")
    table.add_column("Pass@1 (%)", justify="right")
    table.add_column("Peak VRAM (GB)", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Cost ($)", justify="right")

    for p in pareto_pts:
        table.add_row(
            p.configuration_name,
            f"{p.quality_score * 100:.1f}%",
            f"{p.peak_vram_gb:.1f}",
            f"{p.latency_p50_ms:.1f}",
            f"${p.cost_usd:.2f}",
        )
    console.print(table)


@app.command("report")
def report_cmd(
    experiment_dir: Path = typer.Option(
        Path("./artifacts"), "--dir", "-d", help="Experiment directory containing results.json"
    ),
    format_type: str = typer.Option(
        "markdown", "--format", "-f", help="Output format: markdown, html, latex"
    ),
) -> None:
    """Generate or print a report in the specified format."""
    rep_dir = experiment_dir / "reports"
    if format_type == "markdown":
        p = rep_dir / "report.md"
    elif format_type == "html":
        p = rep_dir / "dashboard.html"
    elif format_type == "latex":
        p = rep_dir / "table.tex"
    else:
        p = rep_dir / "report.md"

    if p.exists():
        console.print(p.read_text(encoding="utf-8"))
    else:
        console.print(f"[bold red]Report file not found: {p}[/bold red]")


@app.command("compare")
def compare_cmd(
    baseline_dir: Path = typer.Option(..., "--baseline", "-b"),
    experiment_dir: Path = typer.Option(..., "--candidate", "-c"),
) -> None:
    """Compare candidate experiment results against baseline."""
    console.print(f"Comparing candidate {experiment_dir} vs baseline {baseline_dir}")
    analyze_cmd(experiment_dir=experiment_dir)


@app.command("list-models")
def list_models() -> None:
    """List all registered model adapters."""
    table = Table(title="Registered Model Adapters", header_style="bold cyan")
    table.add_column("Adapter Name")
    table.add_column("Class")
    for name, cls in ModelRegistry.list_adapters().items():
        table.add_row(name, cls.__name__)
    console.print(table)


@app.command("list-compressors")
def list_compressors() -> None:
    """List all registered compression methods."""
    table = Table(title="Registered Compression Methods", header_style="bold magenta")
    table.add_column("Method Name")
    table.add_column("Class")
    for name, cls in CompressionRegistry.list_methods().items():
        table.add_row(name, getattr(cls, "__name__", "DynamicPlugin"))
    console.print(table)


@app.command("list-evaluators")
def list_evaluators() -> None:
    """List all registered evaluation suites."""
    table = Table(title="Registered Evaluation Benchmark Suites", header_style="bold yellow")
    table.add_column("Suite Identifier")
    table.add_column("Class")
    for name, cls in EvaluationRegistry.list_suites().items():
        table.add_row(name, cls.__name__)
    console.print(table)


@app.command("inspect-model")
def inspect_model(
    model_id: str = typer.Option(..., "--model", "-m", help="Model ID or local path"),
) -> None:
    """Inspect model topology, parameter count, and MoE active parameters."""
    try:
        adapter = ModelRegistry.get(model_id)
    except Exception:
        adapter = ModelRegistry.get("hf")
    meta = adapter.inspect_metadata(model_id)
    console.print(f"[bold cyan]Model Metadata:[/bold cyan] {meta.model_id}")
    console.print(f"Total Parameters: [green]{meta.total_parameters / 1e9:.2f}B[/green]")
    console.print(f"Active Parameters: [green]{meta.active_parameters / 1e9:.2f}B[/green]")
    console.print(f"Architecture: [yellow]{meta.architecture_type}[/yellow]")
    if meta.num_experts:
        console.print(
            f"MoE Experts: [magenta]{meta.num_experts}[/magenta] (Active: {meta.num_selected_experts})"
        )


if __name__ == "__main__":
    app()
