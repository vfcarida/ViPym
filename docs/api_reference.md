# ViPym Python API Reference

Comprehensive reference for core classes, pipelines, evaluation runners, and telemetry modules in the `vipym` package.

---

## 1. Pipelines & Orchestration

### `vipym.pipelines.dag.DirectedAcyclicCompressionPipeline`
Topological DAG execution engine implementing Kahn's algorithm with cycle detection.
- **`add_stage(stage_id: str, method: CompressionMethod, dependencies: list[str] | None, parameters: dict[str, Any] | None) -> Self`**: Registers a compression or transform node in the DAG.
- **`get_topological_order() -> list[str]`**: Computes the sequential execution order. Raises `InvalidPipelineDAGError` if cycles exist.
- **`execute(model_adapter: ModelAdapter, model_id: str, output_dir: Path) -> CompressionArtifact`**: Executes all stages topologically with automated GPU memory cleanup.

### `vipym.experiments.runner.ResumableExperimentRunner`
Stateful orchestrator driving the 12-state FSM from baseline evaluation to report generation.
- **`run(resume: bool = True) -> ExperimentRunSummary`**: Executes the experiment pipeline with automatic recovery from `manifest.json` checkpoints.

---

## 2. Calibration & Data Management

### `vipym.data.calibration.CalibrationDatasetManager`
Manages code calibration datasets with automated benchmark contamination purging.
- **`get_calibration_corpus() -> list[str]`**: Loads raw code corpora and filters out prompts overlapping with benchmark suites (HumanEval, MBPP).
- **`tokenize_and_chunk(corpus: list[str], tokenizer: Any, sequence_length: int, max_samples: int) -> list[list[int]]`**: Prepares uniform tokenized tensors for AWQ, GPTQ, and SmoothQuant.

---

## 3. Evaluation & Benchmarking

### `vipym.evaluation.runner.BenchmarkRunner`
Concurrent benchmark evaluation engine with real-time inference telemetry profiling.
- **`run_suite(suite_name: str, backend: InferenceBackend, temperature: float, top_p: float, max_new_tokens: int, task_limit: int | None) -> EvaluationSuiteResult`**: Executes benchmark tasks concurrently via worker pools, recording latency (p50/p95), TTFT, ITL, and peak VRAM.

---

## 4. Multi-Objective Analysis & Statistics

### `vipym.analysis.pareto.ParetoFrontierOptimizer`
Computes the exact non-dominated Pareto frontier across Quality, Latency, VRAM, and Cost.
- **`compute_pareto_frontier(points: list[ParetoPoint], min_quality: float | None, max_cost: float | None, ...) -> list[ParetoPoint]`**: Returns the global non-dominated candidate set with optional constraint filtering.
- **`fast_non_dominated_sort(points: list[ParetoPoint]) -> list[list[ParetoPoint]]`**: Performs fast NSGA-II non-dominated sorting partitioning points into successive Pareto fronts.
- **`find_closest_to_utopia(points: list[ParetoPoint]) -> ParetoPoint | None`**: Identifies candidate with minimum normalized Euclidean distance to ideal Utopia point.

### `vipym.analysis.statistics.StatisticalAnalyzer`
Performs bootstrap confidence interval estimation and non-parametric hypothesis testing.
- **`bootstrap_confidence_interval(samples: list[float], num_resamples: int = 2000, confidence_level: float = 0.95) -> tuple[float, float, float]`**: Returns `(mean, ci_low, ci_high)`.
- **`evaluate_significance(baseline_scores: list[float], compressed_scores: list[float]) -> StatisticalSignificanceReport`**: Computes Mann-Whitney U test p-values and determines statistical equivalence.

### `vipym.analysis.comparator.ExperimentComparator`
Compares multiple compression runs, generating Pareto diffs and annual enterprise ROI models.
- **`generate_html_report(output_path: Path | str) -> Path`**: Produces a standalone comparison HTML dashboard.
- **`format_rich_table() -> Table`**: Generates a Rich terminal matrix of comparative results.

---

## 5. Grid Sweeps & Multi-Experiment Exploration

### `vipym.experiments.sweep.SweepRunner`
Multi-dimensional parameter sweep orchestrator with Cartesian product expansion, automated checkpointing, and Pareto front discovery.
- **`run(resume: bool = True) -> SweepResult`**: Iterates through the hyperparameter grid, resuming from `state.json` if interrupted, and computes the non-dominated Pareto frontier over evaluated combinations.
- **`generate_configurations() -> list[dict[str, Any]]`**: Expands nested dot-notation grid specifications into concrete configuration dictionaries.

### `vipym.experiments.sweep.SweepGridConfig`
Pydantic model defining the sweep space:
- `base_recipe`: Path to base recipe YAML.
- `grid`: Mapping of parameter paths to candidate value arrays (e.g., `stages.0.parameters.bits: [4, 8]`).
- `objectives`: Evaluation metrics to optimize (`quality`, `latency_ms`, `vram_gb`, `cost_per_million`).
- `checkpoint_interval`: Run state persistence frequency.

---

## 6. ViPym Studio & Real-Time Streaming

### `vipym.studio.server.StudioASGIServer`
High-concurrency ASGI server engine built on FastAPI and Uvicorn, replacing legacy synchronous servers with asynchronous non-blocking event loops.
- **`serve_forever()`**: Starts the Uvicorn ASGI server.
- **`shutdown()`**: Gracefully terminates the running server and active WebSocket connections.

### `vipym.studio.app.create_studio_app(artifacts_dir, read_only, api_token, max_requests_per_minute)`
Factory producing the FastAPI application equipped with:
- **`GET /api/dag/graph`**: Topological DAG node and edge visualizer payload.
- **`GET /api/moe/matrix`**: Expert co-activation correlation matrix for MoE architectures.
- **`POST /api/inference/generate`**: Model generation supporting Server-Sent Events (`text/event-stream`) token typewriter streaming.
- **`GET /api/reports/{id}/export`**: Multi-format report download (`markdown`, `latex`, `json`).
- **`WebSocket /ws/progress`**: Real-time push broadcast for compression progress and step telemetry.

