"""SWE-bench Checkpoint Evaluation Example.

Demonstrates how to evaluate a compressed or foundational LLM checkpoint
on real-world software engineering bug resolution tasks using SWE-bench Verified
orchestrated via ViPym's BenchmarkRunner with real-time inference telemetry and cost tracking.
"""

from __future__ import annotations

import os
import sys

from vipym.config.schema import EvaluationConfig
from vipym.evaluation.runner import BenchmarkRunner
from vipym.inference.backends.hf_backend import HuggingFaceInferenceBackend
from vipym.inference.backends.vllm_backend import VLLMBackend

# Allow safe fallback when Docker daemon is not active in local dev environments
os.environ.setdefault("VIPYM_ALLOW_UNSAFE", "1")


def evaluate_swebench(
    model_path_or_id: str = "HuggingFaceTB/SmolLM-135M",
    backend_type: str = "vllm",
    task_limit: int = 2,
) -> None:
    """Run SWE-bench Verified evaluation against target model using BenchmarkRunner."""
    print("=" * 70)
    print(f"Starting SWE-bench Evaluation on: {model_path_or_id}")
    print(f"Inference Backend: {backend_type.upper()} | Task Limit: {task_limit}")
    print("=" * 70)

    # 1. Initialize serving backend
    if backend_type == "vllm":
        backend = VLLMBackend()
    else:
        backend = HuggingFaceInferenceBackend()

    try:
        backend.start(model_path_or_id)
    except Exception as e:
        print(f"[Warning] Failed to start real backend ({e}), proceeding with mock mode.")

    # 2. Configure BenchmarkRunner with sandbox security profile
    eval_cfg = EvaluationConfig(
        suites=["swebench"],
        task_limit=task_limit,
        timeout_per_task_sec=60,
        allow_unsafe_execution=True,  # Safe fallback for environments without docker
    )
    runner = BenchmarkRunner(
        evaluation_config=eval_cfg,
        model_variant=f"{model_path_or_id}-eval",
        hardware_type="A100-80GB",
        baseline_api="gpt-4o",
    )

    # 3. Execute evaluation suite
    print(f"\nExecuting SWE-bench suite on {task_limit} benchmark tasks...")
    results = runner.run_suite(
        suite_name="swebench",
        backend=backend,
        temperature=0.0,
        max_new_tokens=1024,
        task_limit=task_limit,
    )

    # 4. Report benchmark quality metrics
    print("\n" + "=" * 70)
    print("SWE-bench Evaluation Results:")
    print(f"Suite:                {results.suite_name} (v{results.benchmark_version})")
    print(f"Total Tasks:          {results.total_tasks}")
    print(f"Passed / Resolved:    {results.passed_tasks}")
    print(f"Pass@1 Rate:          {results.pass_at_1 * 100:.1f}%")
    print(f"Compilation Rate:     {results.compile_rate * 100:.1f}%")
    print("=" * 70)

    for task in results.task_results:
        status = "RESOLVED" if task.passed else "UNRESOLVED"
        print(f"  [{status}] Task: {task.task_id} ({task.execution_time_ms:.1f}ms)")
        if not task.passed and task.error_message:
            print(f"         Detail: {task.error_message[:80]}")

    # 5. Report inference cost & latency telemetry
    cost_report = runner.cost_tracker.get_report()
    print("\n" + "-" * 70)
    print("Telemetry & Cost Summary:")
    print(f"Total Tokens Processed:   {cost_report.total_tokens:,}")
    print(f"Execution Duration:       {cost_report.total_time_seconds:.2f}s")
    print(f"Hardware Serving Cost:    ${cost_report.hardware_cost_usd:.6f}")
    print(f"Cost per 1M Tokens:       ${cost_report.cost_per_1m_tokens:.4f}")
    print(
        f"Estimated API Savings:    {cost_report.cost_savings_percentage:.1f}% vs {cost_report.baseline_api_name}"
    )
    print("-" * 70)

    backend.stop()


def main() -> None:
    model_target = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM-135M"
    evaluate_swebench(model_path_or_id=model_target, backend_type="vllm", task_limit=2)


if __name__ == "__main__":
    main()
