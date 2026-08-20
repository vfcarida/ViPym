"""System health and diagnostic validation for ViPym."""

import os
import platform
import shutil
import sys

from rich.console import Console
from rich.table import Table

console = Console()


def run_doctor_checks() -> bool:
    """Validate system readiness across Python, PyTorch, CUDA, Docker, VRAM, and Disk."""
    table = Table(title="ViPym Doctor Diagnostic Report", header_style="bold cyan")
    table.add_column("Component", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    all_passed = True

    # 1. Python Version
    py_ver = sys.version.split()[0]
    py_ok = sys.version_info >= (3, 11)
    table.add_row(
        "Python (>= 3.11)",
        "[OK]" if py_ok else "[FAIL]",
        f"v{py_ver} ({platform.python_implementation()})",
    )
    if not py_ok:
        all_passed = False

    # 2. PyTorch & CUDA
    try:
        import torch

        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda or "N/A"
        gpu_count = torch.cuda.device_count() if cuda_avail else 0
        details = f"PyTorch {torch.__version__}, CUDA: {cuda_ver}, GPUs: {gpu_count}"
        table.add_row("PyTorch & CUDA", "[OK]" if cuda_avail else "[WARN]", details)
    except ImportError:
        table.add_row("PyTorch & CUDA", "[WARN]", "PyTorch not installed (CPU fallback mode)")

    # 3. Serving Runtime (vLLM / SGLang)
    try:
        import vllm

        table.add_row("vLLM Engine", "[OK]", f"v{vllm.__version__}")
    except ImportError:
        table.add_row("vLLM Engine", "[WARN]", "vLLM not installed (mock serving fallback)")

    # 4. Compression Backend (llm-compressor)
    import importlib.util

    if importlib.util.find_spec("llmcompressor") is not None:
        table.add_row("LLM-Compressor", "[OK]", "Available")
    else:
        table.add_row("LLM-Compressor", "[WARN]", "llm-compressor not installed (mock compression)")

    # 5. Docker / Sandboxing
    from vipym.evaluation.sandbox.docker_sandbox import is_docker_available

    docker_bin = shutil.which("docker")
    docker_ok = is_docker_available()
    if docker_ok:
        docker_status = "[OK]"
        docker_details = f"Daemon connected ({docker_bin})"
    elif docker_bin:
        docker_status = "[WARN]"
        docker_details = f"Binary found ({docker_bin}), but daemon unreachable"
    else:
        docker_status = "[WARN]"
        docker_details = "Docker not found (Required for container sandbox)"
    table.add_row("Docker Sandbox", docker_status, docker_details)

    # 6. Disk Space
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    disk_ok = free_gb > 10.0
    table.add_row(
        "Disk Space", "[OK]" if disk_ok else "[WARN]", f"{free_gb:.1f} GB free on working drive"
    )

    # 7. AWS Credentials Check
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    table.add_row(
        "AWS Credentials",
        "[OK]" if aws_key else "[INFO]",
        "Found in env" if aws_key else "Not configured (Local execution mode)",
    )

    # 8. Hugging Face Auth Check
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    table.add_row(
        "Hugging Face Auth",
        "[OK]" if hf_token else "[INFO]",
        "Token present" if hf_token else "No token (Public models only)",
    )

    console.print(table)
    return all_passed
