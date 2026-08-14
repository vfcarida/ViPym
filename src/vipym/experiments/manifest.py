"""Deterministic Manifest and Environment Provenance Generator."""

import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vipym.__version__ import __version__ as VIPYM_VERSION
from vipym.config.constants import ExperimentState
from vipym.config.schema import ViPymExperimentConfig


class EnvironmentProvenance(BaseModel):
    """Immutable record of the software, driver, and compute hardware environment."""

    vipym_version: str
    python_version: str
    host_os: str
    host_platform: str
    torch_version: str | None = None
    transformers_version: str | None = None
    vllm_version: str | None = None
    cuda_driver_version: str | None = None
    cuda_runtime_version: str | None = None
    gpu_devices: list[str] = Field(default_factory=list)
    git_commit_sha: str | None = None
    container_digest: str | None = None

    @classmethod
    def capture(cls) -> "EnvironmentProvenance":
        gpu_devices = []
        torch_ver = None
        cuda_driver = None
        cuda_runtime = None
        transformers_ver = None
        vllm_ver = None

        try:
            import torch

            torch_ver = torch.__version__
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_devices.append(torch.cuda.get_device_name(i))
                cuda_runtime = torch.version.cuda
        except ImportError:
            pass

        try:
            import transformers

            transformers_ver = transformers.__version__
        except ImportError:
            pass

        try:
            import vllm

            vllm_ver = vllm.__version__
        except ImportError:
            pass

        git_commit = os.environ.get("GIT_COMMIT_SHA", "unknown")
        if git_commit == "unknown":
            try:
                import subprocess

                res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if res.returncode == 0:
                    git_commit = res.stdout.strip()
            except Exception:
                pass

        return cls(
            vipym_version=VIPYM_VERSION,
            python_version=sys.version.split()[0],
            host_os=platform.system(),
            host_platform=platform.platform(),
            torch_version=torch_ver,
            transformers_version=transformers_ver,
            vllm_version=vllm_ver,
            cuda_driver_version=cuda_driver,
            cuda_runtime_version=cuda_runtime,
            gpu_devices=gpu_devices,
            git_commit_sha=git_commit,
            container_digest=os.environ.get("CONTAINER_IMAGE_DIGEST"),
        )


class ReproducibilityManifest(BaseModel):
    """Immutable reproducibility manifest bundling config, provenance, and output references."""

    manifest_id: str
    experiment_id: str
    timestamp_utc: str
    config_hash_sha256: str
    config: ViPymExperimentConfig
    environment: EnvironmentProvenance
    state: ExperimentState = ExperimentState.CREATED
    duration_seconds: float = 0.0
    total_cost_usd: float = 0.0
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def create(cls, config: ViPymExperimentConfig) -> "ReproducibilityManifest":
        config_json = json.dumps(config.model_dump(), sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        ts = datetime.now(UTC).isoformat()
        manifest_id = f"manifest-{config.experiment_id}-{config_hash[:8]}"

        return cls(
            manifest_id=manifest_id,
            experiment_id=config.experiment_id,
            timestamp_utc=ts,
            config_hash_sha256=config_hash,
            config=config,
            environment=EnvironmentProvenance.capture(),
        )

    def save(self, output_path: Path | str) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
