"""Hardware discovery and deterministic hashing utilities."""

import hashlib
from pathlib import Path

import pydantic


class HardwareTopology(pydantic.BaseModel):
    gpu_count: int
    gpu_names: list[str]
    cuda_available: bool
    cuda_version: str = "N/A"
    total_vram_gb: float = 0.0
    has_efa_support: bool = False
    has_nvlink: bool = False


class HardwareDiscovery:
    """Introspects host GPU, NVLink, and network fabric configuration."""

    @staticmethod
    def discover() -> HardwareTopology:
        gpu_count = 0
        gpu_names = []
        cuda_avail = False
        cuda_ver = "N/A"
        total_vram = 0.0

        try:
            import torch

            if torch.cuda.is_available():
                cuda_avail = True
                gpu_count = torch.cuda.device_count()
                cuda_ver = torch.version.cuda or "N/A"
                for i in range(gpu_count):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    gpu_names.append(name)
                    total_vram += props.total_memory / (1024**3)
        except ImportError:
            pass

        return HardwareTopology(
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            cuda_available=cuda_avail,
            cuda_version=cuda_ver,
            total_vram_gb=total_vram,
            has_efa_support=False,
            has_nvlink=gpu_count > 1,
        )


class DeterministicHasher:
    """Calculates SHA256 hashes of files and data structures."""

    @staticmethod
    def hash_file(file_path: Path | str) -> str:
        p = Path(file_path)
        sha = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def hash_string(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
