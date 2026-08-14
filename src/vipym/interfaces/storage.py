"""Interfaces for Artifact and Manifest Storage."""

from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStore(ABC):
    """Abstract interface for local or remote (S3) artifact storage."""

    @abstractmethod
    def upload_file(self, local_path: Path, remote_key: str) -> str:
        """Upload a single file and return remote URI."""
        pass

    @abstractmethod
    def download_file(self, remote_key: str, local_path: Path) -> Path:
        """Download remote file to local path."""
        pass

    @abstractmethod
    def upload_directory(self, local_dir: Path, remote_prefix: str) -> str:
        """Upload entire directory recursively."""
        pass

    @abstractmethod
    def list_artifacts(self, prefix: str) -> list[str]:
        """List artifacts matching prefix."""
        pass
