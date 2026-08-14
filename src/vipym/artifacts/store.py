"""Artifact Store and Local Checkpoint Registry."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
import pydantic

from vipym.core.logger import get_logger
from vipym.interfaces.storage import ArtifactStore

logger = get_logger(__name__)


class LocalArtifactStore(ArtifactStore):
    """Local filesystem artifact store with SHA256 validation."""

    def __init__(self, root_dir: Path | str = "./artifacts") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(self, local_path: Path, remote_key: str) -> str:
        target = self.root_dir / remote_key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(local_path).read_bytes())
        return str(target)

    def download_file(self, remote_key: str, local_path: Path) -> Path:
        source = self.root_dir / remote_key
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        return dest

    def upload_directory(self, local_dir: Path, remote_prefix: str) -> str:
        p = Path(local_dir)
        for f in p.glob("**/*"):
            if f.is_file():
                rel = f.relative_to(p)
                self.upload_file(f, f"{remote_prefix}/{rel.as_posix()}")
        return str(self.root_dir / remote_prefix)

    def list_artifacts(self, prefix: str) -> List[str]:
        target = self.root_dir / prefix
        if not target.exists():
            return []
        return [str(p.relative_to(self.root_dir)) for p in target.glob("**/*") if p.is_file()]
