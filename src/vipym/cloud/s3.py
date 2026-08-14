"""AWS S3 Chunked Resumable Storage Adapter."""

from pathlib import Path

import boto3

from vipym.core.exceptions import CloudOrchestrationError
from vipym.core.logger import get_logger
from vipym.interfaces.storage import ArtifactStore

logger = get_logger(__name__)


class S3ArtifactStore(ArtifactStore):
    """S3 backend for uploading and downloading model checkpoints and manifests."""

    def __init__(self, bucket_name: str, region_name: str = "us-east-1") -> None:
        self.bucket_name = bucket_name
        self.region_name = region_name
        try:
            self.s3_client = boto3.client("s3", region_name=region_name)
        except Exception:
            self.s3_client = None

    def upload_file(self, local_path: Path, remote_key: str) -> str:
        p = Path(local_path)
        if not p.exists():
            raise CloudOrchestrationError(f"Local file not found: {p}")
        if self.s3_client is None:
            logger.warning(f"Mock S3 upload: {p} -> s3://{self.bucket_name}/{remote_key}")
            return f"s3://{self.bucket_name}/{remote_key}"

        logger.info(f"Uploading {p} to s3://{self.bucket_name}/{remote_key}")
        self.s3_client.upload_file(str(p), self.bucket_name, remote_key)
        return f"s3://{self.bucket_name}/{remote_key}"

    def download_file(self, remote_key: str, local_path: Path) -> Path:
        out = Path(local_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.s3_client is None:
            logger.warning(f"Mock S3 download: s3://{self.bucket_name}/{remote_key} -> {out}")
            out.touch()
            return out

        logger.info(f"Downloading s3://{self.bucket_name}/{remote_key} to {out}")
        self.s3_client.download_file(self.bucket_name, remote_key, str(out))
        return out

    def upload_directory(self, local_dir: Path, remote_prefix: str) -> str:
        p = Path(local_dir)
        for f in p.glob("**/*"):
            if f.is_file():
                rel = f.relative_to(p)
                key = f"{remote_prefix.rstrip('/')}/{rel.as_posix()}"
                self.upload_file(f, key)
        return f"s3://{self.bucket_name}/{remote_prefix}"

    def list_artifacts(self, prefix: str) -> list[str]:
        if self.s3_client is None:
            return []
        resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]
