"""Intelligent Calibration Dataset Manager with Automated Contamination Purging.

Manages code calibration datasets for post-training quantization (AWQ, GPTQ, SmoothQuant, AutoRound)
and pruning (Wanda, SparseGPT). Automatically audits and removes prompts overlapping with
evaluation benchmark suites (HumanEval, MBPP, etc.) prior to model calibration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vipym.evaluation.contamination import ContaminationAuditor
from vipym.evaluation.registry import EvaluationRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bundled High-Quality Code Calibration Samples (Offline / Fallback)
# ---------------------------------------------------------------------------

_CANONICAL_CODE_CORPUS = [
    """import os
import sys
from typing import List, Optional

def scan_directory(root: str, extension: Optional[str] = None) -> List[str]:
    \"\"\"Scan directory recursively for files matching extension.\"\"\"
    matched_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if extension is None or fname.endswith(extension):
                matched_files.append(os.path.join(dirpath, fname))
    return matched_files
""",
    """import math

def calculate_entropy(probabilities: list[float]) -> float:
    \"\"\"Compute Shannon entropy for a given probability distribution.\"\"\"
    total = sum(probabilities)
    if total == 0:
        return 0.0
    normalized = [p / total for p in probabilities if p > 0]
    return -sum(p * math.log2(p) for p in normalized)
""",
    """import numpy as np

def matrix_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    \"\"\"Compute cosine similarity between two feature vectors.\"\"\"
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
""",
    """from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MetricRecord:
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]

    def serialize(self) -> Dict[str, Any]:
        return {"metric": self.name, "val": self.value, "ts": self.timestamp, "tags": self.tags}
""",
]


class CalibrationConfig(BaseModel):
    """Configuration for calibration data ingestion and preprocessing."""

    dataset_name_or_path: str = Field(
        default="bigcode/the-stack",
        description="HuggingFace dataset repository name or local JSONL/TXT file path",
    )
    split: str = Field(default="train", description="Dataset split to extract samples from")
    num_samples: int = Field(default=256, ge=1, description="Number of calibration samples")
    sequence_length: int = Field(default=2048, ge=64, description="Context window chunk length")
    seed: int = Field(default=42, description="Deterministic sampling seed")
    purge_contaminated: bool = Field(
        default=True,
        description="Automatically remove samples that overlap with evaluation benchmark suites",
    )
    eval_suites_to_check: list[str] = Field(
        default_factory=lambda: ["humaneval", "mbpp"],
        description="Benchmark suites to audit and purge from calibration corpus",
    )


class CalibrationDatasetManager:
    """Loads, sanitizes, and chunks code corpora for model calibration."""

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self.auditor = ContaminationAuditor(n_gram_size=8)

    def load_raw_samples(self) -> list[str]:
        """Ingest raw code samples from Hugging Face or local files with fallback."""
        ds_source = self.config.dataset_name_or_path
        samples: list[str] = []

        # 1. Try local file
        local_path = Path(ds_source)
        if local_path.exists() and local_path.is_file():
            try:
                if local_path.suffix in {".jsonl", ".json"}:
                    with open(local_path, encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            code = data.get("content") or data.get("code") or data.get("text", "")
                            if code:
                                samples.append(code)
                else:
                    text = local_path.read_text(encoding="utf-8")
                    samples = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
                logger.info(
                    f"Loaded {len(samples)} calibration samples from local file: {local_path}"
                )
                return samples[: self.config.num_samples]
            except Exception as e:
                logger.warning(f"Failed to read local calibration file {local_path}: {e}")

        # 2. Try Hugging Face datasets
        try:
            from datasets import load_dataset  # type: ignore[import]

            hf_ds = load_dataset(ds_source, split=self.config.split, streaming=True)
            for item in hf_ds:
                code = item.get("content") or item.get("code") or item.get("text", "")
                if code and len(code.strip()) > 50:
                    samples.append(code.strip())
                if len(samples) >= self.config.num_samples * 2:
                    break
            logger.info(
                f"Loaded {len(samples)} calibration samples from Hugging Face dataset: {ds_source}"
            )
            return samples
        except Exception as err:
            logger.info(
                f"HuggingFace dataset load for '{ds_source}' skipped ({err}). "
                f"Using bundled canonical code calibration corpus."
            )

        # 3. Bundled Fallback
        return _CANONICAL_CODE_CORPUS * max(
            1, (self.config.num_samples // len(_CANONICAL_CODE_CORPUS) + 1)
        )

    def purge_contamination(self, samples: list[str]) -> list[str]:
        """Scan samples against target benchmark tasks and purge overlapping items."""
        if not self.config.purge_contaminated:
            return samples

        # Collect benchmark prompts from registered suites
        benchmark_tasks: list[dict[str, Any]] = []
        for suite_name in self.config.eval_suites_to_check:
            try:
                suite = EvaluationRegistry.get(suite_name)
                tasks = suite.load_tasks(limit=100)
                for t in tasks:
                    benchmark_tasks.append({"task_id": t.task_id, "prompt": t.prompt})
            except Exception as e:
                logger.debug(f"Could not load suite '{suite_name}' for contamination audit: {e}")

        if not benchmark_tasks:
            return samples

        clean_samples: list[str] = []
        purged_count = 0

        for sample in samples:
            report = self.auditor.audit_tasks(benchmark_tasks, calibration_corpus=[sample])
            if report.flagged_tasks_count > 0:
                purged_count += 1
                logger.warning(
                    f"Purged contaminated calibration sample overlapping with {report.flagged_tasks_count} "
                    f"benchmark tasks ({report.flagged_task_ids[:3]})."
                )
            else:
                clean_samples.append(sample)

        if purged_count > 0:
            logger.info(
                f"Anti-contamination filter purged {purged_count}/{len(samples)} "
                f"samples ({len(clean_samples)} clean samples remaining)."
            )

        return clean_samples if clean_samples else samples

    def get_calibration_corpus(self) -> list[str]:
        """Fetch, sanitize, and prepare full calibration dataset."""
        raw = self.load_raw_samples()
        clean = self.purge_contamination(raw)
        return clean[: self.config.num_samples]

    def tokenize_and_chunk(
        self,
        corpus: list[str],
        tokenizer: Any,
        sequence_length: int | None = None,
        max_samples: int | None = None,
    ) -> list[Any]:
        """Tokenize code samples and chunk into uniform sequence lengths for quantization."""
        seq_len = sequence_length or self.config.sequence_length
        limit = max_samples or self.config.num_samples

        all_input_ids: list[int] = []
        for doc in corpus:
            tokens = tokenizer(doc, truncation=False, return_tensors=None).get("input_ids", [])
            all_input_ids.extend(tokens)

        # Chunk into fixed sequence lengths
        chunks = []
        for i in range(0, len(all_input_ids) - seq_len + 1, seq_len):
            chunk = all_input_ids[i : i + seq_len]
            chunks.append(chunk)
            if len(chunks) >= limit:
                break

        return chunks
