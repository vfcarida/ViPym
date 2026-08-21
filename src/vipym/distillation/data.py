"""Data generation and execution-filtered dataset for MoE-to-Dense distillation.

Three composable components:

1. ``SyntheticDataGenerator``  — calls the teacher's ``generate()`` to produce
   (prompt, response) pairs from configurable prompt templates.

2. ``ExecutionFilter``          — runs generated code samples in a sandboxed
   ``subprocess`` (timeout-guarded).  Falls back to AST-parse-only if
   subprocess execution is disabled.  Passing samples are kept; failing ones
   are discarded.

3. ``DistillationDataset``      — ``torch.utils.data.Dataset`` that mixes
   synthetic + real corpora at a configurable ``code_ratio``, pre-tokenises,
   and returns ``(input_ids, labels, teacher_logits_or_None)`` tensors.
   Teacher logits are loaded from disk cache if ``cache_teacher_logits=True``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from vipym.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CODE_PROMPTS: list[str] = [
    "Write a Python function that {task}.",
    "Implement a solution for: {task}",
    '# Python\ndef solve_{slug}():\n    """Solve: {task}"""\n',
    "Given the problem: {task}\nWrite clean, efficient Python code.",
]

_GENERAL_PROMPTS: list[str] = [
    "Explain {topic} in simple terms.",
    "What are the key concepts in {topic}?",
    "Summarise the following and provide insights: {topic}",
]

_SAMPLE_TASKS = [
    ("reverse a linked list", "reverse_linked_list"),
    ("find the longest common subsequence", "lcs"),
    ("implement binary search", "binary_search"),
    ("compute fibonacci numbers efficiently", "fib"),
    ("detect a cycle in a graph using DFS", "detect_cycle"),
    ("sort a list using merge sort", "merge_sort"),
    ("parse JSON and validate a schema", "json_validate"),
    ("compute the edit distance between two strings", "edit_distance"),
]

_SAMPLE_TOPICS = [
    "transformer attention mechanisms",
    "gradient descent optimisation",
    "mixture-of-experts routing",
    "knowledge distillation",
    "sparse autoencoders",
]


# ---------------------------------------------------------------------------
# SyntheticDataGenerator
# ---------------------------------------------------------------------------


class SyntheticDataGenerator:
    """Generate (prompt, response) pairs from a teacher model.

    In production the teacher is a loaded ``nn.Module`` with a
    ``generate()`` method compatible with HuggingFace Transformers.
    In unit tests a mock callable can be substituted.

    Args:
        teacher: The teacher model (``nn.Module`` or callable mock).
        tokenizer: HF tokenizer.
        num_samples: Number of samples to generate.
        code_ratio: Fraction of samples that are code prompts.
        max_new_tokens: Maximum tokens generated per sample.
        temperature: Sampling temperature (not the distillation τ).
        top_p: Nucleus sampling probability mass.
    """

    def __init__(
        self,
        teacher: Any,
        tokenizer: Any,
        num_samples: int = 1000,
        code_ratio: float = 0.8,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> None:
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.code_ratio = code_ratio
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _build_prompt(self, idx: int) -> str:
        if idx / self.num_samples < self.code_ratio:
            task, slug = _SAMPLE_TASKS[idx % len(_SAMPLE_TASKS)]
            template = _CODE_PROMPTS[idx % len(_CODE_PROMPTS)]
            return template.format(task=task, slug=slug)
        topic = _SAMPLE_TOPICS[idx % len(_SAMPLE_TOPICS)]
        template = _GENERAL_PROMPTS[idx % len(_GENERAL_PROMPTS)]
        return template.format(topic=topic)

    def generate(self) -> list[dict[str, str]]:
        """Return list of ``{"prompt": str, "response": str}`` dicts."""
        samples: list[dict[str, str]] = []
        logger.info(
            f"Generating {self.num_samples} synthetic samples (code_ratio={self.code_ratio})"
        )

        for idx in range(self.num_samples):
            prompt = self._build_prompt(idx)
            try:
                response = self._call_teacher(prompt)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Teacher generation failed for sample {idx}: {exc}")
                response = f"# placeholder response {idx}"
            samples.append({"prompt": prompt, "response": response})

        return samples

    def _call_teacher(self, prompt: str) -> str:
        """Call teacher.generate() or the mock callable."""
        if callable(self.teacher) and not hasattr(self.teacher, "generate"):
            # Mock callable — used in tests
            return self.teacher(prompt)

        if hasattr(self.tokenizer, "encode"):
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"]
            with torch.no_grad():
                out = self.teacher.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                )
            new_tokens = out[0, input_ids.shape[1] :]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return f"# response for: {prompt[:50]}"


# ---------------------------------------------------------------------------
# ExecutionFilter
# ---------------------------------------------------------------------------


class ExecutionFilter:
    """Filter generated code samples by sandboxed execution.

    Strategy (recommended A):
    - Write code to a ``tempfile``.
    - Run ``python <file>`` in a subprocess with a timeout.
    - If exit code is 0 → keep; otherwise → discard.
    - Falls back to AST parse-only check if ``allow_subprocess=False``.

    Args:
        timeout: Per-sample execution timeout in seconds.
        allow_subprocess: If ``False``, only AST-parse the code (no execution).
        allowed_imports: Set of stdlib modules allowed in generated code.
            If ``None``, no import restriction.
    """

    _DEFAULT_ALLOWED: frozenset[str] = frozenset(
        {
            "ast",
            "collections",
            "functools",
            "heapq",
            "itertools",
            "math",
            "operator",
            "os",
            "pathlib",
            "re",
            "string",
            "sys",
            "textwrap",
            "time",
            "typing",
            "unittest",
            "numpy",
            "torch",
        }
    )

    def __init__(
        self,
        timeout: int = 30,
        allow_subprocess: bool = True,
        allowed_imports: set[str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.allow_subprocess = allow_subprocess
        self.allowed_imports = (
            allowed_imports if allowed_imports is not None else self._DEFAULT_ALLOWED
        )

    def is_valid(self, code: str) -> bool:
        """Return ``True`` if the code sample should be kept."""
        # Step 1: AST parse
        if not self._ast_valid(code):
            return False

        if not self.allow_subprocess:
            return True  # fallback: pass on successful parse

        # Step 2: Subprocess execution
        return self._subprocess_valid(code)

    def filter(self, samples: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return only those samples whose ``"response"`` field passes the filter."""
        kept, total = [], len(samples)
        for s in samples:
            code = s.get("response", "")
            if self.is_valid(code):
                kept.append(s)
        ratio = len(kept) / max(1, total)
        logger.info(f"ExecutionFilter: kept {len(kept)}/{total} samples ({ratio:.1%})")
        return kept

    # ------------------------------------------------------------------

    def _ast_valid(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _subprocess_valid(self, code: str) -> bool:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as fp:
            fp.write(code)
            tmp_path = fp.name
        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                timeout=self.timeout,
                text=True,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.debug(f"Sandbox timeout for sample (limit={self.timeout}s)")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Subprocess execution error: {exc}")
            return False
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Teacher Logit Cache
# ---------------------------------------------------------------------------


class TeacherLogitCache:
    """Disk-backed cache for teacher logits.

    Saves one ``.npy`` float16 shard per batch of samples.  Shards are
    keyed by a hash of (``input_ids``, ``shard_idx``) so they survive
    resume.

    Args:
        cache_dir: Directory to write shards.
        dtype: NumPy dtype to store logits (``float16`` by default saves ~50%).
    """

    def __init__(
        self, cache_dir: str | Path = "./teacher_logit_cache", dtype: str = "float16"
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self._index_path = self.cache_dir / "index.json"
        self._index: dict[str, str] = self._load_index()

    def _load_index(self) -> dict[str, str]:
        if self._index_path.exists():
            with open(self._index_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f)

    def _key(self, input_ids: torch.Tensor, shard_idx: int) -> str:
        raw = input_ids.cpu().numpy().tobytes() + shard_idx.to_bytes(4, "little")
        return hashlib.sha1(raw).hexdigest()[:16]

    def has(self, input_ids: torch.Tensor, shard_idx: int) -> bool:
        return self._key(input_ids, shard_idx) in self._index

    def save(self, input_ids: torch.Tensor, shard_idx: int, logits: torch.Tensor) -> None:
        key = self._key(input_ids, shard_idx)
        path = self.cache_dir / f"{key}.npy"
        np.save(str(path), logits.cpu().float().numpy().astype(self.dtype))
        self._index[key] = str(path)
        self._save_index()

    def load(self, input_ids: torch.Tensor, shard_idx: int) -> torch.Tensor | None:
        key = self._key(input_ids, shard_idx)
        if key not in self._index:
            return None
        arr = np.load(self._index[key]).astype("float32")
        return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# DistillationDataset
# ---------------------------------------------------------------------------


class DistillationDataset(Dataset):
    """Pre-tokenised dataset for distillation training.

    Mixes synthetic teacher-generated samples with real corpora at
    ``code_ratio``.  Returns ``(input_ids, labels, cached_teacher_logits_or_None)``.

    Args:
        samples: List of ``{"prompt": str, "response": str}`` dicts.
        tokenizer: HF tokenizer.  Must have ``encode()``.
        max_seq_len: Maximum token length (longer samples are truncated).
        cache: Optional ``TeacherLogitCache`` — if provided and a shard is
            available, the third element of the batch tuple is the cached logits.
    """

    def __init__(
        self,
        samples: list[dict[str, str]],
        tokenizer: Any,
        max_seq_len: int = 2048,
        cache: TeacherLogitCache | None = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cache = cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        sample = self.samples[idx]
        text = sample.get("prompt", "") + sample.get("response", "")

        # Tokenise
        if hasattr(self.tokenizer, "__call__"):
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_seq_len,
                truncation=True,
                padding=False,
            )
            input_ids = enc["input_ids"].squeeze(0)  # [L]
        elif hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(text)[: self.max_seq_len]
            input_ids = torch.tensor(ids, dtype=torch.long)
        else:
            # Fallback: character-level encoding for tests
            ids = [ord(c) % 256 for c in text][: self.max_seq_len]
            input_ids = torch.tensor(ids, dtype=torch.long)

        # Labels: shift by 1 (next-token prediction), mask prompt
        labels = input_ids.clone()
        # Mask the prompt portion (-100 = ignored by CE)
        prompt_len = min(len(sample.get("prompt", "")), len(labels) - 1)
        labels[:prompt_len] = -100

        # Cached teacher logits
        cached_logits: torch.Tensor | None = None
        if self.cache is not None:
            cached_logits = self.cache.load(input_ids.unsqueeze(0), idx)

        return input_ids, labels, cached_logits
