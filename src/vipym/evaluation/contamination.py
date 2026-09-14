"""Semantic Anti-Contamination Auditor with MinHash LSH and Cryptographic Certificates.

Detects subtle benchmark leakage into pre-training, fine-tuning, or compression calibration datasets:
- Exact & canonical n-gram overlap
- MinHash Locality-Sensitive Hashing (LSH) for fuzzy, re-indented, or paraphrased code prompts
- Normalized token-level Levenshtein similarity
- Cryptographic SHA-256 ContaminationCertificate generation
"""

from __future__ import annotations

import datetime
import hashlib
import random
import re
from typing import Any

from pydantic import BaseModel, Field


class ContaminationCertificate(BaseModel):
    """Cryptographic attestation certifying benchmark non-contamination."""

    certificate_id: str
    model_or_dataset_id: str
    timestamp_iso: str
    total_tasks_audited: int
    clean: bool
    contamination_risk_score: float
    max_observed_similarity: float
    sha256_digest: str
    threshold_used: float
    details: dict[str, Any] = Field(default_factory=dict)


class ContaminationReport(BaseModel):
    """Aggregated contamination audit report with per-task diagnostic traceability."""

    total_tasks_checked: int
    flagged_tasks_count: int
    contamination_risk_score: float
    flagged_task_ids: list[str]
    max_similarity: float = 0.0
    match_reasons: dict[str, str] = Field(default_factory=dict)
    certificate: ContaminationCertificate | None = None


def normalize_code_text(text: str) -> str:
    """Normalize code text by standardizing comments and redundant whitespace."""
    # Strip single line comments (# or //)
    text = re.sub(r"(#|//).*$", "", text, flags=re.MULTILINE)
    # Extract alphanumeric words to avoid punctuation boilerplate collisions
    words = re.findall(r"[A-Za-z0-9_]+", text)
    return " ".join(words).lower()


def compute_token_levenshtein_similarity(seq1: list[str], seq2: list[str]) -> float:
    """Compute normalized token sequence similarity in range [0.0, 1.0]."""
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0

    s1 = seq1[:200]
    s2 = seq2[:200]
    len1, len2 = len(s1), len(s2)

    dp = [list(range(len2 + 1))] + [[i + 1] + [0] * len2 for i in range(len1)]
    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            dp[i + 1][j + 1] = min(
                dp[i][j + 1] + 1,
                dp[i + 1][j + 1] + 1,
                dp[i][j] + cost,
            )

    dist = dp[len1][len2]
    max_len = max(len1, len2)
    return max(0.0, 1.0 - (dist / max_len))


class MinHashSignature:
    """MinHash signature generator using universal hash families."""

    def __init__(self, num_perm: int = 64, seed: int = 42) -> None:
        self.num_perm = num_perm
        self.prime = 4294967311  # Large 32-bit prime
        rng = random.Random(seed)
        self.a = [rng.randint(1, self.prime - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.prime - 1) for _ in range(num_perm)]

    def compute_signature(self, tokens: list[str]) -> list[int]:
        """Compute MinHash signature vector for a sequence of tokens."""
        if not tokens:
            return [0] * self.num_perm

        token_hashes = [
            int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
            for tok in set(tokens)
        ]

        sig = [0xFFFFFFFF] * self.num_perm
        for h in token_hashes:
            for i in range(self.num_perm):
                val = (self.a[i] * h + self.b[i]) % self.prime
                if val < sig[i]:
                    sig[i] = val
        return sig

    def estimate_jaccard(self, sig1: list[int], sig2: list[int]) -> float:
        """Estimate Jaccard similarity between two MinHash signatures."""
        if len(sig1) != len(sig2) or not sig1:
            return 0.0
        matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return matches / len(sig1)


class ContaminationAuditor:
    """Detects potential contamination between training/calibration data and evaluation suites."""

    def __init__(
        self,
        n_gram_size: int = 10,
        num_perm: int = 64,
        similarity_threshold: float = 0.70,
        seed: int = 42,
    ) -> None:
        self.n_gram_size = n_gram_size
        self.similarity_threshold = similarity_threshold
        self.minhasher = MinHashSignature(num_perm=num_perm, seed=seed)

    def extract_ngrams(self, text: str) -> set[str]:
        """Extract word n-grams from text using whitespace separation."""
        words = text.split()
        if len(words) < self.n_gram_size:
            return set()
        return {
            " ".join(words[i : i + self.n_gram_size])
            for i in range(len(words) - self.n_gram_size + 1)
        }

    def audit_tasks(
        self,
        tasks: list[dict[str, Any]],
        calibration_corpus: list[str],
        dataset_id: str = "calibration-dataset",
    ) -> ContaminationReport:
        """Audit benchmark tasks against calibration corpus using multi-tier detection."""
        corpus_ngrams: set[str] = set()
        corpus_signatures: list[tuple[list[int], list[str]]] = []

        for doc in calibration_corpus:
            corpus_ngrams.update(self.extract_ngrams(doc))
            norm_doc = normalize_code_text(doc)
            tokens = norm_doc.split()
            if len(tokens) >= 5:
                sig = self.minhasher.compute_signature(tokens)
                corpus_signatures.append((sig, tokens))

        flagged: list[str] = []
        match_reasons: dict[str, str] = {}
        max_overall_similarity = 0.0

        for task in tasks:
            task_id = str(task.get("task_id", "unknown"))
            prompt = str(task.get("prompt", ""))
            norm_prompt = normalize_code_text(prompt)
            task_tokens = norm_prompt.split()

            task_flagged = False
            task_max_sim = 0.0
            flag_reason = ""

            # Tier 1: Exact word N-gram overlap
            task_ngrams = self.extract_ngrams(prompt)
            overlap = task_ngrams.intersection(corpus_ngrams)
            if overlap:
                task_flagged = True
                task_max_sim = 1.0
                flag_reason = f"Exact {self.n_gram_size}-gram overlap ({len(overlap)} ngrams)"

            # Tier 2: MinHash fuzzy Jaccard & Normalized Edit Distance for substantive prompts
            if not task_flagged and len(task_tokens) >= 5:
                task_sig = self.minhasher.compute_signature(task_tokens)
                for doc_sig, doc_tokens in corpus_signatures:
                    jaccard_est = self.minhasher.estimate_jaccard(task_sig, doc_sig)
                    if jaccard_est > task_max_sim:
                        task_max_sim = jaccard_est

                    if jaccard_est >= 0.50:
                        edit_sim = compute_token_levenshtein_similarity(task_tokens, doc_tokens)
                        if edit_sim > task_max_sim:
                            task_max_sim = edit_sim
                        if edit_sim >= self.similarity_threshold:
                            task_flagged = True
                            flag_reason = f"Fuzzy semantic match (MinHash: {jaccard_est:.2f}, EditSim: {edit_sim:.2f})"
                            break

            if task_max_sim > max_overall_similarity:
                max_overall_similarity = task_max_sim

            if task_flagged:
                flagged.append(task_id)
                match_reasons[task_id] = flag_reason

        risk_score = round(len(flagged) / max(1, len(tasks)), 4)

        report = ContaminationReport(
            total_tasks_checked=len(tasks),
            flagged_tasks_count=len(flagged),
            contamination_risk_score=risk_score,
            flagged_task_ids=flagged,
            max_similarity=round(max_overall_similarity, 4),
            match_reasons=match_reasons,
        )

        # Issue cryptographic certificate
        certificate = self.issue_certificate(report, model_or_dataset_id=dataset_id)
        report.certificate = certificate
        return report

    def issue_certificate(
        self,
        report: ContaminationReport,
        model_or_dataset_id: str,
    ) -> ContaminationCertificate:
        """Issue cryptographic SHA-256 proof of anti-contamination verification."""
        timestamp_iso = datetime.datetime.now(datetime.UTC).isoformat()
        clean = report.flagged_tasks_count == 0

        # Construct deterministic digest payload
        payload = f"{model_or_dataset_id}:{report.total_tasks_checked}:{report.flagged_tasks_count}:{clean}:{self.similarity_threshold}:{timestamp_iso}"
        sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        cert_id = f"CERT-VIPYM-{sha256[:12].upper()}"

        return ContaminationCertificate(
            certificate_id=cert_id,
            model_or_dataset_id=model_or_dataset_id,
            timestamp_iso=timestamp_iso,
            total_tasks_audited=report.total_tasks_checked,
            clean=clean,
            contamination_risk_score=report.contamination_risk_score,
            max_observed_similarity=report.max_similarity,
            sha256_digest=sha256,
            threshold_used=self.similarity_threshold,
            details={
                "flagged_tasks_count": report.flagged_tasks_count,
                "flagged_task_ids": report.flagged_task_ids,
            },
        )
