"""Unit tests for Semantic Anti-Contamination Auditor and Cryptographic Certificates."""

from vipym.evaluation.contamination import (
    ContaminationAuditor,
    ContaminationCertificate,
    MinHashSignature,
    compute_token_levenshtein_similarity,
    normalize_code_text,
)


def test_normalize_code_text() -> None:
    """Verify code normalization strips comments and standardizes whitespace."""
    code = """
    # This is a comment
    def add(a, b): // another comment
        return a + b
    """
    normalized = normalize_code_text(code)
    assert "#" not in normalized
    assert "//" not in normalized
    assert "def add a b return a b" == normalized


def test_compute_token_levenshtein_similarity() -> None:
    """Verify normalized edit distance returns expected similarity bounds."""
    seq1 = ["def", "foo", "(", "x", ")", ":", "return", "x", "+", "1"]
    seq2 = ["def", "foo", "(", "x", ")", ":", "return", "x", "+", "1"]
    assert compute_token_levenshtein_similarity(seq1, seq2) == 1.0

    seq3 = ["def", "foo", "(", "y", ")", ":", "return", "y", "+", "1"]
    sim = compute_token_levenshtein_similarity(seq1, seq3)
    assert sim >= 0.80

    seq_empty: list[str] = []
    assert compute_token_levenshtein_similarity(seq1, seq_empty) == 0.0


def test_minhash_signature() -> None:
    """Verify MinHash Jaccard estimation for identical, overlapping, and disjoint sets."""
    minhasher = MinHashSignature(num_perm=64, seed=42)

    tokens1 = ["def", "sort_array", "arr", "len", "return", "sorted"]
    tokens2 = ["def", "sort_array", "arr", "len", "return", "sorted"]
    tokens3 = ["class", "DatabaseConnection", "connect", "cursor", "execute"]

    sig1 = minhasher.compute_signature(tokens1)
    sig2 = minhasher.compute_signature(tokens2)
    sig3 = minhasher.compute_signature(tokens3)

    assert minhasher.estimate_jaccard(sig1, sig2) == 1.0
    assert minhasher.estimate_jaccard(sig1, sig3) < 0.20


def test_auditor_exact_leakage() -> None:
    """Verify detection of exact benchmark prompt in calibration corpus."""
    auditor = ContaminationAuditor(n_gram_size=6)
    tasks = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:",
        }
    ]
    corpus = [
        "import math\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    pass"
    ]

    report = auditor.audit_tasks(tasks, corpus, dataset_id="test-corpus-exact")
    assert report.total_tasks_checked == 1
    assert report.flagged_tasks_count == 1
    assert "HumanEval/0" in report.flagged_task_ids
    assert report.contamination_risk_score == 1.0
    assert report.certificate is not None
    assert report.certificate.clean is False


def test_auditor_fuzzy_leakage() -> None:
    """Verify detection of re-indented and paraphrased prompt via MinHash and Levenshtein."""
    auditor = ContaminationAuditor(n_gram_size=30, similarity_threshold=0.70)

    tasks = [
        {
            "task_id": "HumanEval/1",
            "prompt": "def separate_paren_groups(paren_string: str) -> List[str]:\n    ''' Input to this function is a string containing multiple groups '''",
        }
    ]
    # Paraphrased in corpus with modified variable names and slight wording edits
    corpus = [
        "def separate_paren_groups(paren_str: str) -> List[str]:\n    ''' Input to this function is string containing multiple groups '''\n    return []"
    ]

    report = auditor.audit_tasks(tasks, corpus, dataset_id="test-corpus-fuzzy")
    assert report.total_tasks_checked == 1
    assert report.flagged_tasks_count == 1
    assert "HumanEval/1" in report.flagged_task_ids
    assert "Fuzzy semantic match" in report.match_reasons["HumanEval/1"]


def test_auditor_clean_corpus_certificate() -> None:
    """Verify clean dataset generates passing certificate with SHA-256 digest."""
    auditor = ContaminationAuditor()
    tasks = [
        {
            "task_id": "HumanEval/2",
            "prompt": "def truncate_number(number: float) -> float:",
        }
    ]
    corpus = [
        "import requests\ndef fetch_user_data(user_id: int):\n    return requests.get(f'/users/{user_id}')"
    ]

    report = auditor.audit_tasks(tasks, corpus, dataset_id="clean-corpus-v1")
    assert report.flagged_tasks_count == 0
    assert report.contamination_risk_score == 0.0

    cert = report.certificate
    assert isinstance(cert, ContaminationCertificate)
    assert cert.clean is True
    assert cert.model_or_dataset_id == "clean-corpus-v1"
    assert len(cert.sha256_digest) == 64
    assert cert.certificate_id.startswith("CERT-VIPYM-")
