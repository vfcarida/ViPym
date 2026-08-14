"""Contamination detection and benchmark release cutoff tracking."""

from pydantic import BaseModel


class ContaminationReport(BaseModel):
    total_tasks_checked: int
    flagged_tasks_count: int
    contamination_risk_score: float
    flagged_task_ids: list[str]


class ContaminationAuditor:
    """Detects potential contamination between training/calibration data and evaluation suites."""

    def __init__(self, n_gram_size: int = 10) -> None:
        self.n_gram_size = n_gram_size

    def extract_ngrams(self, text: str) -> set[str]:
        words = text.split()
        if len(words) < self.n_gram_size:
            return set()
        return {
            " ".join(words[i : i + self.n_gram_size])
            for i in range(len(words) - self.n_gram_size + 1)
        }

    def audit_tasks(self, tasks: list[dict], calibration_corpus: list[str]) -> ContaminationReport:
        corpus_ngrams: set[str] = set()
        for doc in calibration_corpus:
            corpus_ngrams.update(self.extract_ngrams(doc))

        flagged = []
        for task in tasks:
            prompt = task.get("prompt", "")
            task_ngrams = self.extract_ngrams(prompt)
            overlap = task_ngrams.intersection(corpus_ngrams)
            if overlap:
                flagged.append(task.get("task_id", "unknown"))

        score = len(flagged) / max(1, len(tasks))
        return ContaminationReport(
            total_tasks_checked=len(tasks),
            flagged_tasks_count=len(flagged),
            contamination_risk_score=score,
            flagged_task_ids=flagged,
        )
