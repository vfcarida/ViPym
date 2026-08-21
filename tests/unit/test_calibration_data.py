"""Unit tests for CalibrationDatasetManager and anti-contamination filters."""

from __future__ import annotations

import json
from pathlib import Path

from vipym.data.calibration import CalibrationConfig, CalibrationDatasetManager


class TestCalibrationDatasetManager:
    def test_load_fallback_samples(self):
        """Verify manager loads high-quality fallback corpus when offline."""
        cfg = CalibrationConfig(dataset_name_or_path="non_existent/dataset_12345", num_samples=3)
        mgr = CalibrationDatasetManager(cfg)
        samples = mgr.get_calibration_corpus()
        assert len(samples) >= 3
        assert any("def " in s for s in samples)

    def test_load_local_jsonl_samples(self, tmp_path: Path):
        """Verify manager ingests local JSONL dataset files."""
        jsonl_file = tmp_path / "calibration.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"code": "def foo():\n    return 42\n"}) + "\n")
            f.write(json.dumps({"code": "def bar(x):\n    return x * 2\n"}) + "\n")

        cfg = CalibrationConfig(dataset_name_or_path=str(jsonl_file), num_samples=2)
        mgr = CalibrationDatasetManager(cfg)
        samples = mgr.get_calibration_corpus()
        assert len(samples) == 2
        assert "def foo" in samples[0]
        assert "def bar" in samples[1]

    def test_purge_contamination_filter(self):
        """Verify contaminated samples overlapping with benchmark prompts are purged."""
        contaminated_sample = (
            "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
            '    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n'
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
        )
        clean_sample = (
            "import os\ndef scan_files(path: str) -> list[str]:\n    return os.listdir(path)\n"
        )

        cfg = CalibrationConfig(purge_contaminated=True, eval_suites_to_check=["humaneval"])
        mgr = CalibrationDatasetManager(cfg)

        filtered = mgr.purge_contamination([clean_sample, contaminated_sample])
        assert clean_sample in filtered
        assert contaminated_sample not in filtered

    def test_tokenize_and_chunk(self):
        """Verify tokenization and uniform sequence chunking."""
        mgr = CalibrationDatasetManager(CalibrationConfig())

        class MockTokenizer:
            def __call__(self, text: str, **kwargs):
                return {"input_ids": [101] * (len(text.split()) * 5)}

        corpus = ["def hello world test func"] * 20
        chunks = mgr.tokenize_and_chunk(corpus, MockTokenizer(), sequence_length=16, max_samples=4)
        assert len(chunks) == 4
        assert len(chunks[0]) == 16
