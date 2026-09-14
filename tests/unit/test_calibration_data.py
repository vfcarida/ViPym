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

    def test_ast_code_chunker_python_units(self):
        """Verify ASTCodeChunker extracts syntactically complete functions and classes."""
        from vipym.data.calibration import ASTCodeChunker

        code = """import sys

def compute_sum(a: int, b: int) -> int:
    return a + b

class MatrixProcessor:
    def __init__(self, size: int) -> None:
        self.size = size

async def fetch_data(url: str) -> str:
    return "ok"
"""
        units = ASTCodeChunker.extract_python_units(code)
        assert len(units) >= 3
        assert any("def compute_sum" in u for u in units)
        assert any("class MatrixProcessor" in u for u in units)
        assert any("async def fetch_data" in u for u in units)

    def test_ast_code_chunker_fallback_on_syntax_error(self):
        """Verify ASTCodeChunker gracefully falls back to indented blocks on syntax errors."""
        from vipym.data.calibration import ASTCodeChunker

        broken_code = """func main() {
    println("Hello Go")
}

func helper() {
    return 1
}
"""
        units = ASTCodeChunker.extract_python_units(broken_code)
        assert len(units) >= 1
        assert any("func " in u for u in units)

    def test_tokenize_and_chunk_ast_aware(self):
        """Verify AST-aware chunking preserves unit boundaries and inserts delimiter tokens."""
        mgr = CalibrationDatasetManager(CalibrationConfig(ast_aware=True, sequence_length=64))

        class MockTokenizer:
            eos_token_id = 999

            def __call__(self, text: str, **kwargs):
                return {"input_ids": [10, 20, 30]}

        corpus = [
            "def fn1(): pass\ndef fn2(): pass\ndef fn3(): pass",
        ]
        chunks = mgr.tokenize_and_chunk(corpus, MockTokenizer(), sequence_length=64, max_samples=2)
        assert len(chunks) >= 1
        # Should contain delimiter eos_token_id 999 between packed functions
        assert 999 in chunks[0]

    def test_multi_domain_calibration_mixer_proportions(self):
        """Verify MultiDomainCalibrationMixer achieves requested proportions."""
        from vipym.data.calibration import MultiDomainCalibrationMixer

        code_samples = ["def f(): pass", "def g(): pass", "def h(): pass"]
        text_samples = ["Documentation text 1.", "Documentation text 2."]

        corpora = {"code": code_samples, "text": text_samples}
        weights = {"code": 0.70, "text": 0.30}

        mixed = MultiDomainCalibrationMixer.mix_corpora(
            corpora=corpora, weights=weights, total_samples=100, seed=123
        )

        assert len(mixed) == 100
        code_count = sum(1 for s in mixed if "def " in s)
        text_count = sum(1 for s in mixed if "Documentation" in s)

        # Proportions should be approximately 70% and 30% (+/- 5%)
        assert 65 <= code_count <= 75
        assert 25 <= text_count <= 35

    def test_multi_domain_calibration_mixer_deterministic_seed(self):
        """Verify identical seed yields identical sequence of mixed samples."""
        from vipym.data.calibration import MultiDomainCalibrationMixer

        corpora = {
            "code": ["def a(): pass", "def b(): pass"],
            "math": ["formula 1", "formula 2"],
        }
        mix1 = MultiDomainCalibrationMixer.mix_corpora(corpora, total_samples=20, seed=42)
        mix2 = MultiDomainCalibrationMixer.mix_corpora(corpora, total_samples=20, seed=42)
        assert mix1 == mix2

    def test_multi_domain_config_and_manager_integration(self):
        """Verify CalibrationDatasetManager produces mixed corpus via DomainMixSpec."""
        from vipym.data.calibration import DomainMixSpec

        spec_code = DomainMixSpec(
            domain_name="code",
            weight=0.8,
            samples=["def solve_equation(): return 0", "def optimize(): return 1"],
        )
        spec_text = DomainMixSpec(
            domain_name="text",
            weight=0.2,
            samples=["Software architecture explanation.", "System performance guide."],
        )

        cfg = CalibrationConfig(
            domain_mix=[spec_code, spec_text],
            num_samples=10,
            seed=42,
            purge_contaminated=False,
        )
        mgr = CalibrationDatasetManager(cfg)
        corpus = mgr.get_multi_domain_corpus()

        assert len(corpus) == 10
        assert any("def " in s for s in corpus)
        assert any("explanation" in s or "guide" in s for s in corpus)
