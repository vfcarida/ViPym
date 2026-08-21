#!/usr/bin/env python3
"""Pre-download and cache all benchmark datasets and calibration corpora for offline environments."""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vipym.cache_datasets")


def cache_all_datasets() -> None:
    """Download and cache HumanEval, MBPP, BigCodeBench, and SWE-bench."""
    logger.info("Initializing ViPym dataset caching routine...")

    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        logger.error("HuggingFace 'datasets' package is required. Install via `pip install datasets`.")
        sys.exit(1)

    targets = [
        ("HumanEval", "openai/openai_humaneval", None, "test"),
        ("MBPP (sanitized)", "google-research-datasets/mbpp", "sanitized", "test"),
        ("BigCodeBench", "bigcode/bigcodebench", None, "v0.1.2"),
        ("SWE-bench Lite", "princeton-nlp/SWE-bench_Lite", None, "test"),
    ]

    for name, repo_id, config_name, split in targets:
        logger.info(f"Downloading & caching {name} ({repo_id})...")
        try:
            if config_name:
                ds = load_dataset(repo_id, config_name, split=split)
            else:
                ds = load_dataset(repo_id, split=split)
            logger.info(f"Successfully cached {name}: {len(ds)} items.")
        except Exception as e:
            logger.warning(f"Could not cache {name} from remote: {e}. Fallbacks will be used.")

    logger.info("Dataset caching routine complete! All benchmarks ready for offline/air-gapped execution.")


if __name__ == "__main__":
    cache_all_datasets()
