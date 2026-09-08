"""Unit tests for plugin registries verifying decorator and direct registration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vipym.compression.registry import CompressionRegistry
from vipym.evaluation.registry import EvaluationRegistry
from vipym.inference.registry import InferenceRegistry
from vipym.interfaces.compression import CompressionMethod
from vipym.interfaces.evaluation import EvaluationSuite
from vipym.interfaces.inference import InferenceBackend
from vipym.interfaces.model import ModelAdapter
from vipym.models.registry import ModelRegistry


@pytest.fixture(autouse=True)
def clean_registries():
    comp_before = dict(CompressionRegistry._registry)
    eval_before = dict(EvaluationRegistry._registry)
    model_before = dict(ModelRegistry._registry)
    inf_before = dict(InferenceRegistry._registry)
    yield
    CompressionRegistry._registry = comp_before
    EvaluationRegistry._registry = eval_before
    ModelRegistry._registry = model_before
    InferenceRegistry._registry = inf_before


def test_compression_registry_decorator_and_direct():
    @CompressionRegistry.register("mock_compression_dec")
    class MockDec(CompressionMethod):
        @property
        def name(self) -> str:
            return "mock_dec"

        def get_capabilities(self):
            return MagicMock()

        def compress(self, *args, **kwargs):
            return MagicMock()

        def validate_applicability(self, metadata):
            return True

    assert "mock_compression_dec" in CompressionRegistry.list_methods()
    instance = CompressionRegistry.get("mock_compression_dec")
    assert isinstance(instance, MockDec)

    class MockDirect(CompressionMethod):
        @property
        def name(self) -> str:
            return "mock_direct"

        def get_capabilities(self):
            return MagicMock()

        def compress(self, *args, **kwargs):
            return MagicMock()

        def validate_applicability(self, metadata):
            return True

    CompressionRegistry.register("mock_compression_dir", MockDirect)
    assert "mock_compression_dir" in CompressionRegistry.list_methods()


def test_evaluation_registry_decorator_and_direct():
    @EvaluationRegistry.register("mock_eval_dec")
    class MockEvalDec(EvaluationSuite):
        @property
        def name(self) -> str:
            return "mock_eval_dec"

        @property
        def version(self) -> str:
            return "v1.0"

        def load_tasks(self, limit=None):
            return [MagicMock()]

        def format_prompt(self, task, tokenizer=None):
            return ""

        def evaluate_response(self, task, generated_text, sandbox_runner):
            return MagicMock()

    assert "mock_eval_dec" in EvaluationRegistry.list_suites()
    instance = EvaluationRegistry.get("mock_eval_dec")
    assert isinstance(instance, MockEvalDec)


def test_model_registry_decorator_and_direct():
    @ModelRegistry.register("mock_model_dec")
    class MockModelDec(ModelAdapter):
        def load(self, *args, **kwargs):
            pass

        def load_for_compression(self, *args, **kwargs):
            return MagicMock(), MagicMock()

        def inspect_metadata(self, *args, **kwargs):
            return MagicMock()

        def get_tokenizer(self, *args, **kwargs):
            return MagicMock()

        def get_metadata(self):
            return MagicMock()

        def get_capabilities(self):
            return MagicMock()

    assert "mock_model_dec" in ModelRegistry.list_adapters()
    instance = ModelRegistry.get("mock_model_dec")
    assert isinstance(instance, MockModelDec)


def test_inference_registry_decorator_and_direct():
    @InferenceRegistry.register("mock_backend_dec")
    class MockBackendDec(InferenceBackend):
        def start(self, *args, **kwargs):
            pass

        def stop(self):
            pass

        def generate(self, request):
            return MagicMock()

        async def generate_async(self, request):
            return MagicMock()

    instance = InferenceRegistry.get("mock_backend_dec")
    assert isinstance(instance, MockBackendDec)
