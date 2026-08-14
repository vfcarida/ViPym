"""Contract tests verifying that all registered plugins adhere to ABC specifications."""

from vipym.compression.registry import CompressionRegistry
from vipym.evaluation.registry import EvaluationRegistry
from vipym.inference.registry import InferenceRegistry
from vipym.interfaces.compression import CompressionMethod
from vipym.interfaces.evaluation import EvaluationSuite
from vipym.interfaces.inference import InferenceBackend
from vipym.interfaces.model import ModelAdapter
from vipym.models.registry import ModelRegistry


def test_model_adapters_contract():
    for _name, adapter_cls in ModelRegistry.list_adapters().items():
        adapter = adapter_cls()
        assert isinstance(adapter, ModelAdapter)
        caps = adapter.get_capabilities()
        assert hasattr(caps, "supported_architectures")
        assert hasattr(caps, "supported_dtypes")


def test_compression_methods_contract():
    for _name, method_cls in CompressionRegistry.list_methods().items():
        method = method_cls()
        assert isinstance(method, CompressionMethod)
        assert hasattr(method, "name")
        caps = method.get_capabilities()
        assert hasattr(caps, "supports_moe")


def test_inference_backends_contract():
    for _name, backend_cls in InferenceRegistry.list_backends().items():
        backend = backend_cls()
        assert isinstance(backend, InferenceBackend)


def test_evaluation_suites_contract():
    for _name, suite_cls in EvaluationRegistry.list_suites().items():
        suite = suite_cls()
        assert isinstance(suite, EvaluationSuite)
        assert hasattr(suite, "name")
        assert hasattr(suite, "version")
        tasks = suite.load_tasks(limit=1)
        assert len(tasks) >= 1
