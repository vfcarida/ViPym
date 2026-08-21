"""Integration test for distillation pipeline and synthetic dataset caching."""

from vipym.compression.distillation.logit_distill import TeacherSyntheticDataPipeline
from vipym.compression.distillation.response_distill import ResponseDistillationMethod


def test_teacher_synthetic_generation_and_caching():
    pipeline = TeacherSyntheticDataPipeline(teacher_backend=None, prompt_templates=["Template 1"])
    dataset = pipeline.generate_dataset(num_samples=10)
    assert len(dataset) == 10
    assert "instruction" in dataset[0]
    assert "response" in dataset[0]


def test_response_distillation_adapter_metadata():
    distiller = ResponseDistillationMethod(
        student_model_id="Qwen/Qwen2.5-Coder-1.5B",
        distillation_dataset="code_distill_10k",
    )
    assert "distill_response" in distiller.name
    caps = distiller.get_capabilities()
    assert caps.requires_training is True
