"""Unit tests for DAG pipeline validation and topological sort."""

import pytest

from vipym.compression.pipeline import DAGCompressionPipeline
from vipym.compression.quantization.awq import AWQCompressionMethod
from vipym.compression.transforms.spinquant import SpinQuantTransformMethod
from vipym.core.exceptions import InvalidPipelineDAGError


def test_linear_dag_topological_sort():
    pipeline = DAGCompressionPipeline()
    m1 = SpinQuantTransformMethod()
    m2 = AWQCompressionMethod()

    pipeline.add_stage("spinquant", m1)
    pipeline.add_stage("awq", m2, dependencies=["spinquant"])

    order = pipeline.get_topological_order()
    assert order == ["spinquant", "awq"]


def test_dag_cycle_detection():
    pipeline = DAGCompressionPipeline()
    m1 = SpinQuantTransformMethod()
    m2 = AWQCompressionMethod()

    pipeline.add_stage("stage_a", m1, dependencies=["stage_b"])
    pipeline.add_stage("stage_b", m2, dependencies=["stage_a"])

    with pytest.raises(InvalidPipelineDAGError):
        pipeline.get_topological_order()
