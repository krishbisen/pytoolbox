import pytest

from pytoolbox.pipeline_core import BaseModel, BasePipeline


def test_base_classes_are_abstract():
    with pytest.raises(TypeError):
        BaseModel()
    with pytest.raises(TypeError):
        BasePipeline()
