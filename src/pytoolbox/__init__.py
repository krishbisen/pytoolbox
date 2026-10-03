"""Public API for PyToolbox."""

from .design_patterns import (
    LoggerSingleton,
    LogisticRegressionStrategy,
    ModelFactory,
    ModelStrategy,
    RandomForestStrategy,
    SklearnPredictorAdapter,
    TrainingObserver,
    TrainingPipeline,
)
from .pipeline_core import BaseModel, BasePipeline
from .utils import BaseTransformer, DataStandardizer, chunked, clamp, is_palindrome

__all__ = [
    "BaseModel",
    "BasePipeline",
    "BaseTransformer",
    "DataStandardizer",
    "LoggerSingleton",
    "LogisticRegressionStrategy",
    "ModelFactory",
    "ModelStrategy",
    "RandomForestStrategy",
    "SklearnPredictorAdapter",
    "TrainingObserver",
    "TrainingPipeline",
    "chunked",
    "clamp",
    "is_palindrome",
]

__version__ = "0.2.0"
