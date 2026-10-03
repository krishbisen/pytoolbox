"""Reusable abstract base classes for ML models and pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

X = TypeVar("X")
Y = TypeVar("Y")


class BaseModel(ABC, Generic[X, Y]):
    """Minimum contract every reusable ML model wrapper must implement."""

    @abstractmethod
    def fit(self, X: X, y: Y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: X) -> Any:
        raise NotImplementedError


class BasePipeline(ABC, Generic[X, Y]):
    """Contract separating pipeline orchestration from model details."""

    @abstractmethod
    def preprocess(self, data: X) -> X:
        raise NotImplementedError

    @abstractmethod
    def train(self, data: X, target: Y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: X) -> Any:
        raise NotImplementedError
