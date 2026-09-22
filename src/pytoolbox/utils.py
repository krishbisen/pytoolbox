"""Reusable utilities for PyToolbox."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from collections.abc import Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Return value constrained to the inclusive range."""
    if minimum > maximum:
        raise ValueError("minimum cannot be greater than maximum")
    return max(minimum, min(value, maximum))


def is_palindrome(text: str) -> bool:
    """Check whether text is a palindrome, ignoring spaces and case."""
    normalized = "".join(text.split()).casefold()
    return normalized == normalized[::-1]


def chunked(items: Sequence[T], size: int) -> Iterable[list[T]]:
    """Yield a sequence in consecutive chunks of size."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


class BaseTransformer(ABC):
    """Abstract contract for fit/transform style data utilities."""

    @abstractmethod
    def fit(self, data: Sequence[float]) -> None:
        """Learn parameters from data."""

    @abstractmethod
    def transform(self, data: Sequence[float]) -> list[float]:
        """Transform data using learned parameters."""


class DataStandardizer(BaseTransformer):
    """Z-score standardizer implemented without third-party dependencies."""

    def __init__(self) -> None:
        self.mean: float = 0.0
        self.std_dev: float = 1.0
        self._is_fitted = False

    def fit(self, data: Sequence[float]) -> None:
        if not data:
            raise ValueError("cannot fit an empty dataset")
        self.mean = sum(data) / len(data)
        variance = sum((x - self.mean) ** 2 for x in data) / len(data)
        self.std_dev = math.sqrt(variance) if variance > 0 else 1.0
        self._is_fitted = True

    def transform(self, data: Sequence[float]) -> list[float]:
        if not self._is_fitted:
            raise RuntimeError("standardizer must be fitted before transforming")
        return [(x - self.mean) / self.std_dev for x in data]
