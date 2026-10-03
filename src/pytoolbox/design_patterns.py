from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from threading import Lock
from typing import Any, Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ModelStrategy(ABC):
    """Strategy interface for interchangeable ML models."""

    @abstractmethod
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Sequence[Sequence[float]]) -> list[Any]:
        raise NotImplementedError


class LogisticRegressionStrategy(ModelStrategy):
    def __init__(self, **kwargs: Any) -> None:
        self.model = LogisticRegression(**kwargs)

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[Any]) -> None:
        self.model.fit(X, y)

    def predict(self, X: Sequence[Sequence[float]]) -> list[Any]:
        return self.model.predict(X).tolist()


class RandomForestStrategy(ModelStrategy):
    def __init__(self, **kwargs: Any) -> None:
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[Any]) -> None:
        self.model.fit(X, y)

    def predict(self, X: Sequence[Sequence[float]]) -> list[Any]:
        return self.model.predict(X).tolist()


class ModelFactory:
    """Factory for selecting a concrete model strategy by name."""

    _strategies = {
        "logistic_regression": LogisticRegressionStrategy,
        "random_forest": RandomForestStrategy,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> ModelStrategy:
        try:
            strategy = cls._strategies[name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown model strategy: {name}") from exc
        return strategy(**kwargs)


class TrainingObserver(ABC):
    """Observer interface for training lifecycle events."""

    @abstractmethod
    def update(self, event: str, details: dict[str, Any]) -> None:
        raise NotImplementedError


class TrainingSubject:
    """Publishes training lifecycle events to attached observers."""

    def __init__(self) -> None:
        self._observers: list[TrainingObserver] = []

    def attach(self, observer: TrainingObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: TrainingObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, event: str, **details: Any) -> None:
        for observer in self._observers:
            observer.update(event, details)


class LoggingObserver(TrainingObserver):
    def __init__(self, logger: Logger | None = None) -> None:
        self.logger = logger or getLogger(__name__)

    def update(self, event: str, details: dict[str, Any]) -> None:
        self.logger.info("training_event=%s details=%s", event, details)


class LoggerSingleton:
    """One shared logger provider for small applications."""

    _instance: LoggerSingleton | None = None
    _lock = Lock()

    def __new__(cls, name: str = "pytoolbox") -> LoggerSingleton:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance.logger = getLogger(name)
                cls._instance = instance
        return cls._instance


class PredictorAdapter:
    """Target interface expected by inference code."""

    def predict(self, X: Sequence[Sequence[float]]) -> list[Any]:
        raise NotImplementedError


class SklearnPredictorAdapter(PredictorAdapter):
    """Adapter that exposes a consistent list-returning interface for sklearn."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, X: Sequence[Sequence[float]]) -> list[Any]:
        return self.model.predict(X).tolist()


class TrainingPipeline:
    """Pipeline context that delegates model behavior to a strategy."""

    def __init__(self, strategy: ModelStrategy) -> None:
        self.strategy = strategy
        self.events = TrainingSubject()

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[Any]) -> None:
        self.events.notify("training_started", samples=len(X))
        self.strategy.fit(X, y)
        self.events.notify("training_completed", samples=len(X))

    def predict(self, X: Sequence[Sequence[float]]) -> list[Any]:
        return self.strategy.predict(X)
