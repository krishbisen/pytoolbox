from pytoolbox.design_patterns import (
    LoggerSingleton,
    LogisticRegressionStrategy,
    ModelFactory,
    RandomForestStrategy,
    SklearnPredictorAdapter,
    TrainingObserver,
    TrainingPipeline,
)


def test_factory_creates_strategies():
    assert isinstance(
        ModelFactory.create("logistic_regression"),
        LogisticRegressionStrategy,
    )
    assert isinstance(ModelFactory.create("random_forest"), RandomForestStrategy)


def test_factory_rejects_unknown_strategy():
    try:
        ModelFactory.create("unknown")
    except ValueError as exc:
        assert "Unknown model strategy" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_strategy_pipeline_can_swap_models():
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]

    pipeline = TrainingPipeline(ModelFactory.create("logistic_regression"))
    pipeline.fit(X, y)
    assert pipeline.predict([[2.5]]) == [1]

    pipeline.strategy = ModelFactory.create(
        "random_forest",
        random_state=42,
        n_estimators=10,
    )
    pipeline.fit(X, y)
    assert pipeline.predict([[2.5]]) in ([0], [1])


def test_observer_receives_events():
    class Capture(TrainingObserver):
        def __init__(self):
            self.events = []

        def update(self, event, details):
            self.events.append((event, details))

    observer = Capture()
    pipeline = TrainingPipeline(LogisticRegressionStrategy())
    pipeline.events.attach(observer)
    pipeline.fit([[0], [1], [2], [3]], [0, 0, 1, 1])

    assert [event[0] for event in observer.events] == [
        "training_started",
        "training_completed",
    ]


def test_singleton_returns_same_provider():
    assert LoggerSingleton("a") is LoggerSingleton("b")


def test_adapter_wraps_sklearn_model():
    strategy = LogisticRegressionStrategy()
    strategy.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    adapter = SklearnPredictorAdapter(strategy.model)
    assert adapter.predict([[2.5]]) == [1]
