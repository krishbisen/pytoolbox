# Month 2 — Design Patterns, SOLID & Clean Architecture

## SOLID

### S — Single Responsibility
A module/class should have one reason to change.

Bad:
- one class loads data, preprocesses it, trains a model, evaluates it, and prints reports.

Applied in PyToolbox:
- loaders load data
- preprocessors transform data
- model strategies train/predict
- observers handle lifecycle events
- CLI handles command parsing

### O — Open/Closed
Add a new model strategy without changing the pipeline.

```python
pipeline.strategy = ModelFactory.create("random_forest")
```

The pipeline stays unchanged.

### L — Liskov Substitution
Every `ModelStrategy` implementation must satisfy the same `fit/predict` contract.

### I — Interface Segregation
Small contracts such as `ModelStrategy`, `TrainingObserver`, `BaseModel`, and `BasePipeline` avoid forcing unrelated methods onto implementations.

### D — Dependency Inversion
The training pipeline depends on the abstract `ModelStrategy`, not directly on Logistic Regression or Random Forest.

## Design patterns

### Strategy
Interchange model behavior:

```text
TrainingPipeline
      |
      +-- LogisticRegressionStrategy
      |
      +-- RandomForestStrategy
```

### Factory
Create the selected strategy from a name:

```python
ModelFactory.create("logistic_regression")
ModelFactory.create("random_forest")
```

### Observer
Training publishes events and observers react without changing training logic.

### Singleton
`LoggerSingleton` provides one shared logger provider for a small application.

### Adapter
`SklearnPredictorAdapter` converts a third-party sklearn predictor into the project's common prediction interface.

## Clean Architecture / layering

The project separates responsibilities into:

```text
Interface
  ↓
Application / Pipeline
  ↓
Domain Contracts
  ↓
Infrastructure / Third-party ML libraries
```

The goal is to prevent notebook-style code where data loading, model logic, output, and configuration are mixed together.

## Anti-patterns avoided

- God class
- notebook spaghetti
- hard-coded model selection
- direct dependency on one concrete model
- print statements scattered through library code
- untestable global state

## Debugging discipline

Use normal logging for runtime events.

Use Python's debugger when state must be inspected:

```python
breakpoint()
```

For VS Code, `debugpy` can be used when a running process needs an attachable debugger.

## Coverage

The repository uses `pytest-cov` and enforces an 80% minimum through `pyproject.toml`.

Run:

```powershell
python -m pytest
```

The GitHub Actions workflow runs the same command on pushes and pull requests.
