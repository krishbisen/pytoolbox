# PyToolbox

[![Tests](https://github.com/krishbisen/pytoolbox/actions/workflows/tests.yml/badge.svg)](https://github.com/krishbisen/pytoolbox/actions/workflows/tests.yml)
![Coverage](https://img.shields.io/badge/coverage-92.95%25-brightgreen)

Reusable Python utilities developed during the engineering foundation phase of the AI & Robotics roadmap.

## Month 2 engineering work

This repository now contains the Month 2 engineering patterns required by the roadmap:

- SOLID-oriented separation of responsibilities
- Strategy pattern for interchangeable ML models
- Factory pattern for model creation
- Observer pattern for training lifecycle events
- Singleton pattern for shared logger access
- Adapter pattern for a common inference interface
- Abstract `BaseModel` and `BasePipeline` contracts
- Structured logging configuration
- Pytest coverage configuration

## Development setup

```powershell
python -m pip install -e ".[dev]"
```

## Tests

Run the complete suite:

```powershell
python -m pytest
```

Run tests with coverage:

```powershell
python -m pytest --cov=pytoolbox --cov-report=term-missing
```

**Verified engineering-set result:** 31 tests passed with **92.95% coverage**, above the roadmap's 80% target.

The GitHub Actions workflow enforces the 80% minimum on pushes and pull requests.

## Architecture

```text
Application
    |
    v
TrainingPipeline
    |
    +--> ModelStrategy
    |       +--> LogisticRegressionStrategy
    |       +--> RandomForestStrategy
    |
    +--> TrainingObserver
            +--> LoggingObserver

ModelFactory selects the strategy.
SklearnPredictorAdapter normalizes third-party model output.
BaseModel/BasePipeline define reusable contracts.
```

## CLI

```powershell
pytoolbox clamp 15 0 10
pytoolbox palindrome "Never odd or even"
pytoolbox chunk a b c d --size 2
```

## Repository

https://github.com/krishbisen/pytoolbox

## License

MIT
