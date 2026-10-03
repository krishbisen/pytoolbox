"""Reusable loan-classification example using the Strategy pattern."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .design_patterns import ModelFactory, ModelStrategy


class dataset_load:
    """Load a CSV dataset."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.filepath)


class data_prepocessors:
    """Perform the example feature-engineering step."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Credit_Score" in df.columns:
            df["Credit_Score_sq"] = df["Credit_Score"] ** 2
        if "DIT_Ratio" in df.columns:
            df["DIT_Ratio_sq"] = df["DIT_Ratio"] ** 2
        return df


class LoanModelPipeline:
    """Pipeline logic independent from the concrete ML model."""

    def __init__(self, strategy: ModelStrategy | None = None, model: Any | None = None):
        if strategy is not None and model is not None:
            raise ValueError("provide strategy or model, not both")

        if strategy is not None:
            self.strategy = strategy
        elif model is not None:
            self.strategy = ModelFactory.create(
                "logistic_regression",
                max_iter=1000,
            )
            self.strategy.model = model
        else:
            self.strategy = ModelFactory.create(
                "logistic_regression",
                max_iter=1000,
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.strategy.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        prediction = self.strategy.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, prediction),
            "report": classification_report(
                y_test,
                prediction,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, prediction),
        }
