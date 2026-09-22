import numpy as np
import pandas as pd

from pytoolbox.logistic_regression import (
    dataset_load,
    data_prepocessors,
    LoanModelPipeline,
)


def test_dataset_load(tmp_path):
    data = pd.DataFrame({
        "Credit_Score": [700, 600, 750],
        "DIT_Ratio": [0.2, 0.4, 0.1],
        "Loan_Status": [1, 0, 1],
    })

    csv_file = tmp_path / "loan.csv"
    data.to_csv(csv_file, index=False)

    loader = dataset_load(str(csv_file))
    result = loader.load_data()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result.columns) == [
        "Credit_Score",
        "DIT_Ratio",
        "Loan_Status",
    ]


def test_data_preprocessor():
    data = pd.DataFrame({
        "Credit_Score": [700, 600],
        "DIT_Ratio": [0.2, 0.4],
    })

    processor = data_prepocessors()
    result = processor.transform(data)

    assert "Credit_Score_sq" in result.columns
    assert "DIT_Ratio_sq" in result.columns

    assert result["Credit_Score_sq"].tolist() == [
        700 ** 2,
        600 ** 2,
    ]

    assert result["DIT_Ratio_sq"].tolist() == [
        0.2 ** 2,
        0.4 ** 2,
    ]


def test_loan_model_pipeline():
    X_train = np.array([
        [700, 0.2],
        [600, 0.5],
        [750, 0.1],
        [580, 0.6],
    ])

    y_train = np.array([1, 0, 1, 0])

    X_test = np.array([
        [720, 0.2],
        [590, 0.5],
    ])

    y_test = np.array([1, 0])

    pipeline = LoanModelPipeline()

    pipeline.train(X_train, y_train)

    result = pipeline.evaluate(X_test, y_test)

    assert "accuracy" in result
    assert "report" in result
    assert "confusion_matrix" in result

    assert 0 <= result["accuracy"] <= 1
    assert result["confusion_matrix"].shape == (2, 2)