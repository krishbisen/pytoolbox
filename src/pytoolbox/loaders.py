import pandas as pd


class LoanDataLoader:
    """Load loan dataset from a CSV file."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.filepath)


class LoanPreprocessors:
    """Create engineered features for the loan dataset."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Credit_Score" in df.columns:
            df["Credit_Score_Sq"] = df["Credit_Score"] ** 2

        if "DTI_Ratio" in df.columns:
            df["DTI_Ratio_Sq"] =( df["DTI_Ratio"] ** 2).round(10)

        return df