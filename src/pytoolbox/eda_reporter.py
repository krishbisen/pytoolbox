import pandas as pd
import numpy as np
from typing import Dict


class EDAReporter:
    """Encapsulates dataset summary statistics and analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_summary_stats(self) -> Dict[str, dict]:
        return self.df.describe().to_dict()

    def detect_missing_values(self) -> pd.Series:
        return self.df.isnull().sum()

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        numeric_df = self.df.select_dtypes(include=[np.number])
        return numeric_df.corr()