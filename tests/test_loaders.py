import pytest
import pandas as pd
import numpy as np
from pytoolbox.loaders import LoanDataLoader, LoanPreprocessors

def test_loan_preprocessor_feature_engineering():
    """Test that quadratic feature engineering is applied correctly."""
    df = pd.DataFrame({
        'Credit_Score': [700, 650],
        'DTI_Ratio': [0.3, 0.4]
    })
    preprocessor = LoanPreprocessors()
    df_transformed = preprocessor.transform(df)
    
    assert 'Credit_Score_Sq' in df_transformed.columns
    assert 'DTI_Ratio_Sq' in df_transformed.columns
    assert df_transformed['Credit_Score_Sq'].iloc[0] == 490000
    assert df_transformed['DTI_Ratio_Sq'].iloc[1] == 0.16