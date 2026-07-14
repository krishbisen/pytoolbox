import pytest
from pytoolbox.utils import DataStandardizer

def test_data_standardizer_calculation():
    # Arrange: define a sample dataset
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    
    # Act: initialize, fit, and transform
    standardizer = DataStandardizer()
    standardizer.fit(data)
    transformed = standardizer.transform(data)
    
    # Assert: the mean of standardized data should be effectively 0.0
    mean_val = sum(transformed) / len(transformed)
    assert abs(mean_val) < 1e-5

def test_transform_before_fit_raises_error():
    # Arrange: un-fitted standardizer
    standardizer = DataStandardizer()
    
    # Assert: calling transform without fit should raise a RuntimeError
    with pytest.raises(RuntimeError):
        standardizer.transform([1.0, 2.0, 3.0])