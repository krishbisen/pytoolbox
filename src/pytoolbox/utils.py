from abc import ABC, abstractmethod
from typing import List
import math

class BaseTransformer(ABC):
    """Abstract Base Class defining the structural contract for all data steps."""
    
    @abstractmethod
    def fit(self, data: List[float]) -> None:
        """Calculate internal parameters from training data."""
        pass
        
    @abstractmethod
    def transform(self, data: List[float]) -> List[float]:
        """Apply transformation math to the data based on calculated parameters."""
        pass

class DataStandardizer(BaseTransformer):
    """Encapsulates Z-score standardization: (x - mean) / std_dev"""
    
    def __init__(self) -> None:
        self.mean: float = 0.0
        self.std_dev: float = 1.0
        self._is_fitted: bool = False  # Encapsulated state variable

    def fit(self, data: List[float]) -> None:
        """Calculate the mean and standard deviation from the data collection."""
        if not data:
            raise ValueError("Cannot fit an empty dataset.")
            
        n = len(data)
        self.mean = sum(data) / n
        
        # Calculate variance: average of squared differences from the mean
        variance = sum((x - self.mean) ** 2 for x in data) / n
        
        # Standard deviation is the square root of variance
        self.std_dev = math.sqrt(variance) if variance > 0 else 1.0
        self._is_fitted = True

    def transform(self, data: List[float]) -> List[float]:
        """Apply Z-score transformation to scale data vectors."""
        if not self._is_fitted:
            raise RuntimeError("The standardizer must be fitted before transforming data.")
        if not data:
            return []
            
        # Z-score math: (x - mean) / std_dev
        return [(x - self.mean) / self.std_dev for x in data]    