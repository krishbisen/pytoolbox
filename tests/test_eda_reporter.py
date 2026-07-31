import pytest
import pandas as pd
from pytoolbox.eda_reporter import EDAReporter

def test_eda_summary_stats():
    """Test summary statistics generation dictionary structure."""
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    reporter = EDAReporter(df)
    stats = reporter.get_summary_stats()
    
    assert 'A' in stats
    assert 'B' in stats
    assert stats['A']['count'] == 5.0

def test_missing_value_detection():
    """Test missing value count series output."""
    df = pd.DataFrame({
        'A': [1, None, 3],
        'B': [None, None, 5]
    })
    reporter = EDAReporter(df)
    missing = reporter.detect_missing_values()
    
    assert missing['A'] == 1
    assert missing['B'] == 2