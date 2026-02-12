import pandas as pd
from src.preprocess import load_and_preprocess
import os

def test_preprocess():
    # Create a dummy CSV for testing
    data = """id,value
1,10
2,
3,30
"""
    with open('test_data.csv', 'w') as f:
        f.write(data)
    
    try:
        df = load_and_preprocess('test_data.csv')
        assert not df.empty
        assert df.isnull().sum().sum() == 0
        assert len(df) == 2
    finally:
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
