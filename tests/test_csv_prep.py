import os
import tempfile
import pandas as pd
import pytest
from mlbuddy.auto.data import load_csv


def test_load_csv_automatic_preprocessing():
    # Create a temporary CSV file with heterogeneous data and missing values
    data = {
        'id_col': [f'id_{i}' for i in range(100)], # High cardinality -> drop
        'constant_col': [42] * 100,                # Constant -> drop
        'numeric_col': [i * 1.5 for i in range(100)],
        'numeric_missing': [None] + [10.0] * 99,   # Missing numeric -> fill median
        'low_cardinality': ['A', 'B'] * 50,        # Low cardinality -> one-hot
        'med_cardinality': [f'cat_{i % 20}' for i in range(100)], # Medium -> label encode
        'target': ['Yes', 'No'] * 50               # Target string -> label encode
    }
    df = pd.DataFrame(data)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'test_prep.csv')
        df.to_csv(csv_path, index=False)
        
        # Test loading
        X_train, X_test, y_train, y_test = load_csv(csv_path, target_column='target')
        
        # Verify sizes and types
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        
        # Target classes encoded as 0 and 1
        assert set(y_train).issubset({0, 1})
