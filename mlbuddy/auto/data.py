import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load(X, y, test_size=0.2, scale=True, random_state=42):
    print("\n🔍 mlbuddy: Loading your data...\n")

    if not isinstance(X, np.ndarray):
        X = np.array(X)
        print("  ✓ Converted X to numpy array")

    if not isinstance(y, np.ndarray):
        y = np.array(y)
        print("  ✓ Converted y to numpy array")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"\n✗ Mismatch: X has {X.shape[0]} rows but y has {y.shape[0]} values.\n"
            f"  → Make sure X and y have the same number of samples."
        )

    if X.shape[0] < 10:
        print("  ⚠ Warning: Less than 10 samples. ML works better with more data.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  ✓ Split: {len(X_train)} training samples, {len(X_test)} test samples")

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("  ✓ Features scaled with StandardScaler (mean=0, std=1)")

    print(f"\n  📦 Data shape: {X_train.shape[1]} features, {len(set(y))} classes")
    print("\n✅ Data ready! Pass X_train, X_test, y_train, y_test to ml.train()\n")

    return X_train, X_test, y_train, y_test


def load_csv(filepath, target_column):
    """Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        target_column: Name of the column containing the target labels
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    print(f"\n📂 Loading CSV from: {filepath}\n")
    df = pd.read_csv(filepath)
    
    print(f"  ✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    
    if target_column not in df.columns:
        raise ValueError(
            f"✗ Target column '{target_column}' not found.\n"
            f"  Available columns: {list(df.columns)}"
        )
    
    # Extract target y and feature DataFrame
    y = df[target_column]
    X_df = df.drop(columns=[target_column]).copy()
    
    # 1. Handle target encoding if it's not numeric
    if not pd.api.types.is_numeric_dtype(y):
        if y.isnull().any():
            mode_val = y.mode().iloc[0] if not y.mode().empty else "Unknown"
            y = y.fillna(mode_val)
            print(f"  ⚠ Filled missing target values with mode: '{mode_val}'")
        
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        print(f"  ✓ Encoded target '{target_column}' classes: {list(le.classes_)}")
    else:
        if y.isnull().any():
            median_val = y.median()
            y = y.fillna(median_val)
            print(f"  ⚠ Filled missing target values with median: {median_val}")
        y = y.values
        
    # 2. Process features
    processed_cols = []
    
    for col in X_df.columns:
        col_data = X_df[col]
        
        # Check unique count
        unique_count = col_data.nunique(dropna=True)
        
        if unique_count <= 1:
            print(f"  ⚠ Dropped constant column '{col}' (has only {unique_count} unique value)")
            continue
            
        # Check missing values
        if col_data.isnull().any():
            if pd.api.types.is_numeric_dtype(col_data):
                fill_val = col_data.median()
                col_data = col_data.fillna(fill_val)
                print(f"  ⚠ Filled missing values in numeric column '{col}' with median: {fill_val}")
            else:
                fill_val = col_data.mode().iloc[0] if not col_data.mode().empty else "Unknown"
                col_data = col_data.fillna(fill_val)
                print(f"  ⚠ Filled missing values in categorical column '{col}' with mode: '{fill_val}'")
                
        # Check if numeric vs categorical/object
        if pd.api.types.is_numeric_dtype(col_data):
            processed_cols.append(pd.DataFrame({col: col_data}))
        else:
            col_data = col_data.astype(str)
            if unique_count > 100:
                print(f"  ⚠ Dropped high-cardinality text column '{col}' ({unique_count} unique values)")
                continue
            elif unique_count <= 15:
                dummies = pd.get_dummies(col_data, prefix=col, drop_first=True, dtype=float)
                processed_cols.append(dummies)
                print(f"  ✓ One-hot encoded '{col}' ({unique_count} categories -> {dummies.shape[1]} columns)")
            else:
                le_feat = LabelEncoder()
                encoded_feat = le_feat.fit_transform(col_data)
                processed_cols.append(pd.DataFrame({col: encoded_feat}))
                print(f"  ✓ Label encoded '{col}' ({unique_count} categories)")
                
    if not processed_cols:
        raise ValueError("✗ No feature columns remaining after preprocessing!")
        
    X_processed = pd.concat(processed_cols, axis=1)
    X = X_processed.values
    
    return load(X, y)