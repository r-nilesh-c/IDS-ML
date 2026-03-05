"""
Solutions for Lesson 1: Data Preprocessing Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


# Exercise 1: Load a Single Dataset
def load_single_dataset(path: str) -> pd.DataFrame:
    """
    Load a single CSV file with encoding error handling.
    
    Args:
        path: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Try multiple encodings
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            print(f"Successfully loaded with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not load {path} with any supported encoding")


# Exercise 2: Count Missing Values
def count_data_issues(df: pd.DataFrame) -> dict:
    """
    Count NaN, inf, and duplicate rows in DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with counts
    """
    # Count NaN values
    nan_count = df.isna().sum().sum()
    
    # Count infinite values (only in numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    
    # Count duplicate rows
    duplicate_count = df.duplicated().sum()
    
    return {
        'nan': nan_count,
        'inf': inf_count,
        'duplicates': duplicate_count
    }


# Exercise 3: Implement Simple Normalization
def minmax_normalize(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Apply min-max normalization.
    
    Formula: X_norm = (X - X_min) / (X_max - X_min)
    
    Args:
        X_train: Training data
        X_test: Test data
        
    Returns:
        Tuple of (X_train_normalized, X_test_normalized)
    """
    # Compute min and max from training data only
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    
    # Avoid division by zero
    range_vals = X_max - X_min
    range_vals[range_vals == 0] = 1.0
    
    # Normalize both sets using training statistics
    X_train_normalized = (X_train - X_min) / range_vals
    X_test_normalized = (X_test - X_min) / range_vals
    
    # Clip test values to [0, 1] (in case test has values outside training range)
    X_test_normalized = np.clip(X_test_normalized, 0, 1)
    
    return X_train_normalized, X_test_normalized


# Exercise 4: Analyze Feature Distributions
def analyze_feature_distributions(df: pd.DataFrame):
    """
    Analyze and visualize feature distributions.
    
    Args:
        df: Input DataFrame with numeric features
    """
    # Select only numeric columns (exclude Label)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'Label' in numeric_cols:
        numeric_cols = numeric_cols.drop('Label')
    
    # Compute skewness for each feature
    skewness = df[numeric_cols].skew().sort_values(ascending=False)
    
    print("Top 10 Most Skewed Features:")
    print(skewness.head(10))
    
    # Visualize top 5 most skewed features
    top_5_features = skewness.head(5).index
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_5_features):
        ax = axes[idx]
        
        # Plot histogram
        df[feature].hist(bins=50, ax=ax, edgecolor='black')
        ax.set_title(f'{feature}\nSkewness: {skewness[feature]:.2f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=150)
    print("\nVisualization saved to 'feature_distributions.png'")
    plt.show()


# Project: Build Your Own Preprocessor
class SimplePreprocessor:
    """
    Simplified preprocessing pipeline for learning purposes.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def load_and_clean(self, path: str) -> pd.DataFrame:
        """
        Load and clean a single dataset.
        
        Args:
            path: Path to CSV file
            
        Returns:
            Cleaned DataFrame
        """
        # Load dataset
        df = load_single_dataset(path)
        print(f"Loaded {len(df)} samples")
        
        # Handle label column variations
        label_variations = [' Label', 'Label', 'label', ' label']
        for col_name in label_variations:
            if col_name in df.columns:
                df = df.rename(columns={col_name: 'Label'})
                break
        
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"After removing duplicates: {len(df)} samples")
        
        # Remove NaN values
        df = df.dropna()
        print(f"After removing NaN: {len(df)} samples")
        
        # Remove infinite values (only check numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(df[numeric_cols]).any(axis=1)
        df = df[~inf_mask]
        print(f"After removing inf: {len(df)} samples")
        
        # Remove non-numeric features (except Label)
        non_numeric_cols = []
        for col in df.columns:
            if col != 'Label' and not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            df = df.drop(columns=non_numeric_cols)
            print(f"Removed {len(non_numeric_cols)} non-numeric features")
        
        return df
    
    def split_and_normalize(self, df: pd.DataFrame) -> dict:
        """
        Split into benign/attack and normalize.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary with X_train, X_test, y_test
        """
        # Separate benign and attack samples
        benign_mask = df['Label'].str.upper() == 'BENIGN'
        benign_df = df[benign_mask].copy()
        attack_df = df[~benign_mask].copy()
        
        print(f"Benign samples: {len(benign_df)}")
        print(f"Attack samples: {len(attack_df)}")
        
        # Extract features (drop Label column)
        benign_features = benign_df.drop(columns=['Label']).values
        attack_features = attack_df.drop(columns=['Label']).values
        
        # Split benign data into train and test
        X_train, X_test_benign = train_test_split(
            benign_features,
            test_size=0.2,
            random_state=42
        )
        
        # Combine benign and attack for test set
        X_test = np.vstack([X_test_benign, attack_features])
        y_test = np.hstack([
            np.zeros(len(X_test_benign)),  # 0 = benign
            np.ones(len(attack_features))   # 1 = attack
        ])
        
        # Normalize using StandardScaler
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)} (benign: {len(X_test_benign)}, attack: {len(attack_features)})")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test
        }


# Example usage
if __name__ == "__main__":
    # Test Exercise 1
    print("=" * 50)
    print("Exercise 1: Load Single Dataset")
    print("=" * 50)
    # df = load_single_dataset('dataset/cic-ids2017/Monday-WorkingHours.pcap_ISCX.csv')
    # print(f"Shape: {df.shape}")
    # print(f"Columns: {list(df.columns[:5])}...")
    
    # Test Exercise 2
    print("\n" + "=" * 50)
    print("Exercise 2: Count Data Issues")
    print("=" * 50)
    # issues = count_data_issues(df)
    # print(f"NaN values: {issues['nan']}")
    # print(f"Inf values: {issues['inf']}")
    # print(f"Duplicates: {issues['duplicates']}")
    
    # Test Exercise 3
    print("\n" + "=" * 50)
    print("Exercise 3: Min-Max Normalization")
    print("=" * 50)
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[2, 3], [7, 8]])
    X_train_norm, X_test_norm = minmax_normalize(X_train, X_test)
    print(f"Original train:\n{X_train}")
    print(f"Normalized train:\n{X_train_norm}")
    print(f"Normalized test:\n{X_test_norm}")
    
    # Test Project
    print("\n" + "=" * 50)
    print("Project: Simple Preprocessor")
    print("=" * 50)
    # preprocessor = SimplePreprocessor()
    # df = preprocessor.load_and_clean('dataset/cic-ids2017/Monday-WorkingHours.pcap_ISCX.csv')
    # data = preprocessor.split_and_normalize(df)
    # print(f"\nFinal shapes:")
    # print(f"X_train: {data['X_train'].shape}")
    # print(f"X_test: {data['X_test'].shape}")
    # print(f"y_test: {data['y_test'].shape}")
