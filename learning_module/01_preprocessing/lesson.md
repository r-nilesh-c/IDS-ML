# Lesson 1: Data Preprocessing Pipeline

## Learning Objectives

By the end of this lesson, you will:
- Understand why preprocessing is critical for IDS
- Load and merge multiple network flow datasets
- Clean data (handle NaN, inf, duplicates)
- Normalize features using StandardScaler
- Split data for benign-only training

## Why Preprocessing Matters

In intrusion detection, preprocessing determines your model's success. Poor preprocessing leads to:
- **Data leakage**: Test data influencing training
- **Scale issues**: Features with different ranges dominating learning
- **Noise**: Corrupted data causing false patterns

## The CIC-IDS Dataset

CIC-IDS2017/2018 contains network flow features extracted from packet captures:
- **Flow features**: Duration, packet counts, byte counts
- **Statistical features**: Mean, std, min, max of packet sizes
- **Timing features**: Inter-arrival times, idle times
- **Labels**: "BENIGN" or specific attack types

## Architecture Overview

```
PreprocessingPipeline
├── load_datasets()      # Load and merge CSV files
├── clean_data()         # Remove invalid samples
├── split_benign_attack() # Separate benign from attacks
└── normalize_and_split() # Scale features and create splits
```

## Deep Dive: Loading Datasets

### Challenge: Multiple Files, Inconsistent Formats

```python
def load_datasets(self, paths: List[str]) -> pd.DataFrame:
    """Load and merge multiple CSV datasets."""
```

**Key Concepts:**

1. **Encoding Handling**: CSV files may use different encodings
   ```python
   for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
       try:
           df = pd.read_csv(path, encoding=encoding)
           break
       except UnicodeDecodeError:
           continue
   ```

2. **Label Column Variations**: Different datasets use " Label", "Label", "label"
   ```python
   label_variations = [' Label', 'Label', 'label', ' label']
   for col_name in label_variations:
       if col_name in df.columns:
           df = df.rename(columns={col_name: 'Label'})
   ```

3. **Error Handling**: Graceful failures with informative messages
   ```python
   if not os.path.exists(path):
       raise FileNotFoundError(f"Dataset file not found: {path}")
   ```

**Why This Matters**: Real-world datasets are messy. Robust loading prevents silent failures.

## Deep Dive: Data Cleaning

### The Four Horsemen of Bad Data

```python
def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, NaN, inf values, and non-numeric features."""
```

1. **Duplicates**: Exact row copies that bias training
   ```python
   df_no_duplicates = df.drop_duplicates()
   ```

2. **NaN Values**: Missing data that breaks calculations
   ```python
   df_no_nan = df_no_duplicates.dropna()
   ```

3. **Infinite Values**: Division by zero or overflow errors
   ```python
   numeric_cols = df.select_dtypes(include=[np.number]).columns
   inf_mask = np.isinf(df[numeric_cols]).any(axis=1)
   df_no_inf = df[~inf_mask]
   ```

4. **Non-Numeric Features**: Strings that can't be used in ML models
   ```python
   for col in df.columns:
       if col != 'Label' and not pd.api.types.is_numeric_dtype(df[col]):
           non_numeric_cols.append(col)
   df_cleaned = df.drop(columns=non_numeric_cols)
   ```

**Critical Decision**: We remove problematic rows rather than imputing. Why?
- Imputation can introduce artificial patterns
- For IDS, we prefer clean data over maximum data
- Benign traffic is abundant, so we can afford to be strict

## Deep Dive: Benign-Only Training

### Why Train Only on Benign Traffic?

Traditional supervised learning requires labeled attack samples. But:
- **Zero-day attacks**: No training data exists for new attacks
- **Attack evolution**: Attackers constantly change tactics
- **Imbalanced data**: Attacks are rare in real networks

**Solution**: Train only on benign traffic. The model learns "normal" behavior and flags deviations as anomalies.

```python
def split_benign_attack(self, df: pd.DataFrame) -> tuple:
    """Separate benign and attack samples."""
    benign_mask = df['Label'].str.upper() == 'BENIGN'
    benign_df = df[benign_mask].copy()
    attack_df = df[~benign_mask].copy()
    return benign_df, attack_df
```

## Deep Dive: Normalization

### Why Normalize?

Network features have vastly different scales:
- Packet count: 1-10,000
- Duration: 0.001-1000 seconds
- Byte count: 50-1,000,000

Without normalization, large-scale features dominate learning.

### StandardScaler: The Math

```python
X_normalized = (X - mean) / std
```

After scaling:
- Mean ≈ 0
- Standard deviation ≈ 1
- All features contribute equally

### Critical Rule: Fit on Training Only

```python
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train_benign)  # Fit here
X_val_normalized = scaler.transform(X_val_benign)          # Only transform
X_test_normalized = scaler.transform(X_test)               # Only transform
```

**Why?** Fitting on test data causes data leakage - the model "sees" test statistics during training.

## Deep Dive: Data Splitting Strategy

### The Three-Way Split

```
All Data
├── Train (benign only) - 56% of all data
├── Validation (benign only) - 14% of all data
└── Test (benign + attacks) - 30% of all data
```

**Implementation:**
1. Split all data into train_val (70%) and test (30%) with stratification
2. Extract only benign samples from train_val
3. Split benign samples into train (80%) and validation (20%)

**Why Stratified Sampling?**
```python
X_train_val, X_test, y_train_val, y_test = train_test_split(
    all_features, all_labels,
    test_size=0.3,
    stratify=binary_labels,  # Maintains attack type distribution
    random_state=42
)
```

Stratification ensures test set has representative attack types.

## Exercises

### Exercise 1: Load a Single Dataset (Easy)
```python
# TODO: Implement a function that loads one CSV file
# Handle encoding errors gracefully
# Return a pandas DataFrame

def load_single_dataset(path: str) -> pd.DataFrame:
    # Your code here
    pass

# Test with: dataset/cic-ids2017/Monday-WorkingHours.pcap_ISCX.csv
```

### Exercise 2: Count Missing Values (Medium)
```python
# TODO: Write a function that counts NaN, inf, and duplicate rows
# Return a dictionary with counts

def count_data_issues(df: pd.DataFrame) -> dict:
    # Your code here
    # Return: {'nan': count, 'inf': count, 'duplicates': count}
    pass
```

### Exercise 3: Implement Simple Normalization (Medium)
```python
# TODO: Implement min-max normalization (alternative to StandardScaler)
# Formula: X_norm = (X - X_min) / (X_max - X_min)

def minmax_normalize(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    # Your code here
    # Return: (X_train_normalized, X_test_normalized)
    pass
```

### Exercise 4: Analyze Feature Distributions (Hard)
```python
# TODO: Load a dataset and analyze feature distributions
# Identify features with high skewness or outliers
# Visualize top 5 most skewed features

def analyze_feature_distributions(df: pd.DataFrame):
    # Your code here
    # Hint: Use df.skew() and matplotlib
    pass
```

## Quiz

1. **Why do we fit StandardScaler only on training data?**
   - A) To save computation time
   - B) To prevent data leakage
   - C) To make validation faster
   - D) It doesn't matter

2. **What happens if we don't remove infinite values?**
   - A) Model training will fail
   - B) Model will learn incorrect patterns
   - C) Predictions will be NaN
   - D) All of the above

3. **Why use stratified sampling for the test set?**
   - A) To ensure all attack types are represented
   - B) To make the test set larger
   - C) To speed up evaluation
   - D) To reduce memory usage

4. **In benign-only training, what do we use for validation?**
   - A) Attack samples
   - B) Benign samples
   - C) Mixed samples
   - D) No validation needed

## Project: Build Your Own Preprocessor

**Goal**: Implement a simplified preprocessing pipeline

**Requirements**:
1. Load one CIC-IDS CSV file
2. Remove NaN and infinite values
3. Separate benign and attack samples
4. Normalize features using StandardScaler
5. Create train/test split (80/20)

**Starter Code**:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SimplePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def load_and_clean(self, path: str) -> pd.DataFrame:
        # TODO: Implement
        pass
    
    def split_and_normalize(self, df: pd.DataFrame) -> dict:
        # TODO: Implement
        # Return: {'X_train': ..., 'X_test': ..., 'y_test': ...}
        pass

# Test your implementation
preprocessor = SimplePreprocessor()
df = preprocessor.load_and_clean('dataset/cic-ids2017/Monday-WorkingHours.pcap_ISCX.csv')
data = preprocessor.split_and_normalize(df)
print(f"Training samples: {len(data['X_train'])}")
print(f"Test samples: {len(data['X_test'])}")
```

## Common Mistakes

1. **Fitting scaler on test data**: Always fit only on training data
2. **Forgetting to handle label column**: Exclude 'Label' from normalization
3. **Not checking for infinite values**: They break StandardScaler
4. **Using all data for training**: Remember benign-only training!

## Key Takeaways

✅ Preprocessing is 50% of IDS development effort
✅ Robust error handling prevents silent failures
✅ Benign-only training enables zero-day detection
✅ Normalization is critical for neural networks
✅ Data leakage is the #1 mistake to avoid

## Next Lesson

In Lesson 2, we'll build the Autoencoder architecture and understand reconstruction-based anomaly detection.

**Preview**: Why does reconstruction error indicate anomalies? How do we choose the encoding dimension? What's the role of dropout?

Continue to `02_autoencoder/lesson.md` →
