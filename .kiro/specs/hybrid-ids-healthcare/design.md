# Design Document: Hybrid Anomaly-Based IDS for Healthcare Networks

## Overview

The Hybrid Anomaly-Based Intrusion Detection System combines deep learning (Autoencoder) and classical machine learning (Isolation Forest) to detect network intrusions in healthcare environments. The system is designed to detect known attacks, variations of known attacks, and zero-day threats by learning exclusively from benign traffic patterns.

The architecture follows a pipeline approach: raw network flow data is preprocessed, passed through two parallel anomaly detection modules, their outputs are fused using a weighted combination, and a dynamic threshold determines whether traffic is anomalous. The system prioritizes low false positive rates to minimize alert fatigue in clinical settings while maintaining high sensitivity to actual attacks.

Key design principles:
- Train only on benign traffic to enable true anomaly detection
- Use complementary detection mechanisms (reconstruction-based and isolation-based)
- Apply dynamic thresholding based on benign validation distribution
- Maintain modularity for easy testing, debugging, and extension
- Optimize for healthcare constraints (low FPR, near real-time inference, auditability)

## Architecture

The system follows a modular pipeline architecture with five main stages:

```
┌─────────────────┐
│  Raw Network    │
│  Flow Data      │
│ (CIC-IDS 2017/  │
│  CIC-IDS 2018)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Preprocessing Pipeline            │
│  - Load & Merge Datasets            │
│  - Remove Duplicates/NaN/Inf        │
│  - Drop Non-Numeric Features        │
│  - StandardScaler Normalization     │
│  - Stratified Train/Test Split      │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Training Phase (Benign Only)      │
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │ Autoencoder  │  │  Isolation  │ │
│  │   Training   │  │   Forest    │ │
│  │              │  │  Training   │ │
│  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Inference Phase (All Traffic)     │
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │ Autoencoder  │  │  Isolation  │ │
│  │ Reconstruct  │  │   Forest    │ │
│  │    Error     │  │   Score     │ │
│  └──────┬───────┘  └──────┬──────┘ │
│         │                 │        │
│         └────────┬────────┘        │
│                  ▼                 │
│         ┌────────────────┐         │
│         │ Fusion Module  │         │
│         │ (Weighted Avg) │         │
│         └────────┬───────┘         │
│                  ▼                 │
│         ┌────────────────┐         │
│         │   Threshold    │         │
│         │   (Percentile) │         │
│         └────────┬───────┘         │
└──────────────────┼─────────────────┘
                   ▼
         ┌─────────────────┐
         │ Anomaly Decision│
         │  (Binary: 0/1)  │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │ Healthcare Alert│
         │     System      │
         │  - Logging      │
         │  - Reporting    │
         └─────────────────┘
```

### Component Responsibilities

1. **Preprocessing Pipeline**: Data loading, cleaning, normalization, and splitting
2. **Autoencoder Module**: Deep learning-based reconstruction error computation
3. **Isolation Forest Module**: Tree-based anomaly scoring
4. **Fusion Module**: Score normalization and weighted combination
5. **Thresholding Module**: Dynamic threshold computation and classification
6. **Healthcare Alert System**: Logging, reporting, and audit trail generation

## Components and Interfaces

### 1. Preprocessing Pipeline

**Purpose**: Transform raw CSV datasets into clean, normalized feature matrices suitable for anomaly detection.

**Interface**:
```python
class PreprocessingPipeline:
    def __init__(self, config: Dict):
        """
        Initialize preprocessing pipeline with configuration.
        
        Args:
            config: Dictionary containing paths, random seed, test split ratio
        """
        
    def load_datasets(self, paths: List[str]) -> pd.DataFrame:
        """
        Load and merge multiple CSV datasets.
        
        Args:
            paths: List of file paths to CIC-IDS datasets
            
        Returns:
            Merged DataFrame with all samples
        """
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates, NaN, inf values, and non-numeric features.
        
        Args:
            df: Raw merged DataFrame
            
        Returns:
            Cleaned DataFrame with only numeric features
        """
        
    def split_benign_attack(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate benign and attack samples based on label column.
        
        Args:
            df: Cleaned DataFrame with label column
            
        Returns:
            Tuple of (benign_df, attack_df)
        """
        
    def normalize_and_split(self, benign_df: pd.DataFrame, attack_df: pd.DataFrame) -> Dict:
        """
        Normalize features and create train/validation/test splits.
        
        Args:
            benign_df: DataFrame containing only benign samples
            attack_df: DataFrame containing only attack samples
            
        Returns:
            Dictionary containing:
                - X_train_benign: Training features (benign only)
                - X_val_benign: Validation features (benign only)
                - X_test: Test features (benign + attacks)
                - y_test: Test labels
                - scaler: Fitted StandardScaler object
        """
```

**Implementation Details**:
- Use pandas for data loading and manipulation
- Label column typically named "Label" or " Label" (with space)
- Benign samples identified by label "BENIGN" (case-insensitive)
- StandardScaler fitted only on training benign data
- Stratified split ensures attack categories are proportionally represented in test set
- Validation set: 20% of benign training data
- Test set: 30% of all data (benign + attacks)

### 2. Autoencoder Module

**Purpose**: Learn compressed representation of benign traffic and compute reconstruction error as anomaly indicator.

**Interface**:
```python
class AutoencoderDetector:
    def __init__(self, input_dim: int, config: Dict):
        """
        Initialize autoencoder architecture.
        
        Args:
            input_dim: Number of input features
            config: Dictionary with encoding_dim, learning_rate, epochs, batch_size
        """
        
    def build_model(self) -> tf.keras.Model:
        """
        Construct encoder-decoder architecture.
        
        Returns:
            Compiled Keras model
        """
        
    def train(self, X_train: np.ndarray, X_val: np.ndarray) -> History:
        """
        Train autoencoder on benign traffic only.
        
        Args:
            X_train: Benign training samples
            X_val: Benign validation samples
            
        Returns:
            Training history object
        """
        
    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute MSE between input and reconstruction.
        
        Args:
            X: Input samples (any traffic)
            
        Returns:
            Array of reconstruction errors (one per sample)
        """
```

**Architecture Details**:
- **Encoder**: Input → Dense(encoding_dim * 2, relu) → Dense(encoding_dim, relu)
- **Decoder**: Dense(encoding_dim * 2, relu) → Dense(input_dim, sigmoid)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam with configurable learning rate (default: 0.001)
- **Regularization**: Optional dropout layers (0.2) to prevent overfitting
- **Early Stopping**: Monitor validation loss with patience=10
- **Batch Size**: 256 (configurable)
- **Epochs**: 50-100 (configurable)

**Reconstruction Error Computation**:
```
reconstruction_error = mean((X - X_reconstructed)^2, axis=1)
```

### 3. Isolation Forest Module

**Purpose**: Identify anomalies by measuring how easily samples can be isolated in feature space.

**Interface**:
```python
class IsolationForestDetector:
    def __init__(self, config: Dict):
        """
        Initialize Isolation Forest.
        
        Args:
            config: Dictionary with n_estimators, max_samples, contamination, random_state
        """
        
    def train(self, X_train: np.ndarray) -> None:
        """
        Train Isolation Forest on benign traffic only.
        
        Args:
            X_train: Benign training samples
        """
        
    def compute_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (negative of decision_function).
        
        Args:
            X: Input samples (any traffic)
            
        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
```

**Configuration Details**:
- **n_estimators**: 100 (number of isolation trees)
- **max_samples**: 256 or 'auto' (subsample size for each tree)
- **contamination**: 'auto' (not used during training since we have only benign data)
- **random_state**: Set for reproducibility
- **n_jobs**: -1 (use all CPU cores)

**Anomaly Score Computation**:
- Isolation Forest's `decision_function` returns negative scores (more negative = more anomalous)
- We negate these to get positive anomaly scores (higher = more anomalous)

### 4. Fusion Module

**Purpose**: Combine outputs from both detection modules using weighted averaging and apply dynamic thresholding.

**Interface**:
```python
class FusionModule:
    def __init__(self, config: Dict):
        """
        Initialize fusion module.
        
        Args:
            config: Dictionary with weight_autoencoder, weight_isolation, percentile
        """
        
    def fit_threshold(self, recon_errors_benign: np.ndarray, 
                     iso_scores_benign: np.ndarray) -> None:
        """
        Compute threshold from benign validation distribution.
        
        Args:
            recon_errors_benign: Reconstruction errors on benign validation set
            iso_scores_benign: Isolation scores on benign validation set
        """
        
    def normalize_scores(self, recon_errors: np.ndarray, 
                        iso_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize scores to [0, 1] range using min-max scaling.
        
        Args:
            recon_errors: Raw reconstruction errors
            iso_scores: Raw isolation scores
            
        Returns:
            Tuple of (normalized_recon_errors, normalized_iso_scores)
        """
        
    def compute_combined_score(self, recon_errors: np.ndarray, 
                              iso_scores: np.ndarray) -> np.ndarray:
        """
        Compute weighted average of normalized scores.
        
        Args:
            recon_errors: Reconstruction errors
            iso_scores: Isolation scores
            
        Returns:
            Combined anomaly scores
        """
        
    def classify(self, combined_scores: np.ndarray) -> np.ndarray:
        """
        Apply threshold to classify samples.
        
        Args:
            combined_scores: Combined anomaly scores
            
        Returns:
            Binary predictions (0=benign, 1=anomaly)
        """
```

**Fusion Strategy Details**:

1. **Normalization**: Min-max scaling to [0, 1]
   ```
   normalized_score = (score - min_benign) / (max_benign - min_benign)
   ```
   - `min_benign` and `max_benign` computed from benign validation set
   - Clipped to [0, 1] for test samples

2. **Weighted Combination**:
   ```
   combined_score = w_ae * normalized_recon_error + w_if * normalized_iso_score
   ```
   - Default weights: w_ae = 0.5, w_if = 0.5 (equal weighting)
   - Weights must sum to 1.0
   - Can be tuned based on validation performance

3. **Dynamic Thresholding**:
   ```
   threshold = percentile(combined_scores_benign_val, percentile)
   ```
   - Default percentile: 95 (captures 95% of benign traffic)
   - Can be adjusted to 99 for lower FPR
   - Computed from benign validation set combined scores

4. **Classification**:
   ```
   prediction = 1 if combined_score > threshold else 0
   ```

### 5. Healthcare Alert System

**Purpose**: Log anomaly events, generate reports, and provide audit trail for healthcare compliance.

**Interface**:
```python
class HealthcareAlertSystem:
    def __init__(self, config: Dict):
        """
        Initialize alert system.
        
        Args:
            config: Dictionary with log_path, report_path
        """
        
    def log_anomaly(self, timestamp: str, flow_features: Dict, 
                   anomaly_score: float, prediction: int) -> None:
        """
        Log detected anomaly with details.
        
        Args:
            timestamp: Detection timestamp
            flow_features: Dictionary of flow characteristics
            anomaly_score: Combined anomaly score
            prediction: Binary classification (0 or 1)
        """
        
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_scores: np.ndarray, attack_labels: List[str]) -> Dict:
        """
        Generate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels (0=benign, 1=attack)
            y_pred: Predicted labels
            y_scores: Anomaly scores
            attack_labels: List of attack category names
            
        Returns:
            Dictionary containing all metrics and visualizations
        """
        
    def assess_deployment_readiness(self, metrics: Dict) -> str:
        """
        Assess whether system meets healthcare deployment criteria.
        
        Args:
            metrics: Dictionary of evaluation metrics
            
        Returns:
            Deployment readiness assessment (Ready/Not Ready with justification)
        """
```

**Logging Format**:
```json
{
  "timestamp": "2024-01-15T10:30:45Z",
  "anomaly_score": 0.87,
  "prediction": 1,
  "flow_features": {
    "src_ip": "192.168.1.100",
    "dst_ip": "10.0.0.50",
    "protocol": "TCP",
    "flow_duration": 1234,
    ...
  }
}
```

**Report Metrics**:
- Overall Accuracy
- Macro F1-Score
- False Positive Rate (overall and per attack class)
- True Positive Rate / Recall (overall and per attack class)
- Precision (overall and per attack class)
- ROC-AUC score
- Confusion Matrix
- ROC Curve visualization
- Precision-Recall Curve visualization

**Deployment Readiness Criteria**:
- FPR < 5% (acceptable for healthcare)
- Recall > 90% (high sensitivity to attacks)
- Inference latency < 100ms per sample
- System stability over extended testing period

## Data Models

### 1. Network Flow Features

The system processes network flow data with the following feature categories:

**Flow-Based Features** (from CIC-IDS datasets):
- Flow Duration
- Total Fwd Packets, Total Backward Packets
- Total Length of Fwd Packets, Total Length of Bwd Packets
- Fwd Packet Length Max/Min/Mean/Std
- Bwd Packet Length Max/Min/Mean/Std
- Flow Bytes/s, Flow Packets/s
- Flow IAT Mean/Std/Max/Min
- Fwd IAT Total/Mean/Std/Max/Min
- Bwd IAT Total/Mean/Std/Max/Min

**TCP-Specific Features**:
- FIN Flag Count, SYN Flag Count, RST Flag Count
- PSH Flag Count, ACK Flag Count, URG Flag Count
- CWE Flag Count, ECE Flag Count

**Packet-Level Features**:
- Down/Up Ratio
- Average Packet Size
- Avg Fwd Segment Size, Avg Bwd Segment Size
- Fwd Header Length, Bwd Header Length

**Timing Features**:
- Active Mean/Std/Max/Min
- Idle Mean/Std/Max/Min

**Label**:
- BENIGN or specific attack type (e.g., "DoS Hulk", "PortScan", "DDoS", "Bot")

### 2. Configuration Schema

```python
config = {
    # Data paths
    "dataset_paths": [
        "path/to/CIC-IDS2017.csv",
        "path/to/CIC-IDS2018.csv"
    ],
    "log_path": "logs/anomalies.jsonl",
    "report_path": "reports/evaluation.json",
    "model_save_path": "models/",
    
    # Preprocessing
    "test_size": 0.3,
    "val_size": 0.2,
    "random_state": 42,
    
    # Autoencoder
    "autoencoder": {
        "encoding_dim": 32,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 256,
        "early_stopping_patience": 10,
        "use_gpu": True,
        "mixed_precision": True
    },
    
    # Isolation Forest
    "isolation_forest": {
        "n_estimators": 100,
        "max_samples": 256,
        "contamination": "auto",
        "n_jobs": -1
    },
    
    # Fusion
    "fusion": {
        "weight_autoencoder": 0.5,
        "weight_isolation": 0.5,
        "percentile": 95
    },
    
    # Healthcare constraints
    "max_fpr": 0.05,
    "min_recall": 0.90,
    "max_inference_latency_ms": 100
}
```

### 3. Training Data Structure

```python
training_data = {
    "X_train_benign": np.ndarray,  # Shape: (n_benign_train, n_features)
    "X_val_benign": np.ndarray,    # Shape: (n_benign_val, n_features)
    "X_test": np.ndarray,          # Shape: (n_test, n_features)
    "y_test": np.ndarray,          # Shape: (n_test,) - binary labels
    "y_test_detailed": List[str],  # Attack category names
    "scaler": StandardScaler,      # Fitted scaler object
    "feature_names": List[str]     # List of feature column names
}
```

### 4. Inference Output Structure

```python
inference_output = {
    "predictions": np.ndarray,           # Binary predictions (0/1)
    "anomaly_scores": np.ndarray,        # Combined scores
    "reconstruction_errors": np.ndarray, # Autoencoder errors
    "isolation_scores": np.ndarray,      # Isolation Forest scores
    "threshold": float,                  # Applied threshold value
    "inference_time_ms": float          # Per-sample inference time
}
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Benign-Only Training Data

*For any* training execution with a dataset containing both benign and attack samples, the training set (including validation set) should contain only samples labeled as BENIGN, and all attack samples should be reserved exclusively for the test set.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 3.3, 4.3**

### Property 2: Label Preservation During Merge

*For any* set of datasets being merged, the label information for each sample should be identical before and after the merge operation.

**Validates: Requirements 2.2**

### Property 3: Duplicate Removal

*For any* dataset with duplicate rows, the output of the cleaning pipeline should contain no duplicate records.

**Validates: Requirements 2.3**

### Property 4: NaN and Infinity Removal

*For any* dataset, the output of the cleaning pipeline should contain no NaN or infinite values in any column.

**Validates: Requirements 2.4**

### Property 5: Numeric Features Only

*For any* dataset, the output of the preprocessing pipeline should contain only numeric (integer or float) features, with all non-numeric columns removed.

**Validates: Requirements 2.5**

### Property 6: StandardScaler Normalization

*For any* training dataset after normalization, the features should have approximately zero mean and unit standard deviation (within numerical tolerance of 0.1).

**Validates: Requirements 2.6**

### Property 7: Stratified Sampling Preservation

*For any* dataset split using stratified sampling, the proportion of each class in the training and test sets should be approximately equal (within 5% tolerance).

**Validates: Requirements 2.7**

### Property 8: Autoencoder Reconstruction Error Output

*For any* input sample to the trained autoencoder, the output should be a non-negative scalar reconstruction error value.

**Validates: Requirements 3.5**

### Property 9: Isolation Forest Anomaly Score Output

*For any* input sample to the trained isolation forest, the output should be a scalar anomaly score.

**Validates: Requirements 4.4**

### Property 10: Score Normalization to Unit Range

*For any* set of reconstruction errors and isolation scores, the fusion module's normalization should transform them to the range [0, 1] using min-max scaling based on benign validation statistics.

**Validates: Requirements 5.1, 5.2**

### Property 11: Weighted Average Combination

*For any* pair of normalized scores (reconstruction error and isolation score), the combined score should equal the weighted average: w_ae × normalized_recon_error + w_if × normalized_iso_score, where weights sum to 1.0.

**Validates: Requirements 5.3**

### Property 12: Percentile-Based Threshold

*For any* benign validation set, the computed threshold should equal the specified percentile (95th or 99th) of the combined scores on that validation set.

**Validates: Requirements 5.4, 5.5**

### Property 13: Threshold-Based Classification

*For any* combined anomaly score, if the score exceeds the threshold, the prediction should be 1 (anomaly), otherwise 0 (benign).

**Validates: Requirements 5.6**

### Property 14: Anomaly Logging Completeness

*For any* sample classified as anomalous (prediction = 1), a log entry should be created containing timestamp, flow features, anomaly score, and prediction.

**Validates: Requirements 6.5**

### Property 15: ROC-AUC Computation Correctness

*For any* set of true labels and predicted scores, the computed ROC-AUC should match the sklearn.metrics.roc_auc_score calculation.

**Validates: Requirements 6.7, 8.4**

### Property 16: Confusion Matrix Correctness

*For any* set of true labels and predictions, the confusion matrix should correctly count true positives, true negatives, false positives, and false negatives.

**Validates: Requirements 6.9**

### Property 17: Zero-Day Attack Detection

*For any* attack pattern not present in the training data, the system should assign it an anomaly score higher than the majority of benign samples (demonstrating detection capability for novel attacks).

**Validates: Requirements 7.2, 7.5**

### Property 18: Accuracy Computation Correctness

*For any* set of true labels and predictions, the computed accuracy should equal the proportion of correct predictions: (TP + TN) / (TP + TN + FP + FN).

**Validates: Requirements 8.1**

### Property 19: Macro F1-Score Computation Correctness

*For any* set of true labels and predictions with multiple classes, the macro F1-score should equal the unweighted mean of per-class F1-scores.

**Validates: Requirements 8.2**

### Property 20: False Positive Rate Computation Correctness

*For any* set of true labels and predictions, the computed FPR should equal FP / (FP + TN) for the benign class.

**Validates: Requirements 8.3**

### Property 21: Reproducibility with Fixed Seeds

*For any* training execution with fixed random seeds, running the pipeline twice with identical inputs and seeds should produce identical model weights, predictions, and metrics.

**Validates: Requirements 9.4, 10.1**

## Error Handling

The system must handle various error conditions gracefully to ensure reliability in healthcare environments:

### 1. Data Loading Errors

**Error Conditions**:
- Missing dataset files
- Corrupted CSV files
- Incorrect file format
- Insufficient memory for large datasets

**Handling Strategy**:
- Validate file existence before loading
- Use try-except blocks with specific error messages
- Implement chunked reading for large files
- Log all data loading errors with file paths and error details
- Raise informative exceptions that guide users to solutions

### 2. Data Quality Errors

**Error Conditions**:
- All samples are attack traffic (no benign samples for training)
- All samples are benign (no attacks for evaluation)
- Insufficient samples for stratified splitting
- Label column missing or incorrectly named
- All features are non-numeric

**Handling Strategy**:
- Validate minimum sample counts before training (e.g., ≥1000 benign samples)
- Check for label column existence with common name variations
- Provide clear error messages indicating required data characteristics
- Suggest data collection or preprocessing steps to resolve issues

### 3. Training Errors

**Error Conditions**:
- Autoencoder fails to converge
- Isolation Forest training fails due to memory
- GPU out of memory
- Numerical instability (NaN in gradients)

**Handling Strategy**:
- Implement early stopping to prevent infinite training
- Catch GPU OOM errors and automatically fall back to CPU
- Use gradient clipping to prevent exploding gradients
- Log training metrics to detect convergence issues
- Provide checkpointing to resume from failures

### 4. Inference Errors

**Error Conditions**:
- Input features don't match training dimensions
- Input contains NaN or inf values
- Model not trained before inference
- Threshold not computed before classification

**Handling Strategy**:
- Validate input shape matches expected dimensions
- Check for and reject invalid input values
- Verify model is trained before allowing inference
- Ensure threshold is fitted before classification
- Return error codes with descriptive messages

### 5. Configuration Errors

**Error Conditions**:
- Invalid hyperparameter values (e.g., negative learning rate)
- Weights don't sum to 1.0 in fusion module
- Invalid percentile value (outside 0-100)
- Conflicting configuration options

**Handling Strategy**:
- Validate all configuration parameters at initialization
- Use schema validation (e.g., pydantic) for config structure
- Provide default values for optional parameters
- Raise ValueError with clear explanation of valid ranges

### 6. Healthcare-Specific Error Handling

**Error Conditions**:
- Log file write failures (disk full)
- Report generation failures
- Alert system unavailable

**Handling Strategy**:
- Implement retry logic for transient failures
- Use buffered logging with fallback to console
- Continue operation even if logging fails (don't block detection)
- Generate partial reports if full report fails
- Maintain audit trail of all errors for compliance

## Testing Strategy

The testing strategy employs a dual approach combining unit tests for specific examples and edge cases with property-based tests for universal correctness guarantees.

### Unit Testing

Unit tests focus on:

1. **Specific Examples**: Concrete test cases with known inputs and expected outputs
   - Example: Test preprocessing with a small synthetic dataset
   - Example: Test autoencoder with a 10-sample benign dataset
   - Example: Test fusion with known reconstruction errors and isolation scores

2. **Edge Cases**: Boundary conditions and special scenarios
   - Empty datasets
   - Single-sample datasets
   - All-benign or all-attack datasets
   - Extreme feature values (very large or very small)
   - Datasets with only one feature

3. **Error Conditions**: Verify proper error handling
   - Missing files
   - Corrupted data
   - Invalid configurations
   - GPU unavailability

4. **Integration Points**: Component interactions
   - Preprocessing → Autoencoder pipeline
   - Autoencoder + Isolation Forest → Fusion
   - Fusion → Alert System

**Unit Test Framework**: pytest with fixtures for common test data

**Coverage Target**: >80% code coverage for all modules

### Property-Based Testing

Property-based tests verify universal properties across randomly generated inputs. Each property test runs a minimum of 100 iterations with different random inputs.

**Property Testing Library**: 
- Python: Hypothesis
- Alternative: pytest-quickcheck

**Test Configuration**:
```python
@given(strategies.data_frames(...))
@settings(max_examples=100, deadline=None)
def test_property_X(data):
    # Property test implementation
    pass
```

**Property Test Coverage**:

Each correctness property from the design document must be implemented as a property-based test:

1. **Property 1**: Benign-Only Training Data
   - **Tag**: Feature: hybrid-ids-healthcare, Property 1: Benign-only training
   - Generate datasets with mixed labels, verify training set contains only benign

2. **Property 2**: Label Preservation During Merge
   - **Tag**: Feature: hybrid-ids-healthcare, Property 2: Label preservation
   - Generate multiple datasets, merge, verify labels unchanged

3. **Property 3**: Duplicate Removal
   - **Tag**: Feature: hybrid-ids-healthcare, Property 3: Duplicate removal
   - Generate datasets with duplicates, verify output has no duplicates

4. **Property 4**: NaN and Infinity Removal
   - **Tag**: Feature: hybrid-ids-healthcare, Property 4: NaN/inf removal
   - Generate datasets with NaN/inf, verify output is clean

5. **Property 5**: Numeric Features Only
   - **Tag**: Feature: hybrid-ids-healthcare, Property 5: Numeric features
   - Generate datasets with mixed types, verify output is numeric only

6. **Property 6**: StandardScaler Normalization
   - **Tag**: Feature: hybrid-ids-healthcare, Property 6: Normalization
   - Generate datasets, normalize, verify mean≈0 and std≈1

7. **Property 7**: Stratified Sampling Preservation
   - **Tag**: Feature: hybrid-ids-healthcare, Property 7: Stratified sampling
   - Generate datasets, split, verify class proportions maintained

8. **Property 8**: Autoencoder Reconstruction Error Output
   - **Tag**: Feature: hybrid-ids-healthcare, Property 8: Reconstruction error
   - Generate random inputs, verify output is non-negative scalar

9. **Property 9**: Isolation Forest Anomaly Score Output
   - **Tag**: Feature: hybrid-ids-healthcare, Property 9: Anomaly score
   - Generate random inputs, verify output is scalar

10. **Property 10**: Score Normalization to Unit Range
    - **Tag**: Feature: hybrid-ids-healthcare, Property 10: Score normalization
    - Generate random scores, normalize, verify output in [0,1]

11. **Property 11**: Weighted Average Combination
    - **Tag**: Feature: hybrid-ids-healthcare, Property 11: Weighted average
    - Generate random normalized scores, verify combined score equals weighted average

12. **Property 12**: Percentile-Based Threshold
    - **Tag**: Feature: hybrid-ids-healthcare, Property 12: Percentile threshold
    - Generate random benign scores, verify threshold equals specified percentile

13. **Property 13**: Threshold-Based Classification
    - **Tag**: Feature: hybrid-ids-healthcare, Property 13: Threshold classification
    - Generate random scores, verify classification follows threshold rule

14. **Property 14**: Anomaly Logging Completeness
    - **Tag**: Feature: hybrid-ids-healthcare, Property 14: Logging completeness
    - Generate random anomalies, verify all have log entries with required fields

15. **Property 15**: ROC-AUC Computation Correctness
    - **Tag**: Feature: hybrid-ids-healthcare, Property 15: ROC-AUC correctness
    - Generate random labels and scores, verify ROC-AUC matches sklearn

16. **Property 16**: Confusion Matrix Correctness
    - **Tag**: Feature: hybrid-ids-healthcare, Property 16: Confusion matrix
    - Generate random labels and predictions, verify confusion matrix counts

17. **Property 17**: Zero-Day Attack Detection
    - **Tag**: Feature: hybrid-ids-healthcare, Property 17: Zero-day detection
    - Train on benign, test on novel attacks, verify high anomaly scores

18. **Property 18**: Accuracy Computation Correctness
    - **Tag**: Feature: hybrid-ids-healthcare, Property 18: Accuracy correctness
    - Generate random labels and predictions, verify accuracy formula

19. **Property 19**: Macro F1-Score Computation Correctness
    - **Tag**: Feature: hybrid-ids-healthcare, Property 19: F1-score correctness
    - Generate random multi-class labels and predictions, verify F1 calculation

20. **Property 20**: False Positive Rate Computation Correctness
    - **Tag**: Feature: hybrid-ids-healthcare, Property 20: FPR correctness
    - Generate random labels and predictions, verify FPR formula

21. **Property 21**: Reproducibility with Fixed Seeds
    - **Tag**: Feature: hybrid-ids-healthcare, Property 21: Reproducibility
    - Run pipeline twice with same seed, verify identical outputs

### Test Data Strategy

**Synthetic Data Generation**:
- Use Hypothesis strategies to generate realistic network flow features
- Generate benign traffic with normal distributions
- Generate attack traffic with anomalous patterns (outliers, unusual distributions)
- Ensure generated data respects domain constraints (e.g., non-negative packet counts)

**Real Data Testing**:
- Use small subsets of CIC-IDS2017/2018 for integration tests
- Create fixture datasets with known characteristics
- Include edge cases from real data (e.g., zero-duration flows)

### Continuous Integration

**CI Pipeline**:
1. Run all unit tests on every commit
2. Run property tests (100 iterations each) on every commit
3. Run extended property tests (1000 iterations) nightly
4. Run full integration tests with real data weekly
5. Generate coverage reports and fail if <80%

**Test Execution Time**:
- Unit tests: <2 minutes
- Property tests (100 iterations): <10 minutes
- Extended property tests (1000 iterations): <1 hour
- Full integration tests: <2 hours

### Testing Priorities

**Critical Path** (must pass before merge):
- Property 1: Benign-only training
- Property 8, 9: Model output correctness
- Property 13: Classification logic
- Property 17: Zero-day detection capability

**High Priority** (should pass before merge):
- All data preprocessing properties (2-7)
- All fusion properties (10-13)
- All metric computation properties (15-20)

**Medium Priority** (can be fixed post-merge):
- Logging and reporting properties (14)
- Reproducibility (21)
- Edge case unit tests

This dual testing approach ensures both concrete correctness (unit tests) and universal correctness (property tests), providing comprehensive validation suitable for healthcare deployment where reliability is critical.
