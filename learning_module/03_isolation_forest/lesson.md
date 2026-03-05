# Lesson 3: Isolation Forest for Anomaly Detection

## Learning Objectives

By the end of this lesson, you will:
- Understand the isolation principle for anomaly detection
- Implement Isolation Forest from scikit-learn
- Compare tree-based vs neural network approaches
- Tune contamination and n_estimators parameters
- Combine Isolation Forest with Autoencoder

## What is Isolation Forest?

Isolation Forest detects anomalies by measuring how easy it is to isolate a sample:

**Key Insight**: Anomalies are rare and different, so they're easier to isolate than normal samples.

### The Isolation Principle

Imagine randomly splitting data:

```
Benign sample (in dense cluster):
  Split 1: 500 samples left, 500 samples right
  Split 2: 250 samples left, 250 samples right
  Split 3: 125 samples left, 125 samples right
  ...
  Split 12: Finally isolated (deep in tree)

Attack sample (outlier):
  Split 1: 1 sample left, 999 samples right
  Isolated in 1 split! (shallow in tree)
```

**Anomaly Score**: Average path length across multiple trees
- Short paths → Anomaly
- Long paths → Normal

## How Isolation Forest Works

### Step 1: Build Isolation Trees

```python
for each tree:
    1. Randomly select a feature
    2. Randomly select a split value between min and max
    3. Split data into left and right
    4. Repeat until each sample is isolated or max_depth reached
```

**Key Difference from Decision Trees:**
- No labels needed (unsupervised)
- Random splits (not optimized)
- Goal: Isolate samples, not classify them

### Step 2: Compute Path Lengths

For each sample, measure how many splits needed to isolate it:

```python
path_length = number of splits to reach leaf node
```

### Step 3: Aggregate Across Trees

```python
average_path_length = mean(path_lengths across all trees)
anomaly_score = 2^(-average_path_length / c(n))
```

Where `c(n)` is a normalization factor based on sample size.

**Anomaly Score Range**: [0, 1]
- Close to 1: Anomaly
- Close to 0.5: Normal
- Close to 0: Very normal (deep in tree)

## Implementation

### Basic Usage

```python
from sklearn.ensemble import IsolationForest

# Initialize
iso_forest = IsolationForest(
    n_estimators=100,      # Number of trees
    contamination=0.1,     # Expected proportion of anomalies
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

# Train on benign data only
iso_forest.fit(X_train_benign)

# Predict anomaly scores
scores = iso_forest.score_samples(X_test)  # Lower = more anomalous
predictions = iso_forest.predict(X_test)   # -1 = anomaly, 1 = normal
```

### Full Implementation

```python
class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector.
    
    Uses ensemble of isolation trees to detect anomalies
    through path length analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Isolation Forest detector.
        
        Args:
            config: Dictionary with n_estimators, contamination, 
                   max_samples, random_state
        """
        self.config = config
        self.n_estimators = config.get('n_estimators', 100)
        self.contamination = config.get('contamination', 0.1)
        self.max_samples = config.get('max_samples', 'auto')
        self.random_state = config.get('random_state', 42)
        
        # Initialize model
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )
        
        logger.info(f"IsolationForestDetector initialized with "
                   f"n_estimators={self.n_estimators}, "
                   f"contamination={self.contamination}")
    
    def train(self, X_train: np.ndarray) -> None:
        """
        Train Isolation Forest on benign traffic.
        
        Args:
            X_train: Benign training samples, shape (n_samples, n_features)
        """
        logger.info(f"Training Isolation Forest on {X_train.shape[0]} samples")
        
        # Fit the model
        self.model.fit(X_train)
        
        logger.info("Isolation Forest training completed")
    
    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for input samples.
        
        Args:
            X: Input samples, shape (n_samples, n_features)
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        # Get scores from model (lower = more anomalous)
        raw_scores = self.model.score_samples(X)
        
        # Invert scores (higher = more anomalous)
        anomaly_scores = -raw_scores
        
        return anomaly_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Input samples
            
        Returns:
            Binary labels (0 = normal, 1 = anomaly)
        """
        predictions = self.model.predict(X)
        # Convert -1/1 to 0/1
        binary_predictions = (predictions == -1).astype(int)
        return binary_predictions
```

## Hyperparameter Tuning

### n_estimators (Number of Trees)

```
Too few (e.g., 10):
  ✗ High variance in scores
  ✗ Unstable predictions
  ✗ Poor generalization

Too many (e.g., 1000):
  ✗ Slower training and inference
  ✗ Diminishing returns
  ✗ Wasted computation

Sweet spot (100-200):
  ✓ Stable predictions
  ✓ Good performance
  ✓ Reasonable speed
```

**Rule of thumb**: Start with 100, increase if predictions are unstable.

### contamination (Expected Anomaly Rate)

```
Too low (e.g., 0.01):
  ✗ Very strict threshold
  ✗ Misses many attacks (low recall)
  ✓ Few false positives (high precision)

Too high (e.g., 0.5):
  ✗ Very loose threshold
  ✗ Many false positives (low precision)
  ✓ Catches most attacks (high recall)

Sweet spot (0.05-0.15):
  ✓ Balanced precision and recall
  ✓ Suitable for most IDS scenarios
```

**For healthcare**: Use lower contamination (0.05) to minimize false positives.

### max_samples (Samples per Tree)

```
'auto' (default):
  Uses min(256, n_samples)
  ✓ Good for large datasets
  ✓ Faster training

'int' (e.g., 1000):
  Uses exactly 1000 samples per tree
  ✓ More control
  ✓ Consistent across datasets

'float' (e.g., 0.5):
  Uses 50% of samples per tree
  ✓ Scales with dataset size
```

**Recommendation**: Use 'auto' unless you have specific requirements.

## Isolation Forest vs Autoencoder

### Comparison Table

| Aspect | Isolation Forest | Autoencoder |
|--------|-----------------|-------------|
| **Type** | Tree-based | Neural network |
| **Training Speed** | Fast (seconds) | Slow (minutes) |
| **Inference Speed** | Fast | Medium |
| **Memory Usage** | Low | High (GPU) |
| **Interpretability** | Medium (feature importance) | Low (black box) |
| **Hyperparameters** | Few (2-3) | Many (5+) |
| **Feature Interactions** | Limited | Captures complex patterns |
| **Robustness** | High | Medium (sensitive to hyperparameters) |

### When to Use Each

**Use Isolation Forest when:**
- Fast training is critical
- Limited computational resources
- Need interpretability (feature importance)
- Data has clear outliers

**Use Autoencoder when:**
- Complex feature interactions exist
- GPU available
- Need to capture subtle patterns
- Willing to tune hyperparameters

**Use Both (Hybrid) when:**
- Maximum detection accuracy needed
- Can afford computational cost
- Want robustness to different attack types

## Hybrid Approach: Combining Both

### Why Combine?

Isolation Forest and Autoencoder detect different types of anomalies:

```
Isolation Forest excels at:
  - Point anomalies (single outlier samples)
  - Feature-based anomalies (unusual feature values)
  - Fast detection

Autoencoder excels at:
  - Contextual anomalies (unusual feature combinations)
  - Subtle pattern deviations
  - Complex relationships
```

### Fusion Strategy

```python
# Get scores from both detectors
ae_scores = autoencoder.compute_reconstruction_error(X_test)
if_scores = isolation_forest.compute_anomaly_scores(X_test)

# Normalize scores to [0, 1]
ae_scores_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min())
if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())

# Weighted average
w_ae = 0.6  # Weight for autoencoder
w_if = 0.4  # Weight for isolation forest
fused_scores = w_ae * ae_scores_norm + w_if * if_scores_norm

# Threshold for detection
threshold = np.percentile(fused_scores, 95)  # Top 5% as anomalies
predictions = (fused_scores > threshold).astype(int)
```

### Weight Selection

```
Equal weights (0.5, 0.5):
  ✓ Simple, no tuning needed
  ✓ Good starting point
  ✗ May not be optimal

Autoencoder-heavy (0.7, 0.3):
  ✓ Better for complex attacks
  ✓ Captures subtle patterns
  ✗ Slower inference

Isolation Forest-heavy (0.3, 0.7):
  ✓ Faster inference
  ✓ Better for simple outliers
  ✗ May miss complex attacks

Validation-based:
  ✓ Optimal for your dataset
  ✓ Data-driven
  ✗ Requires validation set
```

**Recommendation**: Use validation set to find optimal weights.

## Exercises

### Exercise 1: Train Isolation Forest (Easy)
```python
# TODO: Train Isolation Forest on benign data
# Compute anomaly scores on test set
# Plot score distribution for benign vs attacks

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def train_and_evaluate_if(X_train_benign, X_test, y_test):
    # Your code here
    pass
```

### Exercise 2: Hyperparameter Search (Medium)
```python
# TODO: Grid search over n_estimators and contamination
# Find best combination using F1-score
# Plot heatmap of results

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score

def hyperparameter_search(X_train, X_test, y_test):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'contamination': [0.05, 0.1, 0.15, 0.2]
    }
    # Your code here
    pass
```

### Exercise 3: Feature Importance (Medium)
```python
# TODO: Analyze which features are most important for isolation
# Hint: Features with high variance in path lengths are important
# Visualize top 10 features

def analyze_feature_importance(iso_forest, X_train, feature_names):
    # Your code here
    pass
```

### Exercise 4: Implement Hybrid Detector (Hard)
```python
# TODO: Combine Autoencoder and Isolation Forest
# Implement weighted fusion
# Find optimal weights using validation set
# Compare with individual detectors

class HybridDetector:
    def __init__(self, autoencoder, isolation_forest):
        self.autoencoder = autoencoder
        self.isolation_forest = isolation_forest
        self.w_ae = 0.5
        self.w_if = 0.5
    
    def optimize_weights(self, X_val, y_val):
        # Your code here
        pass
    
    def predict(self, X):
        # Your code here
        pass
```

## Quiz

1. **Why are anomalies easier to isolate?**
   - A) They are always in the same location
   - B) They are rare and different from normal samples
   - C) They have higher feature values
   - D) They are closer to the origin

2. **What does contamination parameter control?**
   - A) Training speed
   - B) Number of trees
   - C) Expected proportion of anomalies
   - D) Feature selection

3. **Why combine Isolation Forest and Autoencoder?**
   - A) To make training faster
   - B) To reduce memory usage
   - C) To detect different types of anomalies
   - D) To simplify the model

4. **What does a short path length indicate?**
   - A) Normal sample
   - B) Anomaly
   - C) Error in data
   - D) Missing value

## Project: Build a Hybrid IDS

**Goal**: Implement a complete hybrid detection system

**Requirements**:
1. Train Isolation Forest on benign data
2. Train Autoencoder on benign data (from Lesson 2)
3. Implement score fusion with configurable weights
4. Evaluate on test set with multiple attack types
5. Generate performance report (precision, recall, F1)

**Starter Code**:
```python
class HybridIDS:
    def __init__(self, input_dim, config):
        self.autoencoder = AutoencoderDetector(input_dim, config)
        self.isolation_forest = IsolationForestDetector(config)
        self.w_ae = config.get('w_ae', 0.6)
        self.w_if = config.get('w_if', 0.4)
    
    def train(self, X_train_benign, X_val_benign):
        # TODO: Train both detectors
        pass
    
    def predict(self, X):
        # TODO: Fuse scores and predict
        pass
    
    def evaluate(self, X_test, y_test):
        # TODO: Compute metrics
        pass

# Test your implementation
# ids = HybridIDS(input_dim=77, config={...})
# ids.train(X_train_benign, X_val_benign)
# metrics = ids.evaluate(X_test, y_test)
# print(f"Precision: {metrics['precision']:.3f}")
# print(f"Recall: {metrics['recall']:.3f}")
# print(f"F1-Score: {metrics['f1']:.3f}")
```

## Common Mistakes

1. **Using contamination as threshold**: It's for training, not prediction
2. **Not normalizing scores before fusion**: Different scales cause imbalance
3. **Training on mixed data**: Always use benign-only training
4. **Ignoring inference speed**: Isolation Forest is much faster
5. **Equal weights without validation**: May not be optimal

## Key Takeaways

✅ Isolation Forest detects anomalies through path length analysis
✅ Fast training and inference (seconds vs minutes)
✅ Complementary to Autoencoder (different anomaly types)
✅ Contamination parameter controls detection threshold
✅ Hybrid approach combines strengths of both methods
✅ Weight optimization improves detection accuracy

## Next Lesson

In Lesson 4, we'll implement the complete fusion module with dynamic thresholding and healthcare-specific optimizations.

**Preview**: How do we set thresholds automatically? What's the trade-off between false positives and false negatives? How do we optimize for healthcare deployment?

Continue to `04_fusion_and_deployment/lesson.md` →
