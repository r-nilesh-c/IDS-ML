# Common Mistakes and How to Fix Them

This guide covers the most frequent errors students encounter when implementing the hybrid IDS system.

## Data Preprocessing Mistakes

### 1. Data Leakage: Fitting Scaler on Test Data

**Mistake:**
```python
# WRONG: Fitting on all data
scaler = StandardScaler()
X_all_normalized = scaler.fit_transform(X_all)
X_train, X_test = train_test_split(X_all_normalized)
```

**Why it's wrong:** The scaler "sees" test data statistics during training, giving the model unfair advantage.

**Fix:**
```python
# CORRECT: Fit only on training data
X_train, X_test = train_test_split(X_all)
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)  # Only transform
```

**How to detect:** If your test accuracy is suspiciously high (>99%), you likely have data leakage.

---

### 2. Normalizing the Label Column

**Mistake:**
```python
# WRONG: Normalizing everything including labels
df_normalized = scaler.fit_transform(df)  # df includes 'Label' column
```

**Why it's wrong:** Labels should remain as categorical values, not normalized numbers.

**Fix:**
```python
# CORRECT: Separate features and labels first
X = df.drop(columns=['Label'])
y = df['Label']
X_normalized = scaler.fit_transform(X)
```

---

### 3. Not Handling Infinite Values

**Mistake:**
```python
# WRONG: Only removing NaN
df_clean = df.dropna()
```

**Why it's wrong:** Infinite values (from division by zero) break StandardScaler.

**Fix:**
```python
# CORRECT: Remove both NaN and inf
df_clean = df.dropna()
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
inf_mask = np.isinf(df_clean[numeric_cols]).any(axis=1)
df_clean = df_clean[~inf_mask]
```

**Error message:** `ValueError: Input contains infinity or a value too large`

---

### 4. Training on Mixed Data (Benign + Attacks)

**Mistake:**
```python
# WRONG: Training on all data
autoencoder.train(X_all, y_all)
```

**Why it's wrong:** The autoencoder learns to reconstruct attacks, defeating the purpose.

**Fix:**
```python
# CORRECT: Train only on benign data
benign_mask = y == 0
X_benign = X[benign_mask]
autoencoder.train(X_benign, X_benign)
```

---

## Autoencoder Mistakes

### 5. Wrong Output Activation Function

**Mistake:**
```python
# WRONG: Using ReLU for output
decoded = layers.Dense(input_dim, activation='relu')(decoded)
```

**Why it's wrong:** ReLU outputs [0, ∞), but normalized data is in [0, 1].

**Fix:**
```python
# CORRECT: Use sigmoid for [0, 1] output
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
```

**Symptom:** Very high reconstruction errors even on training data.

---

### 6. Bottleneck Too Large

**Mistake:**
```python
# WRONG: Encoding dimension equals input dimension
autoencoder = AutoencoderDetector(input_dim=77, encoding_dim=77)
```

**Why it's wrong:** No compression means the model can memorize everything, including attacks.

**Fix:**
```python
# CORRECT: Encoding dimension much smaller than input
autoencoder = AutoencoderDetector(input_dim=77, encoding_dim=32)
```

**Rule of thumb:** encoding_dim should be 25-50% of input_dim.

---

### 7. Not Using Early Stopping

**Mistake:**
```python
# WRONG: Training for fixed epochs
model.fit(X_train, X_train, epochs=100)
```

**Why it's wrong:** Model overfits to training data, poor generalization.

**Fix:**
```python
# CORRECT: Use early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
model.fit(X_train, X_train, epochs=100, 
          validation_data=(X_val, X_val),
          callbacks=[early_stopping])
```

**Symptom:** Training loss decreases but validation loss increases.

---

### 8. Forgetting to Normalize Before Training

**Mistake:**
```python
# WRONG: Training on raw data
autoencoder.train(X_raw, X_raw)
```

**Why it's wrong:** Features with large scales dominate learning, model doesn't converge.

**Fix:**
```python
# CORRECT: Normalize first
X_normalized = scaler.fit_transform(X_raw)
autoencoder.train(X_normalized, X_normalized)
```

**Symptom:** Loss stays high and doesn't decrease during training.

---

### 9. Wrong Input/Output Pairs

**Mistake:**
```python
# WRONG: Using labels as output
model.fit(X_train, y_train)
```

**Why it's wrong:** Autoencoder is a reconstruction task, not classification.

**Fix:**
```python
# CORRECT: Input and output are the same
model.fit(X_train, X_train)
```

---

## Isolation Forest Mistakes

### 10. Using Contamination as Threshold

**Mistake:**
```python
# WRONG: Manually thresholding with contamination
predictions = (scores > contamination).astype(int)
```

**Why it's wrong:** Contamination is a training parameter, not a threshold.

**Fix:**
```python
# CORRECT: Use model's predict method
predictions = iso_forest.predict(X_test)
# Or use score_samples for custom thresholding
scores = iso_forest.score_samples(X_test)
```

---

### 11. Not Inverting Scores

**Mistake:**
```python
# WRONG: Using raw scores (lower = more anomalous)
anomaly_scores = iso_forest.score_samples(X_test)
# Higher scores are treated as anomalies (incorrect!)
```

**Why it's wrong:** Isolation Forest returns negative scores for anomalies.

**Fix:**
```python
# CORRECT: Invert scores
raw_scores = iso_forest.score_samples(X_test)
anomaly_scores = -raw_scores  # Now higher = more anomalous
```

---

### 12. Too Few Trees

**Mistake:**
```python
# WRONG: Using very few trees
iso_forest = IsolationForest(n_estimators=10)
```

**Why it's wrong:** Predictions are unstable and vary between runs.

**Fix:**
```python
# CORRECT: Use sufficient trees
iso_forest = IsolationForest(n_estimators=100)
```

**Rule of thumb:** Start with 100 trees, increase if predictions are unstable.

---

## Fusion Mistakes

### 13. Not Normalizing Scores Before Fusion

**Mistake:**
```python
# WRONG: Fusing scores with different scales
fused = 0.5 * ae_scores + 0.5 * if_scores
```

**Why it's wrong:** Scores have different ranges, one dominates the fusion.

**Fix:**
```python
# CORRECT: Normalize to [0, 1] first
ae_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min())
if_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
fused = 0.5 * ae_norm + 0.5 * if_norm
```

---

### 14. Using Test Data to Set Threshold

**Mistake:**
```python
# WRONG: Using test data statistics
threshold = np.percentile(test_scores, 95)
```

**Why it's wrong:** This is data leakage - you're using test data to make decisions.

**Fix:**
```python
# CORRECT: Use validation data
threshold = np.percentile(val_scores, 95)
# Then apply to test data
predictions = (test_scores > threshold).astype(int)
```

---

## Evaluation Mistakes

### 15. Using Accuracy for Imbalanced Data

**Mistake:**
```python
# WRONG: Reporting only accuracy
accuracy = (predictions == y_test).mean()
print(f"Accuracy: {accuracy:.2%}")
```

**Why it's wrong:** With 95% benign and 5% attacks, predicting all benign gives 95% accuracy!

**Fix:**
```python
# CORRECT: Use precision, recall, F1-score
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
```

---

### 16. Confusing Precision and Recall

**Mistake:**
```python
# WRONG: Optimizing for high precision in healthcare
# (This misses many attacks!)
```

**Why it's wrong:** Healthcare needs high recall (catch all attacks), not just high precision.

**Fix:**
```python
# CORRECT: Optimize for recall while keeping FPR low
# Use F2-score (weights recall higher than precision)
from sklearn.metrics import fbeta_score
f2 = fbeta_score(y_test, predictions, beta=2)
```

**Remember:**
- **Precision**: Of predicted attacks, how many are real? (minimize false alarms)
- **Recall**: Of real attacks, how many did we catch? (minimize missed attacks)

---

## TensorFlow/Keras Mistakes

### 17. Not Setting Random Seeds

**Mistake:**
```python
# WRONG: No random seed
model = build_model()
```

**Why it's wrong:** Results are not reproducible between runs.

**Fix:**
```python
# CORRECT: Set all random seeds
import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
```

---

### 18. GPU Memory Issues

**Mistake:**
```python
# WRONG: Allocating all GPU memory
# (Crashes when running multiple models)
```

**Why it's wrong:** TensorFlow allocates all GPU memory by default.

**Fix:**
```python
# CORRECT: Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

**Error message:** `ResourceExhaustedError: OOM when allocating tensor`

---

### 19. Not Saving the Best Model

**Mistake:**
```python
# WRONG: Saving final model (may be overfit)
model.save('final_model.keras')
```

**Why it's wrong:** The final epoch model may not be the best model.

**Fix:**
```python
# CORRECT: Use ModelCheckpoint to save best model
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True
)
model.fit(..., callbacks=[checkpoint])
```

---

## Performance Mistakes

### 20. Loading Entire Dataset into Memory

**Mistake:**
```python
# WRONG: Loading all CSV files at once
dfs = [pd.read_csv(f) for f in all_files]  # May crash with large datasets
```

**Why it's wrong:** Large datasets (>10GB) may exceed available RAM.

**Fix:**
```python
# CORRECT: Process in chunks or use generators
for chunk in pd.read_csv(file, chunksize=10000):
    process(chunk)
```

---

### 21. Not Using Batch Processing

**Mistake:**
```python
# WRONG: Processing samples one at a time
for sample in X_test:
    score = model.predict(sample.reshape(1, -1))
```

**Why it's wrong:** Very slow, doesn't utilize GPU/CPU parallelism.

**Fix:**
```python
# CORRECT: Process in batches
scores = model.predict(X_test, batch_size=256)
```

**Speed improvement:** 10-100x faster

---

## Debugging Tips

### How to Debug Data Leakage
1. Check if test accuracy is unrealistically high (>99%)
2. Verify scaler is fit only on training data
3. Ensure no test data is used for threshold selection
4. Check that train/test split happens before any preprocessing

### How to Debug Poor Performance
1. Check data normalization (mean ≈ 0, std ≈ 1)
2. Verify benign-only training
3. Check for infinite/NaN values
4. Visualize reconstruction errors (should separate benign/attack)
5. Try different hyperparameters

### How to Debug Training Issues
1. Check loss curve (should decrease)
2. Verify input/output shapes match
3. Check activation functions (sigmoid for output)
4. Ensure learning rate is reasonable (0.001 is good default)
5. Check for NaN in loss (indicates numerical instability)

## Quick Checklist

Before running your model, verify:

- [ ] Scaler fit only on training data
- [ ] Label column excluded from normalization
- [ ] No infinite or NaN values in data
- [ ] Training only on benign samples
- [ ] Encoding dimension < input dimension
- [ ] Early stopping enabled
- [ ] Random seeds set for reproducibility
- [ ] Validation data separate from test data
- [ ] Scores normalized before fusion
- [ ] Using appropriate metrics (not just accuracy)

## Getting Help

If you're stuck:
1. Check error message carefully
2. Search this document for the error
3. Print intermediate values to debug
4. Visualize data at each step
5. Start with a small subset of data
6. Compare with reference implementation

Remember: Everyone makes these mistakes! The key is learning to recognize and fix them quickly.
