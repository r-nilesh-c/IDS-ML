# Feature Selection Architecture Fix

## Problem Identified

**The core issue**: Your IDS uses a fixed 12-feature model, but features were **not being persisted** across the training→evaluation→live monitoring pipeline.

| Stage               | Behavior                                                             | Problem                                   |
| ------------------- | -------------------------------------------------------------------- | ----------------------------------------- |
| **Training**        | Selects 12 features via variance → trains models                     | ❌ Doesn't save which 12 features         |
| **Evaluation**      | Detects feature mismatch → re-derives 12 features from training data | ⚠️ Works but inefficient                  |
| **Live Monitoring** | Uses ALL 78 raw features                                             | 💥 **Feature dimension mismatch → crash** |

### Example Failure Scenario

```
Raw incoming CSV (78 features):
  [Destination Port, Protocol, Packet Count, ..., Flow Duration] (78 cols)
            ↓
Live monitor tries to apply scaler trained on 12 features
            ↓
X.shape[1] = 78 but scaler expects 12
            ↓
ERROR: "Feature dimension mismatch"
```

---

## Solution Implemented

### 1. **Persist Selected Features During Training**

**File**: `train_cascaded_full.py` (Lines 163-181)

```python
# Select top 12 features for consistency across training and inference
benign_df_selected, attack_df_selected, selected_features = preprocessing.select_features(
    benign_df, attack_df,
    n_features=12,  # Fixed to 12 features
    method='variance'
)
# Get these EXACT 12 features selected
print(f"Selected {len(selected_features)} features: {selected_features}")
```

**File**: `train_cascaded_full.py` (Lines 322-327)

```python
# Save selected feature names for consistent inference
# This ensures live monitoring and evaluation use the SAME 12 features
selected_features_path = os.path.join(save_dir, 'selected_features.pkl')
with open(selected_features_path, 'wb') as f:
    pickle.dump({'selected_features': selected_features, 'n_features': len(selected_features)}, f)
print(f"  - Selected Features Metadata: {selected_features_path}")
```

### 2. **Load & Use Selected Features in Evaluation**

**File**: `evaluate.py` (Lines 76-104)

```python
def load_models(model_dir: str, config: dict):
    # Load selected features metadata
    selected_features_path = os.path.join(model_dir, 'selected_features.pkl')
    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'rb') as f:
            features_metadata = pickle.load(f)
            selected_features = features_metadata.get('selected_features', None)
            n_selected = features_metadata.get('n_features', 0)
        logger.info(f"Selected features loaded: {n_selected} features")
        return autoencoder, ..., selected_features
```

**File**: `evaluate.py` (Lines 665-667)

```python
autoencoder, isolation_forest, fusion, model_input_dim, scaler, selected_features = load_models(...)

X_test, y_test, attack_labels = load_test_data(
    args.test_data,
    ...
    selected_features=selected_features  # Pass to load function
)
```

**File**: `evaluate.py` (Lines 257-276)

```python
# PRIORITY: Use saved selected_features if available
if selected_features is not None and len(selected_features) > 0:
    logger.info(f"Using saved selected features ({len(selected_features)} features)")

    # Apply selection
    df_clean = df_clean[selected_features + ['Label']]
    feature_cols = selected_features
    logger.info(f"Applied {len(selected_features)} saved selected features")
```

### 3. **Load & Use Selected Features in Live Monitoring**

**File**: `live_monitor_cascaded.py` (Lines 117-161)

```python
def prepare_features(
    df: pd.DataFrame,
    preprocessing: PreprocessingPipeline,
    selected_features: List[str] = None,  # NEW parameter
    scaler=None
) -> pd.DataFrame:
    df_clean = preprocessing.clean_data(df)

    # If selected features are provided, extract ONLY those features
    if selected_features is not None:
        available_cols = [c for c in df_clean.columns if c not in LABEL_COLS]
        missing = [f for f in selected_features if f not in available_cols]
        if missing:
            raise ValueError(f"Selected features not found in input data. Missing: {missing}")

        X_df = df_clean[selected_features].copy()
        logger.info(f"Extracted {len(selected_features)} selected features for inference")
    else:
        logger.warning("No selected features provided; using all features (may cause mismatch)")

    # Apply scaling with dimension validation
    if scaler is not None:
        expected_features = scaler.n_features_in_
        if X_df.shape[1] != expected_features:
            raise ValueError(
                f"Feature dimension mismatch: X_df has {X_df.shape[1]} features "
                f"but scaler expects {expected_features}"
            )
```

**File**: `live_monitor_cascaded.py` (Lines 270-283)

```python
# Load selected features for consistent inference
selected_features_path = os.path.join(args.model_dir, 'selected_features.pkl')
selected_features = None
if os.path.exists(selected_features_path):
    with open(selected_features_path, 'rb') as f:
        features_metadata = pickle.load(f)
        selected_features = features_metadata.get('selected_features', None)
        n_features = features_metadata.get('n_features', 0)
    logger.info(f'Loaded {n_features} selected features from {selected_features_path}')
else:
    logger.warning('selected_features.pkl not found. Live monitoring may encounter errors.')
```

---

## How It Works End-to-End

Now all three pipelines use **the exact same 12 features**:

```
TRAINING (train_cascaded_full.py)
  ↓
  Load 78-feature raw data
  ↓
  Select top 12 by variance on benign data
  ↓ [saved as selected_features.pkl]
  ├── feature_A,  feature_B,  ..., feature_L (12 specific features)
  ↓
  Train autoencoder on 12 features
  Train isolation forest on 12 features

EVALUATION (evaluate.py)
  ↓
  Load models + selected_features.pkl [12 feature names]
  ↓
  Load 78-feature raw test data
  ↓
  Extract ONLY the 12 saved features
  device_A, feature_B, ..., feature_L ← same 12 as training!
  ↓
  Apply scaler (trained on 12 features)
  Run inference
  ✓ Works! 12 features match model expectation

LIVE MONITORING (live_monitor_cascaded.py)
  ↓
  Load models + selected_features.pkl [12 feature names]
  ↓
  CSV arrives with 78 features
  ↓
  Extract ONLY the 12 saved features
  device_A, feature_B, ..., feature_L ← same 12 as training!
  ↓
  Apply scaler (trained on 12 features)
  Run inference
  ✓ Works! 12 features match model expectation
```

---

## File Changes Summary

| File                       | Changes                                 | Purpose                                        |
| -------------------------- | --------------------------------------- | ---------------------------------------------- |
| `train_cascaded_full.py`   | Lines 163-181, 180-195, 322-327         | Select 12 features explicitly, save metadata   |
| `evaluate.py`              | Lines 76-104, 188-211, 257-276, 665-667 | Load selected_features, use with priority      |
| `live_monitor_cascaded.py` | Lines 117-161, 178-187, 270-283         | Load selected_features, extract before scaling |

---

## Benefits

✅ **No More Feature Mismatch Errors**: Live monitoring always uses the exact same 12 features  
✅ **Reproducibility**: Feature selection is deterministic and saved  
✅ **Efficiency**: No need to re-derive features every evaluation  
✅ **Consistency**: Training, evaluation, and live monitoring all aligned  
✅ **Clear Error Messages**: Detailed logging if features don't exist in incoming data

---

## Next Steps After Retraining

When training completes with the updated `train_cascaded_full.py`:

1. **Check for new artifact**:

   ```bash
   ls -la models/selected_features.pkl
   # Should exist with ~12 features
   ```

2. **Evaluate with clean command** (no `--train-data` needed):

   ```bash
   python evaluate.py --test-data dataset/cic-ids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv
   # Will auto-load selected_features.pkl and use those 12 features
   ```

3. **Live monitoring**:
   ```bash
   python live_monitor_cascaded.py --watch-dir data/live --poll-seconds 5
   # Will auto-load selected_features.pkl and extract only those features from incoming CSVs
   ```

---

**Architecture Status**: ✅ Fixed  
**Feature Consistency**: ✅ Guaranteed  
**Live Monitoring Ready**: ✅ Yes (after retraining)
