# Quick Start Guide

Get up and running with the learning module in 15 minutes!

## Prerequisites

Ensure you have:
- Python 3.8 or higher
- pip package manager
- 8GB+ RAM (16GB recommended)
- GPU optional (speeds up training 10x)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow (deep learning)
- scikit-learn (machine learning)
- pandas (data manipulation)
- numpy (numerical computing)
- matplotlib (visualization)

### 2. Verify Installation

```python
# test_installation.py
import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np

print(f"TensorFlow: {tf.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")
```

Run:
```bash
python test_installation.py
```

Expected output:
```
TensorFlow: 2.13.0
scikit-learn: 1.3.0
pandas: 2.0.3
numpy: 1.24.3
GPUs available: 1  # or 0 if no GPU
```

## Dataset Setup

### Option 1: Use Provided Sample Data (Recommended for Learning)

We'll create synthetic data for quick testing:

```python
# create_sample_data.py
import numpy as np
import pandas as pd

# Generate synthetic benign traffic
np.random.seed(42)
benign = np.random.randn(1000, 77) * 0.5 + 0.5
benign = np.clip(benign, 0, 1)

# Generate synthetic attack traffic (different distribution)
attack = np.random.randn(200, 77) * 1.5 + 0.5
attack = np.clip(attack, 0, 1)

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(77)]
benign_df = pd.DataFrame(benign, columns=feature_names)
benign_df['Label'] = 'BENIGN'

attack_df = pd.DataFrame(attack, columns=feature_names)
attack_df['Label'] = 'ATTACK'

# Combine and save
df = pd.concat([benign_df, attack_df], ignore_index=True)
df.to_csv('sample_data.csv', index=False)

print(f"Created sample_data.csv with {len(df)} samples")
print(f"Benign: {len(benign_df)}, Attack: {len(attack_df)}")
```

Run:
```bash
python create_sample_data.py
```

### Option 2: Download Real CIC-IDS Dataset

1. Visit [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Download CSV files
3. Place in `dataset/cic-ids2017/` directory

## Your First Model (10 Minutes)

### Step 1: Preprocess Data

```python
# step1_preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('sample_data.csv')
print(f"Loaded {len(df)} samples")

# Separate features and labels
X = df.drop(columns=['Label']).values
y = (df['Label'] != 'BENIGN').astype(int).values

# Split benign and attack
benign_mask = y == 0
X_benign = X[benign_mask]
X_attack = X[~benign_mask]

# Split benign into train/val
X_train, X_val = train_test_split(X_benign, test_size=0.2, random_state=42)

# Create test set (benign + attack)
X_test = np.vstack([X_val[:50], X_attack])
y_test = np.hstack([np.zeros(50), np.ones(len(X_attack))])

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print(f"Preprocessed data saved!")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

Run:
```bash
python step1_preprocess.py
```

### Step 2: Train Autoencoder

```python
# step2_train_autoencoder.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load preprocessed data
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')

# Build autoencoder
input_dim = X_train.shape[1]
encoding_dim = 32

input_layer = keras.Input(shape=(input_dim,))
encoded = layers.Dense(64, activation='relu')(input_layer)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

model = keras.Model(inputs=input_layer, outputs=decoded)
model.compile(optimizer='adam', loss='mse')

print("Training autoencoder...")
history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

# Save model
model.save('autoencoder.keras')
print("Model saved to autoencoder.keras")
```

Run:
```bash
python step2_train_autoencoder.py
```

### Step 3: Evaluate

```python
# step3_evaluate.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Load data and model
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
model = tf.keras.models.load_model('autoencoder.keras')

# Compute reconstruction errors
reconstructions = model.predict(X_test, verbose=0)
errors = np.mean(np.square(X_test - reconstructions), axis=1)

# Set threshold (95th percentile of validation errors)
X_val = np.load('X_val.npy')
val_reconstructions = model.predict(X_val, verbose=0)
val_errors = np.mean(np.square(X_val - val_reconstructions), axis=1)
threshold = np.percentile(val_errors, 95)

# Predict
predictions = (errors > threshold).astype(int)

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Benign', 'Attack']))

auc = roc_auc_score(y_test, errors)
print(f"\nROC-AUC Score: {auc:.3f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.hist(errors[y_test == 0], bins=50, alpha=0.5, label='Benign', color='green')
plt.hist(errors[y_test == 1], bins=50, alpha=0.5, label='Attack', color='red')
plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.4f}')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Reconstruction Error Distribution')
plt.legend()
plt.yscale('log')
plt.savefig('results.png', dpi=150)
print("\nVisualization saved to results.png")
plt.show()
```

Run:
```bash
python step3_evaluate.py
```

## Expected Results

With synthetic data, you should see:
- **Precision**: 0.70-0.90
- **Recall**: 0.80-0.95
- **ROC-AUC**: 0.85-0.95

With real CIC-IDS data:
- **Precision**: 0.85-0.95
- **Recall**: 0.90-0.98
- **ROC-AUC**: 0.95-0.99

## Next Steps

Now that you have a working model:

1. **Start Lesson 1**: Learn the theory behind preprocessing
   ```bash
   cd learning_module/01_preprocessing
   ```

2. **Experiment**: Try different hyperparameters
   - Change `encoding_dim` (16, 32, 64)
   - Adjust threshold percentile (90, 95, 99)
   - Add dropout layers

3. **Add Isolation Forest**: Implement hybrid detection
   ```python
   from sklearn.ensemble import IsolationForest
   iso_forest = IsolationForest(n_estimators=100, contamination=0.1)
   iso_forest.fit(X_train)
   ```

4. **Use Real Data**: Download CIC-IDS2017 and repeat

## Troubleshooting

### "TensorFlow not found"
```bash
pip install tensorflow
```

### "Out of memory" error
Reduce batch size:
```python
model.fit(..., batch_size=16)  # Instead of 32
```

### "No GPU available"
That's fine! Training will be slower but still works:
```
CPU: ~5 minutes
GPU: ~30 seconds
```

### Poor performance on synthetic data
This is expected! Synthetic data is for testing only. Use real CIC-IDS data for actual evaluation.

## Quick Reference

### File Structure
```
learning_module/
├── README.md                    # Start here
├── quick_start.md              # This file
├── 01_preprocessing/
│   └── lesson.md               # Lesson 1
├── 02_autoencoder/
│   └── lesson.md               # Lesson 2
├── 03_isolation_forest/
│   └── lesson.md               # Lesson 3
├── solutions/                   # Reference implementations
├── resources.md                 # Additional learning materials
└── common_mistakes.md          # Debugging guide
```

### Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Create sample data
python create_sample_data.py

# Run complete pipeline
python step1_preprocess.py
python step2_train_autoencoder.py
python step3_evaluate.py

# Start learning
cd learning_module/01_preprocessing
```

## Getting Help

- **Stuck?** Check `common_mistakes.md`
- **Need theory?** Read the lesson files
- **Want examples?** See `solutions/` folder
- **More resources?** Check `resources.md`

## Ready to Learn?

You now have a working IDS! Time to understand how it works:

👉 **Start with**: `learning_module/01_preprocessing/lesson.md`

Happy learning! 🚀
