# Lesson 2: Autoencoder for Anomaly Detection

## Module Alignment (March 2026 Update)

- Rebuild roadmap: `learning_module/PROJECT_REBUILD_MODULES.md`
- Previous lesson: `learning_module/01_preprocessing/lesson.md`
- Next lesson: `learning_module/03_isolation_forest/lesson.md`
- Solution files:
  - `learning_module/solutions/02_autoencoder_solutions.py`
  - `src/autoencoder.py`

## Learning Objectives

By the end of this lesson, you will:

- Understand how autoencoders detect anomalies through reconstruction
- Build a deep autoencoder architecture with TensorFlow/Keras
- Implement training with early stopping and checkpointing
- Compute reconstruction errors for anomaly scoring
- Configure GPU acceleration and mixed precision

## What is an Autoencoder?

An autoencoder is a neural network that learns to compress and reconstruct data:

```
Input (77 features) → Encoder → Bottleneck (32 features) → Decoder → Output (77 features)
```

**Goal**: Make output identical to input (reconstruction task)

**For Anomaly Detection**:

- Train on benign traffic only
- Benign samples: Low reconstruction error (model learned them well)
- Attack samples: High reconstruction error (model never saw them)

## Architecture Deep Dive

### The Bottleneck Principle

```python
Input: 77 features
  ↓
Encoder Layer 1: 64 neurons (encoding_dim * 2)
  ↓
Encoder Layer 2: 32 neurons (encoding_dim) ← Bottleneck
  ↓
Decoder Layer 1: 64 neurons (encoding_dim * 2)
  ↓
Decoder Layer 2: 77 neurons (input_dim)
```

**Why the bottleneck?**

- Forces the network to learn compressed representations
- Can't just memorize inputs (not enough capacity)
- Must learn meaningful patterns in benign traffic

### Activation Functions

```python
# Encoder and decoder hidden layers
activation='relu'  # ReLU: max(0, x)

# Output layer
activation='sigmoid'  # Sigmoid: 1 / (1 + e^-x), outputs [0, 1]
```

**Why ReLU?**

- Prevents vanishing gradients
- Computationally efficient
- Introduces non-linearity (enables learning complex patterns)

**Why Sigmoid output?**

- After normalization, features are roughly in [0, 1] range
- Sigmoid naturally outputs [0, 1]
- Smooth gradients for training

### Dropout for Regularization

```python
encoded = layers.Dropout(0.2)(encoded)
```

**What dropout does:**

- Randomly sets 20% of neurons to zero during training
- Prevents overfitting (model relying too much on specific neurons)
- Forces redundant representations

**When to use:**

- Small datasets (high overfitting risk)
- Complex architectures
- When validation loss increases while training loss decreases

## Implementation Walkthrough

### Step 1: Initialization

```python
class AutoencoderDetector:
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        self.input_dim = input_dim  # Number of features (e.g., 77)
        self.encoding_dim = config.get('encoding_dim', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 256)
```

**Key Parameters:**

- `input_dim`: Must match preprocessed feature count
- `encoding_dim`: Bottleneck size (smaller = more compression)
- `learning_rate`: Step size for gradient descent (0.001 is a good default)
- `batch_size`: Samples processed together (larger = faster but more memory)

### Step 2: Building the Model

```python
def build_model(self, use_dropout: bool = True) -> keras.Model:
    # Input layer
    input_layer = keras.Input(shape=(self.input_dim,))

    # Encoder
    encoded = layers.Dense(self.encoding_dim * 2, activation='relu')(input_layer)
    if use_dropout:
        encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(self.encoding_dim * 2, activation='relu')(encoded)
    if use_dropout:
        decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)

    # Create and compile model
    self.model = keras.Model(inputs=input_layer, outputs=decoded)
    self.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
        loss='mse',  # Mean Squared Error
        metrics=['mae']  # Mean Absolute Error
    )
    return self.model
```

**Loss Function: MSE (Mean Squared Error)**

```
MSE = (1/n) * Σ(y_true - y_pred)²
```

- Penalizes large errors heavily (squared term)
- Differentiable (needed for backpropagation)
- Standard for regression tasks

**Optimizer: Adam**

- Adaptive learning rate (adjusts per parameter)
- Combines momentum and RMSprop
- Works well out-of-the-box

### Step 3: Training with Early Stopping

```python
def train(self, X_train: np.ndarray, X_val: np.ndarray):
    # Early stopping: Stop if validation loss doesn't improve
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  # Wait 10 epochs before stopping
        restore_best_weights=True  # Revert to best model
    )

    # Model checkpoint: Save best model
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='models/autoencoder_best.keras',
        monitor='val_loss',
        save_best_only=True
    )

    # Train (note: input and output are the same!)
    history = self.model.fit(
        X_train, X_train,  # Reconstruction task
        epochs=self.epochs,
        batch_size=self.batch_size,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping, model_checkpoint]
    )
    return history
```

**Why Early Stopping?**

```
Epoch 1-20: Training loss ↓, Validation loss ↓  (Good - learning)
Epoch 21-30: Training loss ↓, Validation loss →  (Plateau)
Epoch 31+: Training loss ↓, Validation loss ↑  (Overfitting!)
```

Early stopping prevents overfitting by stopping when validation loss stops improving.

**Patience Parameter:**

- Too low (e.g., 3): Stops too early, underfitting
- Too high (e.g., 50): Wastes time, may overfit
- Sweet spot: 10-15 epochs

### Step 4: Computing Reconstruction Error

```python
def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
    # Get reconstructions
    reconstructions = self.model.predict(X, verbose=0)

    # Compute MSE per sample
    mse_per_sample = np.mean(np.square(X - reconstructions), axis=1)

    return mse_per_sample
```

**Reconstruction Error as Anomaly Score:**

```
Benign sample:
  Input:  [0.2, 0.5, 0.8, ...]
  Output: [0.21, 0.49, 0.79, ...]
  Error:  0.0003 (LOW - normal traffic)

Attack sample:
  Input:  [0.9, 0.1, 0.3, ...]
  Output: [0.4, 0.6, 0.7, ...]
  Error:  0.15 (HIGH - abnormal traffic)
```

## GPU Acceleration

### Why Use GPU?

Training time comparison (100 epochs, 100k samples):

- CPU: ~30 minutes
- GPU: ~3 minutes (10x faster)

### Configuration

```python
def _configure_device(self):
    gpus = tf.config.list_physical_devices('GPU')

    if self.use_gpu and len(gpus) > 0:
        # Enable memory growth (don't allocate all GPU memory)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU acceleration enabled")
    else:
        # Force CPU execution
        tf.config.set_visible_devices([], 'GPU')
        logger.info("CPU execution configured")
```

**Memory Growth:**

- Without: TensorFlow allocates all GPU memory upfront
- With: Allocates memory as needed (allows multiple processes)

### Mixed Precision Training

```python
def _configure_mixed_precision(self):
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
```

**What is mixed precision?**

- Computation: float16 (16-bit, faster)
- Accumulation: float32 (32-bit, stable)
- Result: 2x faster training on modern GPUs (Volta, Turing, Ampere)

**When to use:**

- Large models
- Large datasets
- GPU with Tensor Cores (RTX 20xx, 30xx, 40xx, A100, etc.)

## Hyperparameter Tuning Guide

### Encoding Dimension

```
Too small (e.g., 8):
  ✗ Can't capture benign patterns
  ✗ High reconstruction error on benign traffic
  ✗ Poor detection (everything looks anomalous)

Too large (e.g., 128):
  ✗ Too much capacity
  ✗ Memorizes training data
  ✗ Low reconstruction error on attacks (false negatives)

Sweet spot (32-64):
  ✓ Captures essential patterns
  ✓ Generalizes well
  ✓ Good separation between benign and attacks
```

### Learning Rate

```
Too high (e.g., 0.1):
  ✗ Training unstable
  ✗ Loss oscillates
  ✗ May not converge

Too low (e.g., 0.00001):
  ✗ Training very slow
  ✗ May get stuck in local minima

Sweet spot (0.001-0.01):
  ✓ Stable convergence
  ✓ Reasonable training time
```

### Batch Size

```
Small (32-64):
  ✓ More frequent updates
  ✓ Better generalization
  ✗ Slower training
  ✗ Noisy gradients

Large (512-1024):
  ✓ Faster training
  ✓ Stable gradients
  ✗ May overfit
  ✗ Requires more memory

Sweet spot (128-256):
  ✓ Good balance
```

## Exercises

### Exercise 1: Build a Simple Autoencoder (Easy)

```python
# TODO: Build a 2-layer autoencoder (1 encoder, 1 decoder)
# Input: 10 features, Encoding: 3 features

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_simple_autoencoder(input_dim=10, encoding_dim=3):
    # Your code here
    pass

# Test
model = build_simple_autoencoder()
model.summary()
```

### Exercise 2: Visualize Reconstruction (Medium)

```python
# TODO: Train autoencoder on benign data
# Visualize original vs reconstructed for 5 samples
# Plot reconstruction error distribution

import matplotlib.pyplot as plt

def visualize_reconstruction(model, X_benign, X_attack):
    # Your code here
    pass
```

### Exercise 3: Experiment with Encoding Dimension (Medium)

```python
# TODO: Train autoencoders with encoding_dim = [8, 16, 32, 64, 128]
# Compare validation loss and reconstruction error separation
# Plot results

def compare_encoding_dimensions(X_train, X_val, X_test_benign, X_test_attack):
    # Your code here
    pass
```

### Exercise 4: Implement Denoising Autoencoder (Hard)

```python
# TODO: Add Gaussian noise to inputs during training
# This makes the autoencoder more robust
# Compare with standard autoencoder

def build_denoising_autoencoder(input_dim, encoding_dim, noise_factor=0.1):
    # Your code here
    # Hint: Use layers.GaussianNoise(noise_factor)
    pass
```

## Quiz

1. **Why do we use sigmoid activation in the output layer?**

   - A) It's faster than ReLU
   - B) It outputs values in [0, 1] range
   - C) It prevents overfitting
   - D) It's required by TensorFlow

2. **What does early stopping prevent?**

   - A) Underfitting
   - B) Overfitting
   - C) Data leakage
   - D) GPU errors

3. **Why is the bottleneck layer important?**

   - A) It speeds up training
   - B) It forces compression and pattern learning
   - C) It reduces memory usage
   - D) It's required by Keras

4. **What indicates an anomaly in autoencoder detection?**
   - A) Low reconstruction error
   - B) High reconstruction error
   - C) Negative reconstruction error
   - D) Zero reconstruction error

## Project: Build Your Own Autoencoder Detector

**Goal**: Implement a working autoencoder for anomaly detection

**Requirements**:

1. Build autoencoder with configurable architecture
2. Train on benign traffic with early stopping
3. Compute reconstruction errors
4. Evaluate on test set (benign + attacks)
5. Plot ROC curve

**Starter Code**:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class MyAutoencoder:
    def __init__(self, input_dim, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None

    def build(self):
        # TODO: Implement
        pass

    def train(self, X_train, X_val, epochs=50):
        # TODO: Implement
        pass

    def predict_anomaly_scores(self, X):
        # TODO: Implement (return reconstruction errors)
        pass

# Test your implementation
# Load preprocessed data from Lesson 1
# autoencoder = MyAutoencoder(input_dim=77, encoding_dim=32)
# autoencoder.build()
# autoencoder.train(X_train_benign, X_val_benign)
# scores = autoencoder.predict_anomaly_scores(X_test)
```

## Common Mistakes

1. **Not using benign-only training**: Including attacks in training defeats the purpose
2. **Forgetting to normalize**: Autoencoder won't converge without normalization
3. **Wrong output activation**: Using ReLU instead of sigmoid for output
4. **Ignoring early stopping**: Training too long causes overfitting
5. **Not saving best model**: Final model may not be the best model

## Key Takeaways

✅ Autoencoders detect anomalies through reconstruction error
✅ Bottleneck forces learning of compressed representations
✅ Train only on benign traffic for zero-day detection
✅ Early stopping prevents overfitting
✅ GPU acceleration speeds up training 10x
✅ Encoding dimension is the most important hyperparameter

## Next Lesson

In Lesson 3, we'll implement Isolation Forest for classical ML-based anomaly detection and compare it with the autoencoder.

**Preview**: How does Isolation Forest work? Why is it effective for anomaly detection? How do we combine it with the autoencoder?

Continue to `03_isolation_forest/lesson.md` →
