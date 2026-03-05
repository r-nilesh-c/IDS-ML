"""
Solutions for Lesson 2: Autoencoder for Anomaly Detection
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt


# Exercise 1: Build a Simple Autoencoder
def build_simple_autoencoder(input_dim=10, encoding_dim=3):
    """
    Build a 2-layer autoencoder (1 encoder, 1 decoder).
    
    Args:
        input_dim: Number of input features
        encoding_dim: Size of bottleneck layer
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    input_layer = keras.Input(shape=(input_dim,), name='input')
    
    # Encoder (single layer)
    encoded = layers.Dense(encoding_dim, activation='relu', name='encoder')(input_layer)
    
    # Decoder (single layer)
    decoded = layers.Dense(input_dim, activation='sigmoid', name='decoder')(encoded)
    
    # Create model
    model = keras.Model(inputs=input_layer, outputs=decoded, name='simple_autoencoder')
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


# Exercise 2: Visualize Reconstruction
def visualize_reconstruction(model, X_benign, X_attack):
    """
    Visualize original vs reconstructed samples and error distributions.
    
    Args:
        model: Trained autoencoder
        X_benign: Benign test samples
        X_attack: Attack test samples
    """
    # Get reconstructions
    benign_reconstructed = model.predict(X_benign[:5], verbose=0)
    attack_reconstructed = model.predict(X_attack[:5], verbose=0)
    
    # Compute reconstruction errors
    benign_errors = np.mean(np.square(X_benign - model.predict(X_benign, verbose=0)), axis=1)
    attack_errors = np.mean(np.square(X_attack - model.predict(X_attack, verbose=0)), axis=1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Sample reconstructions (benign)
    ax1 = plt.subplot(2, 2, 1)
    for i in range(5):
        ax1.plot(X_benign[i], alpha=0.5, label=f'Original {i+1}')
        ax1.plot(benign_reconstructed[i], '--', alpha=0.5, label=f'Reconstructed {i+1}')
    ax1.set_title('Benign Samples: Original vs Reconstructed')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Feature Value')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Sample reconstructions (attack)
    ax2 = plt.subplot(2, 2, 2)
    for i in range(5):
        ax2.plot(X_attack[i], alpha=0.5, label=f'Original {i+1}')
        ax2.plot(attack_reconstructed[i], '--', alpha=0.5, label=f'Reconstructed {i+1}')
    ax2.set_title('Attack Samples: Original vs Reconstructed')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Feature Value')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 3: Reconstruction error distribution
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(benign_errors, bins=50, alpha=0.5, label='Benign', color='green')
    ax3.hist(attack_errors, bins=50, alpha=0.5, label='Attack', color='red')
    ax3.set_xlabel('Reconstruction Error (MSE)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Reconstruction Error Distribution')
    ax3.legend()
    ax3.set_yscale('log')
    
    # Plot 4: Box plot comparison
    ax4 = plt.subplot(2, 2, 4)
    ax4.boxplot([benign_errors, attack_errors], labels=['Benign', 'Attack'])
    ax4.set_ylabel('Reconstruction Error (MSE)')
    ax4.set_title('Reconstruction Error Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reconstruction_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'reconstruction_visualization.png'")
    plt.show()
    
    # Print statistics
    print(f"\nBenign reconstruction error: {benign_errors.mean():.6f} ± {benign_errors.std():.6f}")
    print(f"Attack reconstruction error: {attack_errors.mean():.6f} ± {attack_errors.std():.6f}")
    print(f"Separation ratio: {attack_errors.mean() / benign_errors.mean():.2f}x")


# Exercise 3: Experiment with Encoding Dimension
def compare_encoding_dimensions(X_train, X_val, X_test_benign, X_test_attack):
    """
    Compare autoencoders with different encoding dimensions.
    
    Args:
        X_train: Training data (benign)
        X_val: Validation data (benign)
        X_test_benign: Test benign samples
        X_test_attack: Test attack samples
    """
    input_dim = X_train.shape[1]
    encoding_dims = [8, 16, 32, 64, 128]
    
    results = {
        'encoding_dim': [],
        'val_loss': [],
        'benign_error_mean': [],
        'attack_error_mean': [],
        'separation': []
    }
    
    for enc_dim in encoding_dims:
        print(f"\nTraining with encoding_dim={enc_dim}")
        
        # Build model
        input_layer = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(enc_dim * 2, activation='relu')(input_layer)
        encoded = layers.Dense(enc_dim, activation='relu')(encoded)
        decoded = layers.Dense(enc_dim * 2, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        model = keras.Model(inputs=input_layer, outputs=decoded)
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        history = model.fit(
            X_train, X_train,
            epochs=50,
            batch_size=256,
            validation_data=(X_val, X_val),
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        
        # Evaluate
        val_loss = min(history.history['val_loss'])
        
        benign_errors = np.mean(np.square(X_test_benign - model.predict(X_test_benign, verbose=0)), axis=1)
        attack_errors = np.mean(np.square(X_test_attack - model.predict(X_test_attack, verbose=0)), axis=1)
        
        separation = attack_errors.mean() / benign_errors.mean()
        
        # Store results
        results['encoding_dim'].append(enc_dim)
        results['val_loss'].append(val_loss)
        results['benign_error_mean'].append(benign_errors.mean())
        results['attack_error_mean'].append(attack_errors.mean())
        results['separation'].append(separation)
        
        print(f"Val loss: {val_loss:.6f}, Separation: {separation:.2f}x")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Validation loss
    axes[0, 0].plot(results['encoding_dim'], results['val_loss'], marker='o')
    axes[0, 0].set_xlabel('Encoding Dimension')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].set_title('Validation Loss vs Encoding Dimension')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction errors
    axes[0, 1].plot(results['encoding_dim'], results['benign_error_mean'], marker='o', label='Benign')
    axes[0, 1].plot(results['encoding_dim'], results['attack_error_mean'], marker='s', label='Attack')
    axes[0, 1].set_xlabel('Encoding Dimension')
    axes[0, 1].set_ylabel('Mean Reconstruction Error')
    axes[0, 1].set_title('Reconstruction Error vs Encoding Dimension')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Separation ratio
    axes[1, 0].plot(results['encoding_dim'], results['separation'], marker='o', color='purple')
    axes[1, 0].set_xlabel('Encoding Dimension')
    axes[1, 0].set_ylabel('Separation Ratio (Attack/Benign)')
    axes[1, 0].set_title('Error Separation vs Encoding Dimension')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No separation')
    axes[1, 0].legend()
    
    # Plot 4: Summary table
    axes[1, 1].axis('off')
    table_data = []
    for i in range(len(encoding_dims)):
        table_data.append([
            encoding_dims[i],
            f"{results['val_loss'][i]:.4f}",
            f"{results['separation'][i]:.2f}x"
        ])
    
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Encoding Dim', 'Val Loss', 'Separation'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Summary Table')
    
    plt.tight_layout()
    plt.savefig('encoding_dimension_comparison.png', dpi=150)
    print("\nComparison saved to 'encoding_dimension_comparison.png'")
    plt.show()


# Exercise 4: Implement Denoising Autoencoder
def build_denoising_autoencoder(input_dim, encoding_dim, noise_factor=0.1):
    """
    Build a denoising autoencoder with Gaussian noise.
    
    Args:
        input_dim: Number of input features
        encoding_dim: Size of bottleneck layer
        noise_factor: Standard deviation of Gaussian noise
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    input_layer = keras.Input(shape=(input_dim,), name='input')
    
    # Add Gaussian noise during training
    noisy = layers.GaussianNoise(noise_factor, name='noise')(input_layer)
    
    # Encoder
    encoded = layers.Dense(encoding_dim * 2, activation='relu', name='encoder1')(noisy)
    encoded = layers.Dropout(0.2, name='encoder_dropout')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu', name='encoder2')(encoded)
    
    # Decoder
    decoded = layers.Dense(encoding_dim * 2, activation='relu', name='decoder1')(encoded)
    decoded = layers.Dropout(0.2, name='decoder_dropout')(decoded)
    decoded = layers.Dense(input_dim, activation='sigmoid', name='decoder2')(decoded)
    
    # Create model
    model = keras.Model(inputs=input_layer, outputs=decoded, name='denoising_autoencoder')
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# Project: Build Your Own Autoencoder Detector
class MyAutoencoder:
    """
    Complete autoencoder implementation for anomaly detection.
    """
    
    def __init__(self, input_dim, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.history = None
    
    def build(self, use_dropout=True, dropout_rate=0.2):
        """Build the autoencoder architecture."""
        input_layer = keras.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = layers.Dense(self.encoding_dim * 2, activation='relu')(input_layer)
        if use_dropout:
            encoded = layers.Dropout(dropout_rate)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(self.encoding_dim * 2, activation='relu')(encoded)
        if use_dropout:
            decoded = layers.Dropout(dropout_rate)(decoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        # Create and compile
        self.model = keras.Model(inputs=input_layer, outputs=decoded)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"Model built with {self.model.count_params()} parameters")
        return self.model
    
    def train(self, X_train, X_val, epochs=50, batch_size=256):
        """Train the autoencoder."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_autoencoder.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict_anomaly_scores(self, X):
        """Compute reconstruction errors as anomaly scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        reconstructions = self.model.predict(X, verbose=0)
        mse_per_sample = np.mean(np.square(X - reconstructions), axis=1)
        return mse_per_sample
    
    def plot_training_history(self):
        """Plot training and validation loss."""
        if self.history is None:
            raise ValueError("No training history available.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_history.png', dpi=150)
        plt.show()


# Example usage
if __name__ == "__main__":
    print("Autoencoder Solutions - Example Usage")
    print("=" * 50)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    X_train = np.random.randn(1000, 20) * 0.5 + 0.5
    X_val = np.random.randn(200, 20) * 0.5 + 0.5
    X_test_benign = np.random.randn(100, 20) * 0.5 + 0.5
    X_test_attack = np.random.randn(100, 20) * 2.0 + 0.5  # Different distribution
    
    # Clip to [0, 1]
    X_train = np.clip(X_train, 0, 1)
    X_val = np.clip(X_val, 0, 1)
    X_test_benign = np.clip(X_test_benign, 0, 1)
    X_test_attack = np.clip(X_test_attack, 0, 1)
    
    # Test Exercise 1
    print("\nExercise 1: Simple Autoencoder")
    model = build_simple_autoencoder(input_dim=10, encoding_dim=3)
    model.summary()
    
    # Test Project
    print("\nProject: Complete Autoencoder")
    autoencoder = MyAutoencoder(input_dim=20, encoding_dim=8)
    autoencoder.build()
    # autoencoder.train(X_train, X_val, epochs=10)
    # scores = autoencoder.predict_anomaly_scores(X_test_benign)
    # print(f"Mean anomaly score: {scores.mean():.6f}")
