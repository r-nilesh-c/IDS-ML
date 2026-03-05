"""
Autoencoder-based anomaly detection module.

This module implements a deep learning autoencoder for detecting anomalies
through reconstruction error.
"""

import logging
import os
from typing import Dict, Any
import numpy as np

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.error("TensorFlow not available. Please install tensorflow.")


logger = logging.getLogger(__name__)


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector.
    
    This class implements a deep learning autoencoder that learns to reconstruct
    benign network traffic. Anomalies are detected through high reconstruction error.
    
    The autoencoder is trained exclusively on benign traffic to enable detection
    of novel attacks and zero-day threats.
    """
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        """
        Initialize autoencoder architecture.
        
        Args:
            input_dim: Number of input features
            config: Dictionary with encoding_dim, learning_rate, epochs, batch_size,
                   early_stopping_patience, use_gpu, mixed_precision
                   
        Raises:
            ImportError: If TensorFlow is not available
            ValueError: If configuration parameters are invalid
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for AutoencoderDetector")
        
        # Validate input dimension
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        
        self.input_dim = input_dim
        self.config = config
        
        # Extract configuration parameters with defaults
        self.encoding_dim = config.get('encoding_dim', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 256)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.use_gpu = config.get('use_gpu', True)
        self.mixed_precision = config.get('mixed_precision', True)
        
        # Validate configuration parameters
        if self.encoding_dim <= 0:
            raise ValueError(f"encoding_dim must be positive, got {self.encoding_dim}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        # Set random seeds for TensorFlow reproducibility
        random_state = config.get('random_state', 42)
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Enable deterministic operations for reproducibility
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        
        logger.info(f"TensorFlow random seed set to {random_state}")
        
        # Configure GPU/CPU execution
        self._configure_device()
        
        # Configure mixed precision if available and requested
        if self.mixed_precision and self.use_gpu:
            self._configure_mixed_precision()
        
        # Model will be built later
        self.model = None
        
        logger.info(f"AutoencoderDetector initialized with input_dim={input_dim}, "
                   f"encoding_dim={self.encoding_dim}")
    
    def _configure_device(self) -> None:
        """
        Configure GPU/CPU execution based on availability and configuration.
        
        If GPU is requested but not available, falls back to CPU gracefully.
        """
        gpus = tf.config.list_physical_devices('GPU')
        
        if self.use_gpu and len(gpus) > 0:
            try:
                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                logger.info(f"GPU acceleration enabled. Found {len(gpus)} GPU(s)")
                logger.info(f"GPU devices: {[gpu.name for gpu in gpus]}")
                
            except RuntimeError as e:
                logger.warning(f"Failed to configure GPU: {e}")
                logger.info("Falling back to CPU execution")
                
        elif self.use_gpu and len(gpus) == 0:
            logger.warning("GPU requested but no GPU devices found. Using CPU.")
            
        else:
            # Force CPU execution
            tf.config.set_visible_devices([], 'GPU')
            logger.info("CPU execution configured")
    
    def _configure_mixed_precision(self) -> None:
        """
        Configure mixed precision training for efficiency on compatible GPUs.
        
        Mixed precision uses float16 for computation and float32 for numerical
        stability, providing faster training on modern GPUs.
        """
        try:
            # Check if GPU supports mixed precision
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) > 0:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision training enabled (float16)")
            else:
                logger.info("Mixed precision not enabled (no GPU available)")
                
        except Exception as e:
            logger.warning(f"Failed to enable mixed precision: {e}")
            logger.info("Continuing with default precision (float32)")

    def build_model(self, use_dropout: bool = True, dropout_rate: float = 0.2) -> keras.Model:
        """
        Build and compile the autoencoder model.

        Architecture:
        - Encoder: Input → Dense(encoding_dim*2, relu) → Dense(encoding_dim, relu)
        - Decoder: Dense(encoding_dim*2, relu) → Dense(input_dim, sigmoid)
        - Optional dropout layers (0.2) for regularization

        Args:
            use_dropout: Whether to add dropout layers for regularization
            dropout_rate: Dropout rate (default: 0.2)

        Returns:
            Compiled Keras model

        Raises:
            ValueError: If dropout_rate is not in valid range [0, 1)
        """
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        logger.info(f"Building autoencoder model with input_dim={self.input_dim}, "
                   f"encoding_dim={self.encoding_dim}, use_dropout={use_dropout}")

        # Input layer
        input_layer = keras.Input(shape=(self.input_dim,), name='input')

        # Encoder
        encoded = layers.Dense(
            self.encoding_dim * 2,
            activation='relu',
            name='encoder_layer1'
        )(input_layer)

        if use_dropout:
            encoded = layers.Dropout(dropout_rate, name='encoder_dropout')(encoded)

        encoded = layers.Dense(
            self.encoding_dim,
            activation='relu',
            name='encoder_layer2'
        )(encoded)

        # Decoder
        decoded = layers.Dense(
            self.encoding_dim * 2,
            activation='relu',
            name='decoder_layer1'
        )(encoded)

        if use_dropout:
            decoded = layers.Dropout(dropout_rate, name='decoder_dropout')(decoded)

        decoded = layers.Dense(
            self.input_dim,
            activation='sigmoid',
            name='decoder_output'
        )(decoded)

        # Create model
        self.model = keras.Model(inputs=input_layer, outputs=decoded, name='autoencoder')

        # Compile model with MSE loss and Adam optimizer
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        logger.info(f"Model compiled with Adam optimizer (lr={self.learning_rate}) and MSE loss")
        logger.info(f"Model summary: {self.model.count_params()} total parameters")

        return self.model

    def train(self, X_train: np.ndarray, X_val: np.ndarray) -> keras.callbacks.History:
        """
        Train autoencoder on benign traffic only.

        This method trains the autoencoder using only benign samples, with early stopping
        to prevent overfitting. Training progress is logged, and the model is saved
        at checkpoints.

        Args:
            X_train: Benign training samples, shape (n_samples, n_features)
            X_val: Benign validation samples, shape (n_samples, n_features)

        Returns:
            Training history object containing loss and metrics per epoch

        Raises:
            ValueError: If model is not built, or if input shapes don't match expected dimensions
            RuntimeError: If training fails
        """
        # Validate model is built
        if self.model is None:
            raise ValueError("Model must be built before training. Call build_model() first.")

        # Validate input shapes
        if X_train.shape[1] != self.input_dim:
            raise ValueError(
                f"X_train feature dimension {X_train.shape[1]} doesn't match "
                f"expected input_dim {self.input_dim}"
            )

        if X_val.shape[1] != self.input_dim:
            raise ValueError(
                f"X_val feature dimension {X_val.shape[1]} doesn't match "
                f"expected input_dim {self.input_dim}"
            )

        # Validate sufficient samples
        if X_train.shape[0] < self.batch_size:
            logger.warning(
                f"Training samples ({X_train.shape[0]}) less than batch_size ({self.batch_size}). "
                f"Consider reducing batch_size."
            )

        logger.info(f"Starting autoencoder training with {X_train.shape[0]} training samples "
                   f"and {X_val.shape[0]} validation samples")
        logger.info(f"Training configuration: epochs={self.epochs}, batch_size={self.batch_size}, "
                   f"early_stopping_patience={self.early_stopping_patience}")

        # Set up callbacks
        callbacks = []

        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # Model checkpoint callback (save best model)
        checkpoint_path = self.config.get('model_save_path', 'models/')
        checkpoint_file = None
        if checkpoint_path:
            # Create directory if it doesn't exist
            os.makedirs(checkpoint_path, exist_ok=True)

            checkpoint_file = os.path.join(checkpoint_path, 'autoencoder_best.keras')
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_file,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(model_checkpoint)
            logger.info(f"Model checkpoints will be saved to: {checkpoint_file}")

        # Custom logging callback for progress tracking
        class TrainingLogger(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                logger.info(
                    f"Epoch {epoch + 1}/{self.params['epochs']}: "
                    f"loss={logs.get('loss', 0):.6f}, "
                    f"val_loss={logs.get('val_loss', 0):.6f}, "
                    f"mae={logs.get('mae', 0):.6f}, "
                    f"val_mae={logs.get('val_mae', 0):.6f}"
                )

        callbacks.append(TrainingLogger())

        try:
            # Train the model
            # Note: For autoencoder, input and output are the same (reconstruction task)
            logger.info("Training started...")
            history = self.model.fit(
                X_train, X_train,  # Input and target are the same
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, X_val),
                callbacks=callbacks,
                verbose=0  # Suppress default output, use custom logger
            )

            logger.info("Training completed successfully")

            # Log final metrics
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            logger.info(f"Final training loss: {final_train_loss:.6f}")
            logger.info(f"Final validation loss: {final_val_loss:.6f}")

            # Check if early stopping was triggered
            if len(history.history['loss']) < self.epochs:
                logger.info(
                    f"Early stopping triggered at epoch {len(history.history['loss'])} "
                    f"(patience={self.early_stopping_patience})"
                )

            # Persist final model to guarantee artifact freshness even when
            # callback checkpointing is skipped or not triggered as expected.
            if checkpoint_file:
                self.model.save(checkpoint_file)
                logger.info(f"Final autoencoder model saved to: {checkpoint_file}")

            return history

        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise RuntimeError(f"Autoencoder training failed: {str(e)}") from e

    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for input samples.

        This method processes input data through the trained autoencoder and computes
        the Mean Squared Error (MSE) between the original input and its reconstruction.
        Higher reconstruction errors indicate potential anomalies.

        The computation is performed in batches for memory efficiency, making it
        suitable for large datasets.

        Args:
            X: Input samples, shape (n_samples, n_features)

        Returns:
            Array of reconstruction errors (MSE per sample), shape (n_samples,)

        Raises:
            ValueError: If model is not trained, or if input shape doesn't match expected dimensions
        """
        # Validate model is trained
        if self.model is None:
            raise ValueError("Model must be built and trained before computing reconstruction error. "
                           "Call build_model() and train() first.")

        # Validate input shape
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Input feature dimension {X.shape[1]} doesn't match "
                f"expected input_dim {self.input_dim}"
            )

        logger.info(f"Computing reconstruction error for {X.shape[0]} samples")

        # Process in batches for memory efficiency
        n_samples = X.shape[0]
        reconstruction_errors = np.zeros(n_samples)

        # Process data in batches
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = X[start_idx:end_idx]

            # Get reconstructions from the model
            reconstructions = self.model.predict(batch, verbose=0)

            # Compute MSE for each sample in the batch
            # MSE = mean((original - reconstruction)^2) across features
            mse_per_sample = np.mean(np.square(batch - reconstructions), axis=1)

            # Store the errors
            reconstruction_errors[start_idx:end_idx] = mse_per_sample

        logger.info(f"Reconstruction error computation completed. "
                   f"Mean error: {np.mean(reconstruction_errors):.6f}, "
                   f"Std: {np.std(reconstruction_errors):.6f}")

        return reconstruction_errors


