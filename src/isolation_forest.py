"""
Isolation Forest anomaly detection module.

This module implements classical machine learning-based anomaly detection
using Isolation Forest.
"""

import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector.
    
    This detector uses the Isolation Forest algorithm to identify anomalies
    by measuring how easily samples can be isolated in feature space.
    Trained exclusively on benign traffic to enable true anomaly detection.
    
    Attributes:
        n_estimators: Number of isolation trees
        max_samples: Number of samples to draw for each tree
        contamination: Expected proportion of outliers (not used during training)
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
        model: Fitted IsolationForest model
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Isolation Forest detector with configuration.
        
        Args:
            config: Dictionary containing:
                - n_estimators: Number of trees (default: 100)
                - max_samples: Samples per tree (default: 256 or 'auto')
                - contamination: Expected outlier proportion (default: 'auto')
                - random_state: Random seed (default: 42)
                - n_jobs: Parallel jobs (default: -1)
        
        Raises:
            ValueError: If configuration parameters are invalid
        """
        self.config = config
        
        # Load configuration with defaults
        self.n_estimators = config.get('n_estimators', 100)
        self.max_samples = config.get('max_samples', 256)
        self.contamination = config.get('contamination', 'auto')
        self.random_state = config.get('random_state', 42)
        self.n_jobs = config.get('n_jobs', -1)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize model (will be fitted during training)
        self.model: Optional[IsolationForest] = None
        
        logger.info(
            f"Initialized IsolationForestDetector with n_estimators={self.n_estimators}, "
            f"max_samples={self.max_samples}, contamination={self.contamination}, "
            f"random_state={self.random_state}, n_jobs={self.n_jobs}"
        )
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if not isinstance(self.n_estimators, int) or self.n_estimators <= 0:
            raise ValueError(f"n_estimators must be a positive integer, got {self.n_estimators}")
        
        if self.max_samples != 'auto':
            if not isinstance(self.max_samples, int) or self.max_samples <= 0:
                raise ValueError(
                    f"max_samples must be 'auto' or a positive integer, got {self.max_samples}"
                )
        
        if self.contamination != 'auto':
            if not isinstance(self.contamination, (int, float)):
                raise ValueError(
                    f"contamination must be 'auto' or a number, got {self.contamination}"
                )
            if not (0 < self.contamination < 0.5):
                raise ValueError(
                    f"contamination must be in range (0, 0.5), got {self.contamination}"
                )
        
        if not isinstance(self.random_state, int) or self.random_state < 0:
            raise ValueError(f"random_state must be a non-negative integer, got {self.random_state}")
        
        if not isinstance(self.n_jobs, int):
            raise ValueError(f"n_jobs must be an integer, got {self.n_jobs}")
    
    def train(self, X_train: np.ndarray) -> None:
        """
        Train Isolation Forest on benign traffic only.
        
        This method fits the Isolation Forest model exclusively on benign samples
        to learn the normal traffic patterns. The model will then identify
        deviations from these patterns as anomalies.
        
        Args:
            X_train: Training features (benign samples only), shape (n_samples, n_features)
        
        Raises:
            ValueError: If X_train is empty, not 2D, or contains invalid values
        
        Requirements:
            - Validates: Requirement 4.3 (train on benign samples only)
        """
        # Validate input
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train cannot be empty")
        
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D array, got shape {X_train.shape}")
        
        if not np.all(np.isfinite(X_train)):
            raise ValueError("X_train contains NaN or infinite values")
        
        n_samples, n_features = X_train.shape
        
        if n_samples < 2:
            raise ValueError(f"X_train must have at least 2 samples, got {n_samples}")
        
        logger.info(f"Training Isolation Forest on {n_samples} benign samples with {n_features} features")
        
        # Create and fit Isolation Forest model
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # Fit on benign training data
        self.model.fit(X_train)
        
        logger.info("Isolation Forest training completed successfully")
    
    def compute_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores using decision_function.
        
        The Isolation Forest's decision_function returns negative scores where
        more negative values indicate more anomalous samples. This method negates
        the scores to make higher values more anomalous, which is more intuitive.
        
        Args:
            X: Input samples, shape (n_samples, n_features)
        
        Returns:
            Array of anomaly scores, shape (n_samples,)
            Higher scores indicate more anomalous samples
        
        Raises:
            ValueError: If model not trained, X is invalid, or dimensions don't match
            
        Requirements:
            - Validates: Requirement 4.4 (output anomaly score)
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model must be trained before computing anomaly scores")
        
        # Validate input
        if X is None or len(X) == 0:
            raise ValueError("X cannot be empty")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        
        # Check feature dimension matches training
        expected_features = self.model.n_features_in_
        if X.shape[1] != expected_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with {expected_features} features"
            )
        
        n_samples = X.shape[0]
        logger.debug(f"Computing anomaly scores for {n_samples} samples")
        
        # Get decision function scores (negative values = more anomalous)
        decision_scores = self.model.decision_function(X)
        
        # Negate scores to make higher values more anomalous
        anomaly_scores = -decision_scores
        
        logger.debug(
            f"Anomaly scores computed: min={np.min(anomaly_scores):.4f}, "
            f"max={np.max(anomaly_scores):.4f}, mean={np.mean(anomaly_scores):.4f}"
        )
        
        return anomaly_scores
