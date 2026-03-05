"""
Fusion module for combining autoencoder and isolation forest scores.

This module implements score normalization, weighted combination, and
dynamic thresholding for the hybrid IDS.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Any


logger = logging.getLogger(__name__)


class FusionModule:
    """
    Fusion module for combining detection scores.
    
    This class normalizes scores from both detection modules (autoencoder and
    isolation forest), combines them using weighted averaging, and applies
    dynamic thresholding based on benign validation distribution.
    
    The fusion strategy prioritizes low false positive rates for healthcare
    deployment while maintaining high sensitivity to attacks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fusion module.
        
        Args:
            config: Dictionary with:
                - weight_autoencoder: Weight for autoencoder scores (default: 0.5)
                - weight_isolation: Weight for isolation forest scores (default: 0.5)
                - percentile: Percentile for threshold computation (default: 95)
                
        Raises:
            ValueError: If weights don't sum to 1.0 or percentile is invalid
        """
        self.config = config
        
        # Extract weights with defaults
        self.weight_autoencoder = config.get('weight_autoencoder', 0.5)
        self.weight_isolation = config.get('weight_isolation', 0.5)
        
        # Validate weights sum to 1.0
        weight_sum = self.weight_autoencoder + self.weight_isolation
        if not np.isclose(weight_sum, 1.0, rtol=1e-5):
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum} "
                f"(autoencoder={self.weight_autoencoder}, isolation={self.weight_isolation})"
            )
        
        # Extract percentile with default
        self.percentile = config.get('percentile', 95)
        
        # Validate percentile
        if not (0 < self.percentile < 100):
            raise ValueError(
                f"Percentile must be in range (0, 100), got {self.percentile}"
            )
        
        # Initialize storage for normalization statistics
        self.recon_min = None
        self.recon_max = None
        self.iso_min = None
        self.iso_max = None
        self.threshold = None
        
        logger.info(
            f"FusionModule initialized with weights: "
            f"autoencoder={self.weight_autoencoder}, "
            f"isolation={self.weight_isolation}, "
            f"percentile={self.percentile}"
        )
    
    def fit_threshold(self, recon_errors_benign: np.ndarray, 
                     iso_scores_benign: np.ndarray) -> None:
        """
        Compute threshold from benign validation distribution.
        
        This method:
        1. Computes min/max statistics for normalization from benign validation set
        2. Normalizes the benign scores
        3. Computes combined scores
        4. Sets threshold at specified percentile
        
        Args:
            recon_errors_benign: Reconstruction errors on benign validation set
            iso_scores_benign: Isolation scores on benign validation set
            
        Raises:
            ValueError: If inputs are empty, have mismatched lengths, or contain invalid values
        """
        # Validate inputs
        if recon_errors_benign is None or len(recon_errors_benign) == 0:
            raise ValueError("recon_errors_benign cannot be empty")
        
        if iso_scores_benign is None or len(iso_scores_benign) == 0:
            raise ValueError("iso_scores_benign cannot be empty")
        
        if len(recon_errors_benign) != len(iso_scores_benign):
            raise ValueError(
                f"Input lengths must match: recon_errors={len(recon_errors_benign)}, "
                f"iso_scores={len(iso_scores_benign)}"
            )
        
        if not np.all(np.isfinite(recon_errors_benign)):
            raise ValueError("recon_errors_benign contains NaN or infinite values")
        
        if not np.all(np.isfinite(iso_scores_benign)):
            raise ValueError("iso_scores_benign contains NaN or infinite values")
        
        logger.info(
            f"Fitting threshold on {len(recon_errors_benign)} benign validation samples"
        )
        
        # Compute normalization statistics
        self.recon_min = np.min(recon_errors_benign)
        self.recon_max = np.max(recon_errors_benign)
        self.iso_min = np.min(iso_scores_benign)
        self.iso_max = np.max(iso_scores_benign)
        
        logger.info(
            f"Normalization statistics computed: "
            f"recon_min={self.recon_min:.6f}, recon_max={self.recon_max:.6f}, "
            f"iso_min={self.iso_min:.6f}, iso_max={self.iso_max:.6f}"
        )
        
        # Normalize benign scores
        recon_normalized, iso_normalized = self.normalize_scores(
            recon_errors_benign, iso_scores_benign
        )
        
        # Compute combined scores
        combined_scores_benign = self.compute_combined_score(
            recon_errors_benign, iso_scores_benign
        )
        
        # Compute threshold at specified percentile
        self.threshold = np.percentile(combined_scores_benign, self.percentile)
        
        logger.info(
            f"Threshold computed: {self.threshold:.6f} "
            f"(percentile={self.percentile})"
        )
        
        # Log statistics
        logger.info(
            f"Benign validation combined scores: "
            f"mean={np.mean(combined_scores_benign):.6f}, "
            f"std={np.std(combined_scores_benign):.6f}, "
            f"min={np.min(combined_scores_benign):.6f}, "
            f"max={np.max(combined_scores_benign):.6f}"
        )
    
    def normalize_scores(self, recon_errors: np.ndarray, 
                        iso_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize scores to [0, 1] range using min-max scaling.
        
        Uses normalization statistics computed from benign validation set.
        Test scores are clipped to [0, 1] range.
        
        Args:
            recon_errors: Raw reconstruction errors
            iso_scores: Raw isolation scores
            
        Returns:
            Tuple of (normalized_recon_errors, normalized_iso_scores)
            
        Raises:
            ValueError: If normalization statistics not computed (call fit_threshold first)
            ValueError: If inputs are empty, have mismatched lengths, or contain invalid values
        """
        # Validate that fit_threshold has been called
        if self.recon_min is None or self.recon_max is None:
            raise ValueError(
                "Normalization statistics not computed. Call fit_threshold() first."
            )
        
        if self.iso_min is None or self.iso_max is None:
            raise ValueError(
                "Normalization statistics not computed. Call fit_threshold() first."
            )
        
        # Validate inputs
        if recon_errors is None or len(recon_errors) == 0:
            raise ValueError("recon_errors cannot be empty")
        
        if iso_scores is None or len(iso_scores) == 0:
            raise ValueError("iso_scores cannot be empty")
        
        if len(recon_errors) != len(iso_scores):
            raise ValueError(
                f"Input lengths must match: recon_errors={len(recon_errors)}, "
                f"iso_scores={len(iso_scores)}"
            )
        
        if not np.all(np.isfinite(recon_errors)):
            raise ValueError("recon_errors contains NaN or infinite values")
        
        if not np.all(np.isfinite(iso_scores)):
            raise ValueError("iso_scores contains NaN or infinite values")
        
        # Normalize reconstruction errors
        recon_range = self.recon_max - self.recon_min
        if recon_range == 0:
            # All benign samples have same reconstruction error
            logger.warning(
                "Reconstruction error range is zero. Setting all normalized values to 0.5"
            )
            recon_normalized = np.full_like(recon_errors, 0.5, dtype=np.float64)
        else:
            recon_normalized = (recon_errors - self.recon_min) / recon_range
            # Clip to [0, 1] range
            recon_normalized = np.clip(recon_normalized, 0.0, 1.0)
        
        # Normalize isolation scores
        iso_range = self.iso_max - self.iso_min
        if iso_range == 0:
            # All benign samples have same isolation score
            logger.warning(
                "Isolation score range is zero. Setting all normalized values to 0.5"
            )
            iso_normalized = np.full_like(iso_scores, 0.5, dtype=np.float64)
        else:
            iso_normalized = (iso_scores - self.iso_min) / iso_range
            # Clip to [0, 1] range
            iso_normalized = np.clip(iso_normalized, 0.0, 1.0)
        
        return recon_normalized, iso_normalized
    
    def compute_combined_score(self, recon_errors: np.ndarray, 
                              iso_scores: np.ndarray) -> np.ndarray:
        """
        Compute weighted average of normalized scores.
        
        This method normalizes the input scores and computes their weighted average:
        combined_score = w_ae * normalized_recon_error + w_if * normalized_iso_score
        
        Args:
            recon_errors: Reconstruction errors
            iso_scores: Isolation scores
            
        Returns:
            Combined anomaly scores in range [0, 1]
            
        Raises:
            ValueError: If inputs are invalid or normalization fails
        """
        # Normalize scores
        recon_normalized, iso_normalized = self.normalize_scores(
            recon_errors, iso_scores
        )
        
        # Compute weighted average
        combined_scores = (
            self.weight_autoencoder * recon_normalized +
            self.weight_isolation * iso_normalized
        )
        
        # Verify output is in [0, 1] range
        assert np.all(combined_scores >= 0) and np.all(combined_scores <= 1), \
            "Combined scores must be in [0, 1] range"
        
        return combined_scores
    
    def classify(self, combined_scores: np.ndarray) -> np.ndarray:
        """
        Apply threshold to classify samples.
        
        Args:
            combined_scores: Combined anomaly scores
            
        Returns:
            Binary predictions (0=benign, 1=anomaly)
            
        Raises:
            ValueError: If threshold not computed (call fit_threshold first)
            ValueError: If combined_scores is empty or contains invalid values
        """
        # Validate that fit_threshold has been called
        if self.threshold is None:
            raise ValueError(
                "Threshold not computed. Call fit_threshold() first."
            )
        
        # Validate input
        if combined_scores is None or len(combined_scores) == 0:
            raise ValueError("combined_scores cannot be empty")
        
        if not np.all(np.isfinite(combined_scores)):
            raise ValueError("combined_scores contains NaN or infinite values")
        
        # Apply threshold
        predictions = (combined_scores > self.threshold).astype(int)
        
        return predictions
