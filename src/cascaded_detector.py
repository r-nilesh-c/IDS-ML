"""
Cascaded Hybrid IDS - Two-Stage Detection System.

This module implements the cascaded inference pipeline:
Stage 1: Anomaly Detection (Zero-Day Layer)
Stage 2: Supervised Classification (Refinement Layer)

The system achieves:
- Zero-day attack detection (Stage 1)
- False positive reduction (Stage 2)
- Attack type classification
- Healthcare-grade performance (FPR <5%, Recall >90%)
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.autoencoder import AutoencoderDetector
from src.isolation_forest import IsolationForestDetector
from src.fusion import FusionModule
from src.supervised_classifier import SupervisedClassifier

logger = logging.getLogger(__name__)


class CascadedDetector:
    """
    Two-stage cascaded intrusion detection system.
    
    Stage 1: Anomaly detection (high recall, moderate FPR)
    Stage 2: Supervised classification (false positive reduction)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize cascaded detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Support both root-level cascaded config and nested 'cascaded_ids' config
        cascaded_config = self.config.get('cascaded_ids')
        if isinstance(cascaded_config, dict):
            runtime_config = cascaded_config
        else:
            runtime_config = self.config
        
        # Stage 1 components
        self.autoencoder = None
        self.isolation_forest = None
        self.fusion = None
        self.stage1_threshold = None
        
        # Stage 2 component
        self.classifier = None
        
        # Configuration
        self.stage1_enabled = runtime_config.get('stage1', {}).get('enabled', True)
        self.stage2_enabled = runtime_config.get('stage2', {}).get('enabled', True)
        
        # Logging configuration
        self.log_stage1_decisions = runtime_config.get('inference', {}).get('log_stage1_decisions', True)
        self.log_stage2_decisions = runtime_config.get('inference', {}).get('log_stage2_decisions', True)
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'stage1_benign': 0,
            'stage1_suspicious': 0,
            'stage2_benign': 0,
            'stage2_attack': 0,
            'stage1_latency_ms': [],
            'stage2_latency_ms': []
        }
        
        logger.info("CascadedDetector initialized")
        logger.info(f"Stage 1 enabled: {self.stage1_enabled}")
        logger.info(f"Stage 2 enabled: {self.stage2_enabled}")
    
    def load_stage1(self, autoencoder: AutoencoderDetector,
                    isolation_forest: IsolationForestDetector,
                    fusion: FusionModule,
                    threshold: float):
        """
        Load Stage 1 anomaly detection components.
        
        Args:
            autoencoder: Trained autoencoder detector
            isolation_forest: Trained isolation forest detector
            fusion: Fitted fusion module
            threshold: Anomaly threshold (e.g., 95th percentile)
        """
        self.autoencoder = autoencoder
        self.isolation_forest = isolation_forest
        self.fusion = fusion
        self.stage1_threshold = threshold
        
        logger.info("Stage 1 components loaded")
        logger.info(f"Stage 1 threshold: {threshold:.6f}")
    
    def load_stage2(self, classifier: SupervisedClassifier):
        """
        Load Stage 2 supervised classifier.
        
        Args:
            classifier: Trained supervised classifier
        """
        self.classifier = classifier
        
        logger.info("Stage 2 classifier loaded")
        logger.info(f"Number of classes: {len(classifier.classes_)}")
    
    def predict_single(self, sample: np.ndarray) -> Dict:
        """
        Predict single sample through cascaded pipeline.
        
        Args:
            sample: Single sample features (n_features,) or (1, n_features)
            
        Returns:
            Dictionary with prediction details:
            - prediction: 'BENIGN' or 'ATTACK'
            - stage: 1 or 2 (which stage made final decision)
            - anomaly_score: Score from Stage 1
            - attack_type: Specific attack type (if Stage 2)
            - confidence: Confidence score
            - latency_ms: Total inference time
            - stage1_latency_ms: Stage 1 time
            - stage2_latency_ms: Stage 2 time (if applicable)
        """
        start_time = time.time()
        
        # Ensure 2D array
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        self.stats['total_samples'] += 1
        
        # Stage 1: Anomaly Detection
        stage1_start = time.time()
        anomaly_score = self._compute_anomaly_score(sample)
        stage1_latency = (time.time() - stage1_start) * 1000  # ms
        self.stats['stage1_latency_ms'].append(stage1_latency)
        
        if anomaly_score < self.stage1_threshold:
            # Fast path: Clearly benign
            self.stats['stage1_benign'] += 1
            
            result = {
                'prediction': 'BENIGN',
                'stage': 1,
                'anomaly_score': float(anomaly_score),
                'confidence': float(1.0 - anomaly_score),
                'latency_ms': (time.time() - start_time) * 1000,
                'stage1_latency_ms': stage1_latency,
                'stage2_latency_ms': 0.0
            }
            
            if self.log_stage1_decisions:
                logger.debug(f"Stage 1: BENIGN (score={anomaly_score:.4f})")
            
            return result
        
        # Sample flagged as suspicious by Stage 1
        self.stats['stage1_suspicious'] += 1
        
        if not self.stage2_enabled or self.classifier is None:
            # Stage 2 disabled, return Stage 1 result
            return {
                'prediction': 'ATTACK',
                'stage': 1,
                'anomaly_score': float(anomaly_score),
                'confidence': float(anomaly_score),
                'latency_ms': (time.time() - start_time) * 1000,
                'stage1_latency_ms': stage1_latency,
                'stage2_latency_ms': 0.0
            }
        
        # Stage 2: Supervised Classification
        stage2_start = time.time()
        classifier_result = self.classifier.predict_single(sample[0])
        stage2_latency = (time.time() - stage2_start) * 1000  # ms
        self.stats['stage2_latency_ms'].append(stage2_latency)
        
        class_label = classifier_result['class_label']
        
        if class_label == 'BENIGN':
            # False positive correction
            self.stats['stage2_benign'] += 1
            
            result = {
                'prediction': 'BENIGN',
                'stage': 2,
                'anomaly_score': float(anomaly_score),
                'classifier_confidence': classifier_result['confidence'],
                'confidence': classifier_result['confidence'],
                'note': 'Flagged by Stage 1 but classified as benign by Stage 2',
                'latency_ms': (time.time() - start_time) * 1000,
                'stage1_latency_ms': stage1_latency,
                'stage2_latency_ms': stage2_latency
            }
            
            if self.log_stage2_decisions:
                logger.debug(f"Stage 2: False positive corrected (anomaly={anomaly_score:.4f}, classifier_conf={classifier_result['confidence']:.4f})")
            
            return result
        else:
            # Confirmed attack
            self.stats['stage2_attack'] += 1
            
            result = {
                'prediction': 'ATTACK',
                'attack_type': class_label,
                'stage': 2,
                'anomaly_score': float(anomaly_score),
                'classifier_confidence': classifier_result['confidence'],
                'confidence': classifier_result['confidence'],
                'probabilities': classifier_result['probabilities'],
                'top_features': classifier_result['top_features'],
                'latency_ms': (time.time() - start_time) * 1000,
                'stage1_latency_ms': stage1_latency,
                'stage2_latency_ms': stage2_latency
            }
            
            if self.log_stage2_decisions:
                logger.debug(f"Stage 2: ATTACK detected - {class_label} (conf={classifier_result['confidence']:.4f})")
            
            return result
    
    def predict_batch(self, X: np.ndarray) -> List[Dict]:
        """
        Predict batch of samples through cascaded pipeline.
        
        Args:
            X: Batch of samples (n_samples, n_features)
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Processing batch of {len(X)} samples")
        
        results = []
        for sample in X:
            result = self.predict_single(sample)
            results.append(result)
        
        return results
    
    def _compute_anomaly_score(self, sample: np.ndarray) -> float:
        """
        Compute anomaly score from Stage 1 components.
        
        Args:
            sample: Single sample (1, n_features)
            
        Returns:
            Combined anomaly score [0, 1]
        """
        # Compute reconstruction error
        recon_error = self.autoencoder.compute_reconstruction_error(sample)[0]
        
        # Compute isolation forest score
        iso_score = self.isolation_forest.compute_anomaly_score(sample)[0]
        
        # Normalize scores together
        recon_norm, iso_norm = self.fusion.normalize_scores(
            np.array([recon_error]),
            np.array([iso_score])
        )
        
        # Compute combined score
        combined_score = (
            self.fusion.weight_autoencoder * recon_norm[0] +
            self.fusion.weight_isolation * iso_norm[0]
        )
        
        return combined_score
    
    def get_statistics(self) -> Dict:
        """
        Get detection statistics.
        
        Returns:
            Dictionary with statistics:
            - total_samples: Total samples processed
            - stage1_benign: Samples classified as benign by Stage 1
            - stage1_suspicious: Samples flagged by Stage 1
            - stage2_benign: False positives corrected by Stage 2
            - stage2_attack: Attacks confirmed by Stage 2
            - avg_stage1_latency_ms: Average Stage 1 latency
            - avg_stage2_latency_ms: Average Stage 2 latency
            - stage1_flagging_rate: Percentage flagged by Stage 1
            - stage2_correction_rate: Percentage corrected by Stage 2
        """
        stats = self.stats.copy()
        
        # Compute averages
        if stats['stage1_latency_ms']:
            stats['avg_stage1_latency_ms'] = np.mean(stats['stage1_latency_ms'])
        else:
            stats['avg_stage1_latency_ms'] = 0.0
        
        if stats['stage2_latency_ms']:
            stats['avg_stage2_latency_ms'] = np.mean(stats['stage2_latency_ms'])
        else:
            stats['avg_stage2_latency_ms'] = 0.0
        
        # Compute rates
        if stats['total_samples'] > 0:
            stats['stage1_flagging_rate'] = stats['stage1_suspicious'] / stats['total_samples']
        else:
            stats['stage1_flagging_rate'] = 0.0
        
        if stats['stage1_suspicious'] > 0:
            stats['stage2_correction_rate'] = stats['stage2_benign'] / stats['stage1_suspicious']
        else:
            stats['stage2_correction_rate'] = 0.0
        
        # Remove raw latency lists
        del stats['stage1_latency_ms']
        del stats['stage2_latency_ms']
        
        return stats
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.stats = {
            'total_samples': 0,
            'stage1_benign': 0,
            'stage1_suspicious': 0,
            'stage2_benign': 0,
            'stage2_attack': 0,
            'stage1_latency_ms': [],
            'stage2_latency_ms': []
        }
        logger.info("Statistics reset")
