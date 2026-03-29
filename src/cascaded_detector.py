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
        self.stage1_base_threshold = None
        
        # Stage 2 component
        self.classifier = None
        
        # Configuration
        self.stage1_enabled = runtime_config.get('stage1', {}).get('enabled', True)
        self.stage2_enabled = runtime_config.get('stage2', {}).get('enabled', True)
        stage1_cfg = runtime_config.get('stage1', {}) if isinstance(runtime_config.get('stage1', {}), dict) else {}
        stage2_cfg = runtime_config.get('stage2', {}) if isinstance(runtime_config.get('stage2', {}), dict) else {}

        self.stage1_runtime_threshold_scale = float(stage1_cfg.get('runtime_threshold_scale', 1.0))
        if self.stage1_runtime_threshold_scale <= 0.0:
            raise ValueError('cascaded_ids.stage1.runtime_threshold_scale must be > 0')

        self.stage2_medium_requires_base_stage1 = bool(
            stage2_cfg.get('medium_requires_base_stage1', True)
        )

        legacy_threshold = stage2_cfg.get('attack_probability_threshold')
        if legacy_threshold is not None:
            self.stage2_attack_probability_threshold_high = float(legacy_threshold)
            self.stage2_attack_probability_threshold_medium = float(legacy_threshold)
        else:
            self.stage2_attack_probability_threshold_high = float(
                stage2_cfg.get('attack_probability_threshold_high', 0.50)
            )
            self.stage2_attack_probability_threshold_medium = float(
                stage2_cfg.get('attack_probability_threshold_medium', 0.30)
            )

        if not (0.0 < self.stage2_attack_probability_threshold_high < 1.0):
            raise ValueError(
                "cascaded_ids.stage2.attack_probability_threshold_high must be in (0, 1)"
            )
        if not (0.0 < self.stage2_attack_probability_threshold_medium < 1.0):
            raise ValueError(
                "cascaded_ids.stage2.attack_probability_threshold_medium must be in (0, 1)"
            )
        if self.stage2_attack_probability_threshold_medium > self.stage2_attack_probability_threshold_high:
            raise ValueError(
                "cascaded_ids.stage2.attack_probability_threshold_medium must be <= high threshold"
            )
        # Optional Stage-2 minimum classifier confidence to suppress low-confidence alerts
        self.stage2_min_confidence = float(stage2_cfg.get('min_confidence', 0.0))
        if not (0.0 <= self.stage2_min_confidence <= 1.0):
            raise ValueError('cascaded_ids.stage2.min_confidence must be between 0.0 and 1.0')
        
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
        logger.info(f"Stage 2 minimum classifier confidence gate: {self.stage2_min_confidence:.3f}")
    
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
        self.stage1_base_threshold = float(threshold)
        self.stage1_threshold = float(threshold) * self.stage1_runtime_threshold_scale
        
        logger.info("Stage 1 components loaded")
        logger.info(
            "Stage 1 threshold: runtime=%.6f (base=%.6f, scale=%.4f)",
            self.stage1_threshold,
            self.stage1_base_threshold,
            self.stage1_runtime_threshold_scale,
        )
    
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
        
        stage1_flagged = anomaly_score >= self.stage1_threshold
        stage1_medium_eligible = stage1_flagged
        if (
            stage1_medium_eligible
            and self.stage2_medium_requires_base_stage1
            and self.stage1_base_threshold is not None
        ):
            stage1_medium_eligible = anomaly_score >= self.stage1_base_threshold

        if stage1_flagged:
            self.stats['stage1_suspicious'] += 1
        else:
            self.stats['stage1_benign'] += 1
        
        if not self.stage2_enabled or self.classifier is None:
            # Stage 2 disabled, Stage 1 alone decides.
            return {
                'prediction': 'ATTACK' if stage1_flagged else 'BENIGN',
                'stage': 1,
                'anomaly_score': float(anomaly_score),
                'confidence': float(anomaly_score if stage1_flagged else (1.0 - anomaly_score)),
                'latency_ms': (time.time() - start_time) * 1000,
                'stage1_latency_ms': stage1_latency,
                'stage2_latency_ms': 0.0
            }
        
        # Stage 2: Supervised Classification
        stage2_start = time.time()
        stage2_input = self._build_stage2_input(sample[0], float(anomaly_score))
        classifier_result = self.classifier.predict_single(stage2_input)
        stage2_latency = (time.time() - stage2_start) * 1000  # ms
        self.stats['stage2_latency_ms'].append(stage2_latency)
        
        class_label = classifier_result['class_label']
        attack_probability = self._extract_attack_probability(
            class_label,
            classifier_result.get('probabilities', {}),
        )
        # Classifier outputs
        classifier_confidence = float(classifier_result.get('confidence', 0.0))
        stage2_high = attack_probability > self.stage2_attack_probability_threshold_high
        stage2_medium = attack_probability > self.stage2_attack_probability_threshold_medium
        final_attack = bool(stage2_high or (stage1_medium_eligible and stage2_medium))

        # Apply optional minimum-confidence gate: suppress low-confidence ATTACK predictions
        suppressed_by_min_conf = False
        if final_attack and classifier_confidence < self.stage2_min_confidence:
            suppressed_by_min_conf = True
            if self.log_stage2_decisions:
                logger.debug(
                    f"Stage 2: suppressed ATTACK by min_confidence gate (attack_prob={attack_probability:.4f}, "
                    f"class_conf={classifier_confidence:.4f}, min_conf={self.stage2_min_confidence:.4f})"
                )
            final_attack = False

        if not final_attack:
            self.stats['stage2_benign'] += 1

            note_text = 'Smart gate kept sample benign'
            if suppressed_by_min_conf:
                note_text = 'Suppressed by stage2.min_confidence'

            result = {
                'prediction': 'BENIGN',
                'stage': 2,
                'anomaly_score': float(anomaly_score),
                'classifier_confidence': classifier_result['confidence'],
                'confidence': float(1.0 - attack_probability),
                'note': note_text,
                'stage1_flagged': bool(stage1_flagged),
                'stage1_medium_eligible': bool(stage1_medium_eligible),
                'stage2_high': bool(stage2_high),
                'stage2_medium': bool(stage2_medium),
                'attack_probability': float(attack_probability),
                'latency_ms': (time.time() - start_time) * 1000,
                'stage1_latency_ms': stage1_latency,
                'stage2_latency_ms': stage2_latency
            }

            if self.log_stage2_decisions:
                logger.debug(
                    f"Stage 2: BENIGN by smart gate (anomaly={anomaly_score:.4f}, "
                    f"attack_prob={attack_probability:.4f}, "
                    f"high={self.stage2_attack_probability_threshold_high:.4f}, "
                    f"medium={self.stage2_attack_probability_threshold_medium:.4f}, "
                    f"stage1_flagged={stage1_flagged}, "
                    f"stage1_medium_eligible={stage1_medium_eligible})"
                )

            return result
        else:
            self.stats['stage2_attack'] += 1

            resolved_attack_type = self._resolve_attack_type(
                class_label,
                classifier_result.get('probabilities', {}),
            )

            result = {
                'prediction': 'ATTACK',
                'attack_type': resolved_attack_type,
                'stage2_predicted_label': self._format_attack_type(class_label),
                'stage': 2,
                'anomaly_score': float(anomaly_score),
                'classifier_confidence': classifier_result['confidence'],
                'confidence': float(attack_probability),
                'stage1_flagged': bool(stage1_flagged),
                'stage1_medium_eligible': bool(stage1_medium_eligible),
                'stage2_high': bool(stage2_high),
                'stage2_medium': bool(stage2_medium),
                'attack_probability': float(attack_probability),
                'probabilities': classifier_result['probabilities'],
                'top_features': classifier_result['top_features'],
                'latency_ms': (time.time() - start_time) * 1000,
                'stage1_latency_ms': stage1_latency,
                'stage2_latency_ms': stage2_latency
            }

            if self.log_stage2_decisions:
                logger.debug(
                    f"Stage 2: ATTACK detected - {class_label} "
                    f"(attack_prob={attack_probability:.4f}, "
                    f"high={self.stage2_attack_probability_threshold_high:.4f}, "
                    f"medium={self.stage2_attack_probability_threshold_medium:.4f}, "
                    f"stage1_flagged={stage1_flagged}, "
                    f"stage1_medium_eligible={stage1_medium_eligible})"
                )

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
###### IF ANYTHING HAPPENS CHECK THIS STUFF OUT
        # DEBUG PRINT for raw and normalized scores
        '''print(f"DEBUG recon_raw={recon_error:.4f} recon_norm={recon_norm[0]:.4f} "
              f"iso_raw={iso_score:.4f} iso_norm={iso_norm[0]:.4f} "
              f"fused={combined_score:.4f} threshold={self.stage1_threshold:.4f}")'''

        return combined_score

    @staticmethod
    def _is_benign_stage2_label(class_label) -> bool:
        """Return True if Stage-2 predicted benign for common label encodings."""
        # Numeric binary encoding: 0=benign, 1=attack
        if isinstance(class_label, (int, np.integer, float, np.floating)):
            return int(class_label) == 0

        label_text = str(class_label).strip().upper()
        return label_text in {'BENIGN', '0'}

    @staticmethod
    def _format_attack_type(class_label) -> str:
        """Map classifier class label to a human-readable attack type."""
        if isinstance(class_label, (int, np.integer, float, np.floating)):
            return 'ATTACK' if int(class_label) != 0 else 'BENIGN'

        label_text = str(class_label).strip()
        if label_text:
            return label_text
        return 'ATTACK'

    @staticmethod
    def _extract_attack_probability(class_label, probabilities: Dict) -> float:
        """Compute P(attack) from classifier probabilities across label schemas."""
        if isinstance(probabilities, dict) and probabilities:
            # Binary numeric schema: 0=benign, 1=attack
            attack_key = None
            benign_key = None
            for key in probabilities.keys():
                if isinstance(key, (int, np.integer, float, np.floating)):
                    if int(key) == 1:
                        attack_key = key
                    elif int(key) == 0:
                        benign_key = key
                else:
                    key_text = str(key).strip().upper()
                    if key_text in {'ATTACK', 'MALICIOUS'}:
                        attack_key = key
                    elif key_text == 'BENIGN':
                        benign_key = key

            if attack_key is not None:
                return float(probabilities[attack_key])
            if benign_key is not None:
                return float(1.0 - float(probabilities[benign_key]))

            # Multiclass fallback: attack probability is max non-benign class probability.
            non_benign = []
            for key, val in probabilities.items():
                if isinstance(key, (int, np.integer, float, np.floating)) and int(key) == 0:
                    continue
                if str(key).strip().upper() == 'BENIGN':
                    continue
                non_benign.append(float(val))
            if non_benign:
                return float(max(non_benign))

        # Fallback when probabilities are not available.
        return 0.0 if CascadedDetector._is_benign_stage2_label(class_label) else 1.0

    @staticmethod
    def _resolve_attack_type(class_label, probabilities: Dict) -> str:
        """Return a consistent attack type label for final ATTACK decisions.

        If the top-1 class is benign (possible with custom gate thresholds),
        choose the highest-probability non-benign class from probabilities.
        """
        if not CascadedDetector._is_benign_stage2_label(class_label):
            return CascadedDetector._format_attack_type(class_label)

        if isinstance(probabilities, dict) and probabilities:
            best_label = None
            best_prob = -1.0
            for key, val in probabilities.items():
                if isinstance(key, (int, np.integer, float, np.floating)) and int(key) == 0:
                    continue
                if str(key).strip().upper() == 'BENIGN':
                    continue

                prob = float(val)
                if prob > best_prob:
                    best_prob = prob
                    best_label = key

            if best_label is not None and best_prob > 0.0:
                return CascadedDetector._format_attack_type(best_label)

        return 'Network Attack'

    def _build_stage2_input(self, sample_row: np.ndarray, anomaly_score: float) -> np.ndarray:
        """Build Stage 2 feature vector with backward-compatible dimensionality.

        Supported classifier input layouts:
        - 1 feature: [fused_score]
        - N features: raw flow feature vector
        - N+1 features: [fused_score, raw flow features...]
        """
        if self.classifier is None or self.classifier.model is None:
            return sample_row

        expected = getattr(self.classifier.model, 'n_features_in_', None)
        if expected is None:
            return sample_row

        raw_dim = int(sample_row.shape[0])
        if expected == 1:
            return np.array([anomaly_score], dtype=np.float64)
        if expected == raw_dim + 1:
            return np.concatenate([np.array([anomaly_score], dtype=np.float64), sample_row])
        if expected == raw_dim:
            return sample_row

        raise ValueError(
            f"Stage 2 feature dimension mismatch: classifier expects {expected}, "
            f"but runtime can provide 1, {raw_dim}, or {raw_dim + 1}."
        )
    
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
