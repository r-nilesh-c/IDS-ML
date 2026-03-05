"""
Healthcare alert system for logging and reporting.

This module handles anomaly logging, evaluation metrics computation,
and deployment readiness assessment.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


class HealthcareAlertSystem:
    """
    Healthcare alert system for anomaly logging and reporting.
    
    This class provides functionality for:
    - Logging detected anomalies in JSON Lines format
    - Computing comprehensive evaluation metrics
    - Generating visualizations (ROC curve, PR curve, confusion matrix)
    - Assessing deployment readiness based on healthcare criteria
    
    The system prioritizes low false positive rates and high recall
    to meet healthcare deployment requirements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert system.
        
        Args:
            config: Dictionary with:
                - log_path: Path to anomaly log file (default: 'logs/anomalies.jsonl')
                - report_path: Path to report directory (default: 'reports/')
                
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        
        # Extract paths with defaults
        self.log_path = config.get('log_path', 'logs/anomalies.jsonl')
        self.report_path = config.get('report_path', 'reports/')
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(self.report_path, exist_ok=True)
        
        logger.info(
            f"HealthcareAlertSystem initialized with log_path={self.log_path}, "
            f"report_path={self.report_path}"
        )
    
    def log_anomaly(self, timestamp: str, flow_features: Dict[str, Any],
                   anomaly_score: float, prediction: int) -> None:
        """
        Log detected anomaly with details.
        
        Writes anomaly information to log file in JSON Lines format.
        Each line is a valid JSON object containing timestamp, score,
        prediction, and flow features.
        
        Args:
            timestamp: Detection timestamp (ISO 8601 format recommended)
            flow_features: Dictionary of flow characteristics
            anomaly_score: Combined anomaly score
            prediction: Binary classification (0 or 1)
            
        Raises:
            ValueError: If inputs are invalid
            IOError: If log file write fails
        """
        # Validate inputs
        if not timestamp:
            raise ValueError("timestamp cannot be empty")
        
        if flow_features is None:
            raise ValueError("flow_features cannot be None")
        
        if not isinstance(anomaly_score, (int, float)):
            raise ValueError(f"anomaly_score must be numeric, got {type(anomaly_score)}")
        
        if not np.isfinite(anomaly_score):
            raise ValueError("anomaly_score must be finite")
        
        if prediction not in [0, 1]:
            raise ValueError(f"prediction must be 0 or 1, got {prediction}")
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "anomaly_score": float(anomaly_score),
            "prediction": int(prediction),
            "flow_features": flow_features
        }
        
        # Write to log file (append mode)
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            logger.debug(f"Logged anomaly: timestamp={timestamp}, score={anomaly_score:.6f}, prediction={prediction}")
            
        except IOError as e:
            logger.error(f"Failed to write to log file {self.log_path}: {e}")
            raise IOError(f"Failed to write to log file: {e}") from e
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_scores: np.ndarray, 
                                   attack_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation metrics.
        
        Computes all required metrics including:
        - Overall accuracy, precision, recall, F1-score
        - Per-class metrics for each attack category
        - False positive rate (overall and per-class)
        - ROC-AUC score
        - Confusion matrix
        - Generates ROC curve and Precision-Recall curve visualizations
        
        Args:
            y_true: True labels (0=benign, 1=attack)
            y_pred: Predicted labels (0=benign, 1=attack)
            y_scores: Anomaly scores (continuous values)
            attack_labels: Optional list of attack category names for detailed reporting
            
        Returns:
            Dictionary containing all metrics and paths to visualizations
            
        Raises:
            ValueError: If inputs are invalid or have mismatched lengths
        """
        # Validate inputs
        if y_true is None or len(y_true) == 0:
            raise ValueError("y_true cannot be empty")
        
        if y_pred is None or len(y_pred) == 0:
            raise ValueError("y_pred cannot be empty")
        
        if y_scores is None or len(y_scores) == 0:
            raise ValueError("y_scores cannot be empty")
        
        if len(y_true) != len(y_pred) or len(y_true) != len(y_scores):
            raise ValueError(
                f"Input lengths must match: y_true={len(y_true)}, "
                f"y_pred={len(y_pred)}, y_scores={len(y_scores)}"
            )
        
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("y_true must contain only 0 or 1")
        
        if not np.all((y_pred == 0) | (y_pred == 1)):
            raise ValueError("y_pred must contain only 0 or 1")
        
        if not np.all(np.isfinite(y_scores)):
            raise ValueError("y_scores must be finite")
        
        logger.info(f"Generating evaluation report for {len(y_true)} samples")
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Compute overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Compute macro F1-score (average of per-class F1 scores)
        metrics['macro_f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['confusion_matrix'] = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
        
        # Compute False Positive Rate
        if (fp + tn) > 0:
            metrics['false_positive_rate'] = fp / (fp + tn)
        else:
            metrics['false_positive_rate'] = 0.0
        
        # Compute ROC-AUC score
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics['roc_auc'] = None
        
        # Log overall metrics
        logger.info(f"Overall Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Macro F1-Score: {metrics['macro_f1_score']:.4f}")
        logger.info(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        if metrics['roc_auc'] is not None:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Compute per-class metrics if attack labels provided
        if attack_labels is not None:
            metrics['per_class_metrics'] = self._compute_per_class_metrics(
                y_true, y_pred, attack_labels
            )
        
        # Generate visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ROC Curve
        roc_curve_path = os.path.join(self.report_path, f'roc_curve_{timestamp}.png')
        self._plot_roc_curve(y_true, y_scores, roc_curve_path)
        metrics['roc_curve_path'] = roc_curve_path
        
        # Precision-Recall Curve
        pr_curve_path = os.path.join(self.report_path, f'pr_curve_{timestamp}.png')
        self._plot_precision_recall_curve(y_true, y_scores, pr_curve_path)
        metrics['pr_curve_path'] = pr_curve_path
        
        # Confusion Matrix
        cm_path = os.path.join(self.report_path, f'confusion_matrix_{timestamp}.png')
        self._plot_confusion_matrix(cm, cm_path)
        metrics['confusion_matrix_path'] = cm_path
        
        logger.info(f"Evaluation report generated. Visualizations saved to {self.report_path}")
        
        return metrics
    
    def _compute_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   attack_labels: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics for each attack category.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            attack_labels: List of attack category names
            
        Returns:
            Dictionary mapping attack categories to their metrics
        """
        per_class = {}
        
        # Get unique attack types
        unique_labels = np.unique(attack_labels)
        
        for label in unique_labels:
            # Create binary mask for this attack type
            mask = np.array([l == label for l in attack_labels])
            
            if np.sum(mask) == 0:
                continue
            
            y_true_class = y_true[mask]
            y_pred_class = y_pred[mask]
            
            # Compute metrics for this class
            per_class[label] = {
                'precision': precision_score(y_true_class, y_pred_class, zero_division=0),
                'recall': recall_score(y_true_class, y_pred_class, zero_division=0),
                'f1_score': f1_score(y_true_class, y_pred_class, zero_division=0),
                'support': int(np.sum(mask))
            }
            
            # Compute FPR for this class
            cm_class = confusion_matrix(y_true_class, y_pred_class)
            if cm_class.size == 4:
                tn, fp, fn, tp = cm_class.ravel()
                if (fp + tn) > 0:
                    per_class[label]['false_positive_rate'] = fp / (fp + tn)
                else:
                    per_class[label]['false_positive_rate'] = 0.0
        
        return per_class
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                       save_path: str) -> None:
        """
        Generate and save ROC curve visualization.
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            save_path: Path to save the plot
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = roc_auc_score(y_true, y_scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curve saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate ROC curve: {e}")
    
    def _plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                     save_path: str) -> None:
        """
        Generate and save Precision-Recall curve visualization.
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            save_path: Path to save the plot
        """
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Precision-Recall curve saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate Precision-Recall curve: {e}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: str) -> None:
        """
        Generate and save confusion matrix visualization.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Benign', 'Attack'],
                       yticklabels=['Benign', 'Attack'])
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate confusion matrix: {e}")
    
    def assess_deployment_readiness(self, metrics: Dict[str, Any]) -> str:
        """
        Assess whether system meets healthcare deployment criteria.
        
        Checks if the system meets the following criteria:
        - False Positive Rate < 5% (acceptable for healthcare)
        - Recall > 90% (high sensitivity to attacks)
        
        Args:
            metrics: Dictionary of evaluation metrics from generate_evaluation_report()
            
        Returns:
            Deployment readiness assessment string with justification
            
        Raises:
            ValueError: If required metrics are missing
        """
        # Validate required metrics are present
        required_metrics = ['false_positive_rate', 'recall']
        for metric in required_metrics:
            if metric not in metrics:
                raise ValueError(f"Required metric '{metric}' not found in metrics dictionary")
        
        fpr = metrics['false_positive_rate']
        recall = metrics['recall']
        
        # Healthcare deployment criteria
        max_fpr = 0.05  # 5%
        min_recall = 0.90  # 90%
        
        # Check criteria
        fpr_ok = fpr < max_fpr
        recall_ok = recall > min_recall
        
        # Generate assessment
        if fpr_ok and recall_ok:
            assessment = "READY FOR DEPLOYMENT"
            justification = (
                f"System meets healthcare deployment criteria:\n"
                f"  ✓ False Positive Rate: {fpr:.4f} < {max_fpr:.4f} (target)\n"
                f"  ✓ Recall: {recall:.4f} > {min_recall:.4f} (target)\n"
                f"\nThe system demonstrates acceptable false positive rate for clinical "
                f"environments while maintaining high sensitivity to attacks."
            )
        else:
            assessment = "NOT READY FOR DEPLOYMENT"
            reasons = []
            
            if not fpr_ok:
                reasons.append(
                    f"  ✗ False Positive Rate: {fpr:.4f} >= {max_fpr:.4f} (target) - "
                    f"Too many false alarms for healthcare setting"
                )
            else:
                reasons.append(f"  ✓ False Positive Rate: {fpr:.4f} < {max_fpr:.4f} (target)")
            
            if not recall_ok:
                reasons.append(
                    f"  ✗ Recall: {recall:.4f} <= {min_recall:.4f} (target) - "
                    f"Insufficient attack detection sensitivity"
                )
            else:
                reasons.append(f"  ✓ Recall: {recall:.4f} > {min_recall:.4f} (target)")
            
            justification = (
                f"System does NOT meet healthcare deployment criteria:\n" +
                "\n".join(reasons) +
                f"\n\nRecommendations:\n"
            )
            
            if not fpr_ok:
                justification += "  - Increase detection threshold to reduce false positives\n"
                justification += "  - Retrain with more diverse benign traffic samples\n"
            
            if not recall_ok:
                justification += "  - Decrease detection threshold to improve sensitivity\n"
                justification += "  - Retrain with more attack samples\n"
                justification += "  - Adjust fusion weights to prioritize recall\n"
        
        full_assessment = f"{assessment}\n\n{justification}"
        
        logger.info(f"Deployment readiness: {assessment}")
        logger.info(f"FPR: {fpr:.4f}, Recall: {recall:.4f}")
        
        return full_assessment
