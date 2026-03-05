"""
Supervised Classifier Module for Stage 2 of Cascaded Hybrid IDS.

This module implements a Random Forest classifier that:
1. Reduces false positives from Stage 1 anomaly detection
2. Classifies known attack types
3. Provides feature importance for explainability

The classifier is trained on full labeled data (benign + all attack types)
and only processes samples flagged as suspicious by Stage 1.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class SupervisedClassifier:
    """
    Random Forest classifier for Stage 2 attack classification.
    
    This classifier:
    - Trains on full labeled dataset (benign + attacks)
    - Performs multi-class classification
    - Provides probability scores and feature importance
    - Handles class imbalance with balanced weights
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize supervised classifier.
        
        Args:
            config: Configuration dictionary with hyperparameters
        """
        self.config = config or {}
        self.model = None
        self.classes_ = None
        self.feature_names_ = None
        self.class_mapping_ = None
        
        # Default hyperparameters
        self.n_estimators = self.config.get('n_estimators', 200)
        self.max_depth = self.config.get('max_depth', 20)
        self.min_samples_split = self.config.get('min_samples_split', 5)
        self.min_samples_leaf = self.config.get('min_samples_leaf', 2)
        self.class_weight = self.config.get('class_weight', 'balanced')
        self.n_jobs = self.config.get('n_jobs', -1)
        self.random_state = self.config.get('random_state', 42)
        
        logger.info("SupervisedClassifier initialized")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: Optional[List[str]] = None,
              optimize_hyperparameters: bool = False) -> Dict:
        """
        Train Random Forest classifier on labeled data.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            feature_names: Optional list of feature names
            optimize_hyperparameters: Whether to perform grid search
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training supervised classifier on {len(X_train)} samples")
        logger.info(f"Number of features: {X_train.shape[1]}")
        
        # Store feature names
        if feature_names is not None:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Get unique classes and create mapping
        self.classes_ = np.unique(y_train)
        self.class_mapping_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        
        logger.info(f"Number of classes: {len(self.classes_)}")
        logger.info(f"Classes: {self.classes_}")
        
        # Log class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            logger.info(f"  {cls}: {count} samples ({count/len(y_train)*100:.2f}%)")
        
        if optimize_hyperparameters:
            logger.info("Performing hyperparameter optimization...")
            self.model = self._optimize_hyperparameters(X_train, y_train)
        else:
            logger.info("Training with default hyperparameters...")
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=0
            )
            self.model.fit(X_train, y_train)
        
        # Compute training metrics
        y_pred_train = self.model.predict(X_train)
        train_accuracy = np.mean(y_pred_train == y_train)
        
        logger.info(f"Training completed successfully")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        return {
            'train_accuracy': train_accuracy,
            'n_classes': len(self.classes_),
            'n_features': X_train.shape[1],
            'n_samples': len(X_train)
        }
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, 
                                  y_train: np.ndarray) -> RandomForestClassifier:
        """
        Perform grid search for hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Best Random Forest model
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [15, 20, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        base_model = RandomForestClassifier(
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        # Use stratified k-fold for imbalanced data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        logger.info("Starting grid search...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels and probabilities.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Tuple of (predicted_labels, probability_matrix)
            - predicted_labels: Array of class labels (n_samples,)
            - probability_matrix: Probability for each class (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_single(self, sample: np.ndarray) -> Dict:
        """
        Predict single sample with detailed output.
        
        Args:
            sample: Single sample features (n_features,) or (1, n_features)
            
        Returns:
            Dictionary with prediction details:
            - class_label: Predicted class
            - confidence: Probability of predicted class
            - probabilities: Dict of all class probabilities
            - top_features: Top contributing features
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure 2D array
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        # Get prediction and probabilities
        pred_label = self.model.predict(sample)[0]
        pred_proba = self.model.predict_proba(sample)[0]
        
        # Create probability dictionary
        prob_dict = {cls: float(prob) for cls, prob in zip(self.classes_, pred_proba)}
        
        # Get feature importance for this prediction
        top_features = self.get_top_features(sample[0], n=10)
        
        return {
            'class_label': pred_label,
            'confidence': float(pred_proba[self.class_mapping_[pred_label]]),
            'probabilities': prob_dict,
            'top_features': top_features
        }
    
    def get_feature_importance(self, n: int = 20) -> Dict[str, float]:
        """
        Get global feature importance from trained model.
        
        Args:
            n: Number of top features to return
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importances = self.model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:n]
        
        feature_importance = {}
        for idx in indices:
            feature_name = self.feature_names_[idx]
            feature_importance[feature_name] = float(importances[idx])
        
        return feature_importance
    
    def get_top_features(self, sample: np.ndarray, n: int = 10) -> List[Dict]:
        """
        Get top contributing features for a specific sample.
        
        Args:
            sample: Single sample features (n_features,)
            n: Number of top features to return
            
        Returns:
            List of dicts with feature name, value, and importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get global feature importance
        importances = self.model.feature_importances_
        
        # Combine with sample values
        feature_contributions = []
        for idx, (value, importance) in enumerate(zip(sample, importances)):
            feature_contributions.append({
                'name': self.feature_names_[idx],
                'value': float(value),
                'importance': float(importance),
                'contribution': float(abs(value) * importance)
            })
        
        # Sort by contribution
        feature_contributions.sort(key=lambda x: x['contribution'], reverse=True)
        
        return feature_contributions[:n]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate classifier on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Evaluating on {len(X_test)} test samples")
        
        # Get predictions
        y_pred, y_proba = self.predict(X_test)
        
        # Overall accuracy
        accuracy = np.mean(y_pred == y_test)
        
        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1-score: {report['macro avg']['f1-score']:.4f}")
        logger.info(f"Weighted F1-score: {report['weighted avg']['f1-score']:.4f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'classes': self.classes_.tolist()
        }
    
    def save(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'model': self.model,
            'classes': self.classes_,
            'feature_names': self.feature_names_,
            'class_mapping': self.class_mapping_,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load trained model from file.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.classes_ = model_data['classes']
        self.feature_names_ = model_data['feature_names']
        self.class_mapping_ = model_data['class_mapping']
        self.config = model_data.get('config', {})
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Classes: {self.classes_}")
        logger.info(f"Number of features: {len(self.feature_names_)}")
