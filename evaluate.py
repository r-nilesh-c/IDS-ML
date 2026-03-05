"""
Evaluation script for Hybrid Anomaly-Based IDS.

This script:
1. Loads trained models and normalization parameters
2. Loads test data (benign + attacks)
3. Runs inference on all test samples
4. Generates comprehensive evaluation report using HealthcareAlertSystem
5. Saves report as JSON and human-readable text
6. Assesses deployment readiness

Usage:
    python evaluate.py --test-data <test_file> [--config <config_file>]
    python evaluate.py --zero-day --train-data <2017_data> --test-data <2018_data>
"""

import argparse
import logging
import os
import sys
import yaml
import pickle
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.preprocessing import PreprocessingPipeline
from src.autoencoder import AutoencoderDetector
from src.isolation_forest import IsolationForestDetector
from src.fusion import FusionModule
from src.alert_system import HealthcareAlertSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    return config


def load_models(model_dir: str, config: dict):
    """
    Load trained models and fusion parameters.
    
    Args:
        model_dir: Directory containing saved models
        config: Configuration dictionary
        
    Returns:
        Tuple of (autoencoder, isolation_forest, fusion, model_input_dim, scaler, selected_features)
        
    Raises:
        FileNotFoundError: If model files are missing
        Exception: If models cannot be loaded
    """
    logger.info(f"Loading models from {model_dir}")
    
    # Load selected features metadata (NEW: for consistent feature selection)
    selected_features = None
    selected_features_path = os.path.join(model_dir, 'selected_features.pkl')
    if os.path.exists(selected_features_path):
        try:
            with open(selected_features_path, 'rb') as f:
                features_metadata = pickle.load(f)
                selected_features = features_metadata.get('selected_features', None)
                n_selected = features_metadata.get('n_features', 0)
            logger.info(f"Selected features loaded: {n_selected} features")
            if selected_features:
                logger.info(f"Features: {selected_features}")
        except Exception as e:
            logger.warning(f"Could not load selected features from {selected_features_path}: {str(e)}")
    
    # Load autoencoder
    ae_model_path = os.path.join(model_dir, 'autoencoder_best.keras')
    if not os.path.exists(ae_model_path):
        raise FileNotFoundError(f"Autoencoder model not found: {ae_model_path}")
    
    try:
        import tensorflow as tf
        ae_config = config.get('autoencoder', {})
        autoencoder = AutoencoderDetector(input_dim=1, config=ae_config)
        autoencoder.model = tf.keras.models.load_model(ae_model_path)
        loaded_ae_input_dim = int(autoencoder.model.input_shape[-1])
        autoencoder.input_dim = loaded_ae_input_dim
        logger.info(f"Autoencoder loaded from {ae_model_path}")
        logger.info(f"Autoencoder input dimension: {loaded_ae_input_dim}")
    except Exception as e:
        raise Exception(f"Failed to load autoencoder: {str(e)}")
    
    # Load isolation forest
    if_model_path = os.path.join(model_dir, 'isolation_forest.pkl')
    if not os.path.exists(if_model_path):
        raise FileNotFoundError(f"Isolation Forest model not found: {if_model_path}")
    
    try:
        with open(if_model_path, 'rb') as f:
            if_model = pickle.load(f)
        
        if_config = config.get('isolation_forest', {})
        isolation_forest = IsolationForestDetector(if_config)
        isolation_forest.model = if_model
        logger.info(f"Isolation Forest loaded from {if_model_path}")

        if_n_features = getattr(if_model, 'n_features_in_', None)
        if if_n_features is not None:
            logger.info(f"Isolation Forest input dimension: {if_n_features}")
            if int(if_n_features) != loaded_ae_input_dim:
                raise ValueError(
                    "Model dimension mismatch in model directory: "
                    f"autoencoder expects {loaded_ae_input_dim} features but "
                    f"isolation forest expects {if_n_features}. "
                    "Artifacts appear inconsistent/stale. Re-train models or use the correct --model-dir."
                )

    except Exception as e:
        raise Exception(f"Failed to load Isolation Forest: {str(e)}")
    
    # Load fusion parameters
    fusion_params_path = os.path.join(model_dir, 'fusion_params.pkl')
    if not os.path.exists(fusion_params_path):
        raise FileNotFoundError(f"Fusion parameters not found: {fusion_params_path}")
    
    try:
        with open(fusion_params_path, 'rb') as f:
            fusion_params = pickle.load(f)
        
        fusion_config = config.get('fusion', {})
        fusion = FusionModule(fusion_config)
        
        # Restore normalization parameters and threshold
        fusion.recon_min = fusion_params['recon_min']
        fusion.recon_max = fusion_params['recon_max']
        fusion.iso_min = fusion_params['iso_min']
        fusion.iso_max = fusion_params['iso_max']
        fusion.threshold = fusion_params['threshold']
        
        logger.info(f"Fusion parameters loaded from {fusion_params_path}")
        logger.info(f"Threshold: {fusion.threshold:.6f}")
    except Exception as e:
        raise Exception(f"Failed to load fusion parameters: {str(e)}")
    
    scaler = None
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            logger.warning(f"Could not load scaler from {scaler_path}: {str(e)}")

    return autoencoder, isolation_forest, fusion, loaded_ae_input_dim, scaler, selected_features
def load_test_data(test_path: str,
                  preprocessing: PreprocessingPipeline,
                  expected_features: int = None,
                  train_data_path: str = None,
                  config: dict = None,
                  scaler=None,
                  selected_features: list = None):
    """
    Load and preprocess test data.
    
    Args:
        test_path: Path to test data CSV file
        preprocessing: Preprocessing pipeline instance
        expected_features: Expected feature count from trained model
        train_data_path: Optional training CSV path for feature alignment
        config: Configuration dictionary
        scaler: Optional fitted scaler loaded from model artifacts
        selected_features: Optional list of feature names to use (from saved metadata)
        
    Returns:
        Tuple of (X_test, y_test, attack_labels)
        
    Raises:
        FileNotFoundError: If test file doesn't exist
        ValueError: If test data cannot be loaded
    """
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    logger.info(f"Loading test data from {test_path}")
    
    try:
        # Try loading with different encodings
        df = None
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(test_path, encoding=encoding, low_memory=False)
                logger.info(f"Successfully loaded {test_path} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Could not load {test_path} with any supported encoding")
        
        logger.info(f"Loaded {len(df)} samples from {test_path}")

        # Standardize label column name before cleaning
        label_col = None
        for col in df.columns:
            if str(col).strip().lower() == 'label':
                label_col = col
                break

        if label_col is None:
            raise ValueError("No label column found in test data")

        if label_col != 'Label':
            df = df.rename(columns={label_col: 'Label'})
            logger.info(f"Renamed label column '{label_col}' to 'Label'")
        
        # Clean data
        df_clean = preprocessing.clean_data(df)
        logger.info(f"After cleaning: {len(df_clean)} samples")
        
        # Extract labels
        if 'Label' not in df_clean.columns:
            raise ValueError("No label column found in test data")

        train_clean = None

        # Align features to model expectation when needed
        feature_cols = [col for col in df_clean.columns if col != 'Label']
        
        # PRIORITY: Use saved selected_features if available (NEW)
        if selected_features is not None and len(selected_features) > 0:
            logger.info(f"Using saved selected features ({len(selected_features)} features)")
            
            # Check if all selected features exist in test data
            missing_features = [f for f in selected_features if f not in feature_cols]
            if missing_features:
                raise ValueError(
                    f"Test data is missing selected features required by model. Missing: {missing_features[:10]}"
                )
            
            # Apply selection
            df_clean = df_clean[selected_features + ['Label']]
            feature_cols = selected_features
            logger.info(f"Applied {len(selected_features)} saved selected features")
        
        elif expected_features is not None and len(feature_cols) != expected_features:
            if not train_data_path:
                raise ValueError(
                    f"Feature mismatch: test data has {len(feature_cols)} features but model expects "
                    f"{expected_features}. Re-run with --train-data to reproduce feature selection "
                    "used during training."
                )

            logger.info(
                f"Feature alignment required: selecting {expected_features} features using training data "
                f"from {train_data_path}"
            )

            if not os.path.exists(train_data_path):
                raise FileNotFoundError(f"Training file not found: {train_data_path}")

            train_df = None
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    train_df = pd.read_csv(train_data_path, encoding=encoding, low_memory=False)
                    logger.info(f"Successfully loaded {train_data_path} with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if train_df is None:
                raise ValueError(f"Could not load {train_data_path} with any supported encoding")

            train_label_col = None
            for col in train_df.columns:
                if str(col).strip().lower() == 'label':
                    train_label_col = col
                    break

            if train_label_col is None:
                raise ValueError("No label column found in training data")

            if train_label_col != 'Label':
                train_df = train_df.rename(columns={train_label_col: 'Label'})
                logger.info(f"Renamed training label column '{train_label_col}' to 'Label'")

            train_clean = preprocessing.clean_data(train_df)

            preprocessing_config = config.get('preprocessing', {}) if config else {}
            selection_method = preprocessing_config.get('feature_selection_method', 'variance')

            feature_source_cols = [col for col in train_clean.columns if col != 'Label']
            if expected_features >= len(feature_source_cols):
                selected_features = feature_source_cols
            elif selection_method == 'statistical':
                try:
                    train_benign, train_attack = preprocessing.split_benign_attack(train_clean)
                    _, _, selected_features = preprocessing.select_features(
                        train_benign,
                        train_attack,
                        n_features=expected_features,
                        method=selection_method
                    )
                except Exception:
                    benign_only = train_clean[train_clean['Label'].str.upper() == 'BENIGN']
                    if benign_only.empty:
                        raise ValueError("No benign samples found in training data for feature alignment")
                    variances = benign_only[feature_source_cols].var()
                    selected_features = variances.nlargest(expected_features).index.tolist()
                    logger.warning(
                        "Statistical feature selection unavailable for training data; "
                        "falling back to variance-based selection"
                    )
            else:
                benign_only = train_clean[train_clean['Label'].str.upper() == 'BENIGN']
                if benign_only.empty:
                    raise ValueError("No benign samples found in training data for feature alignment")
                variances = benign_only[feature_source_cols].var()
                selected_features = variances.nlargest(expected_features).index.tolist()

            missing_features = [col for col in selected_features if col not in df_clean.columns]
            if missing_features:
                raise ValueError(
                    "Test data is missing selected features required by model: "
                    f"{missing_features[:10]}"
                )

            df_clean = df_clean[selected_features + ['Label']]
            feature_cols = selected_features
            logger.info(f"Applied feature alignment using {len(selected_features)} selected features")

        # Apply scaling consistently with training pipeline
        X_features = df_clean[feature_cols].values

        if scaler is not None:
            X_test = scaler.transform(X_features)
            logger.info("Applied scaler from model artifacts")
        elif train_data_path:
            logger.info("No saved scaler found; deriving scaler from --train-data benign samples")

            if train_clean is None:
                train_df = None
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        train_df = pd.read_csv(train_data_path, encoding=encoding, low_memory=False)
                        logger.info(f"Successfully loaded {train_data_path} with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue

                if train_df is None:
                    raise ValueError(f"Could not load {train_data_path} with any supported encoding")

                train_label_col = None
                for col in train_df.columns:
                    if str(col).strip().lower() == 'label':
                        train_label_col = col
                        break

                if train_label_col is None:
                    raise ValueError("No label column found in training data")

                if train_label_col != 'Label':
                    train_df = train_df.rename(columns={train_label_col: 'Label'})
                    logger.info(f"Renamed training label column '{train_label_col}' to 'Label'")

                train_clean = preprocessing.clean_data(train_df)

            missing_scale_features = [col for col in feature_cols if col not in train_clean.columns]
            if missing_scale_features:
                raise ValueError(
                    "Training data is missing features required for scaling: "
                    f"{missing_scale_features[:10]}"
                )

            benign_train = train_clean[train_clean['Label'].str.upper() == 'BENIGN']
            if benign_train.empty:
                raise ValueError("No benign samples found in training data for scaling")

            preprocessing_config = config.get('preprocessing', {}) if config else {}
            use_robust_scaler = preprocessing_config.get('use_robust_scaler', True)
            derived_scaler = RobustScaler() if use_robust_scaler else StandardScaler()
            derived_scaler.fit(benign_train[feature_cols].values)

            X_test = derived_scaler.transform(X_features)
            logger.info(
                f"Applied derived {'RobustScaler' if use_robust_scaler else 'StandardScaler'} "
                "fitted on benign training samples"
            )
        else:
            X_test = X_features
            logger.warning(
                "No scaler applied. For model-consistent evaluation, provide models/scaler.pkl "
                "or run with --train-data."
            )
        
        # Store attack labels for detailed reporting
        attack_labels = df_clean['Label'].values
        
        # Convert to binary labels (0=benign, 1=attack)
        y_test = np.where(df_clean['Label'].str.upper() == 'BENIGN', 0, 1)
        
        logger.info(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        logger.info(f"Benign samples: {np.sum(y_test == 0)}")
        logger.info(f"Attack samples: {np.sum(y_test == 1)}")
        
        return X_test, y_test, attack_labels
        
    except Exception as e:
        raise ValueError(f"Failed to load test data: {str(e)}")


def run_evaluation(X_test: np.ndarray, y_test: np.ndarray, attack_labels: np.ndarray,
                  autoencoder: AutoencoderDetector, isolation_forest: IsolationForestDetector,
                  fusion: FusionModule, alert_system: HealthcareAlertSystem):
    """
    Run complete evaluation pipeline.
    
    Args:
        X_test: Test feature matrix
        y_test: Test labels (0=benign, 1=attack)
        attack_labels: Attack category names
        autoencoder: Trained autoencoder detector
        isolation_forest: Trained isolation forest detector
        fusion: Fitted fusion module
        alert_system: Healthcare alert system for reporting
        
    Returns:
        Dictionary containing evaluation results and metrics
    """
    logger.info(f"Running evaluation on {X_test.shape[0]} test samples")
    
    # Compute reconstruction errors from autoencoder
    logger.info("Computing autoencoder reconstruction errors...")
    recon_errors = autoencoder.compute_reconstruction_error(X_test)
    
    # Compute anomaly scores from isolation forest
    logger.info("Computing isolation forest anomaly scores...")
    iso_scores = isolation_forest.compute_anomaly_score(X_test)
    
    # Normalize scores
    logger.info("Normalizing scores...")
    recon_errors_norm, iso_scores_norm = fusion.normalize_scores(recon_errors, iso_scores)
    
    # Compute combined scores
    logger.info("Computing combined anomaly scores...")
    combined_scores = fusion.compute_combined_score(recon_errors_norm, iso_scores_norm)
    
    # Apply threshold for classification
    logger.info("Applying threshold for classification...")
    predictions = fusion.classify(combined_scores)
    
    logger.info(f"Predictions: {np.sum(predictions)} anomalies detected")
    
    # Log anomalies
    logger.info("Logging detected anomalies...")
    timestamp = datetime.now().isoformat()
    for i in range(len(predictions)):
        if predictions[i] == 1:
            # Create flow features dict (simplified for logging)
            flow_features = {
                'sample_index': int(i),
                'true_label': str(attack_labels[i]),
                'reconstruction_error': float(recon_errors[i]),
                'isolation_score': float(iso_scores[i])
            }
            alert_system.log_anomaly(timestamp, flow_features, combined_scores[i], predictions[i])
    
    logger.info(f"Logged {np.sum(predictions)} anomalies")
    
    # Generate evaluation report
    logger.info("Generating evaluation report...")
    metrics = alert_system.generate_evaluation_report(
        y_test, predictions, combined_scores, attack_labels
    )
    
    # Assess deployment readiness
    logger.info("Assessing deployment readiness...")
    readiness = alert_system.assess_deployment_readiness(metrics)
    metrics['deployment_readiness'] = readiness
    
    return {
        'predictions': predictions,
        'anomaly_scores': combined_scores,
        'metrics': metrics
    }


def save_evaluation_report(results: dict, output_path: str):
    """
    Save evaluation report as JSON and human-readable text.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Base path for output files (without extension)
    """
    logger.info(f"Saving evaluation report to {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save JSON report
    json_path = output_path + '.json'
    
    # Prepare JSON-serializable metrics
    json_metrics = {}
    for key, value in results['metrics'].items():
        if key in ['roc_curve_path', 'pr_curve_path', 'confusion_matrix_path', 'deployment_readiness']:
            json_metrics[key] = value
        elif key == 'per_class_metrics':
            json_metrics[key] = value
        elif key == 'confusion_matrix':
            json_metrics[key] = value
        elif isinstance(value, (int, float, str, bool, type(None))):
            json_metrics[key] = value
        elif isinstance(value, np.ndarray):
            json_metrics[key] = value.tolist()
        else:
            json_metrics[key] = str(value)
    
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    logger.info(f"JSON report saved to {json_path}")
    
    # Save human-readable text report
    text_path = output_path + '.txt'
    
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HYBRID ANOMALY-BASED IDS - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n\n")
        
        metrics = results['metrics']
        f.write(f"Accuracy:              {metrics['accuracy']:.4f}\n")
        f.write(f"Precision:             {metrics['precision']:.4f}\n")
        f.write(f"Recall:                {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:              {metrics['f1_score']:.4f}\n")
        f.write(f"Macro F1-Score:        {metrics['macro_f1_score']:.4f}\n")
        f.write(f"False Positive Rate:   {metrics['false_positive_rate']:.4f}\n")
        if metrics.get('roc_auc') is not None:
            f.write(f"ROC-AUC:               {metrics['roc_auc']:.4f}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 80 + "\n\n")
        
        cm = metrics['confusion_matrix']
        f.write(f"True Negatives:   {cm['true_negative']}\n")
        f.write(f"False Positives:  {cm['false_positive']}\n")
        f.write(f"False Negatives:  {cm['false_negative']}\n")
        f.write(f"True Positives:   {cm['true_positive']}\n")
        
        # Per-class metrics if available
        if 'per_class_metrics' in metrics:
            f.write("\n" + "-" * 80 + "\n")
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 80 + "\n\n")
            
            for attack_type, class_metrics in metrics['per_class_metrics'].items():
                f.write(f"{attack_type}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {class_metrics['f1_score']:.4f}\n")
                if 'false_positive_rate' in class_metrics:
                    f.write(f"  FPR:       {class_metrics['false_positive_rate']:.4f}\n")
                f.write(f"  Support:   {class_metrics['support']}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("VISUALIZATIONS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"ROC Curve:              {metrics.get('roc_curve_path', 'N/A')}\n")
        f.write(f"Precision-Recall Curve: {metrics.get('pr_curve_path', 'N/A')}\n")
        f.write(f"Confusion Matrix:       {metrics.get('confusion_matrix_path', 'N/A')}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("DEPLOYMENT READINESS ASSESSMENT\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(metrics['deployment_readiness'])
        
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Text report saved to {text_path}")


def main():
    """Main evaluation pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Hybrid Anomaly-Based IDS')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data CSV file')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model-dir', type=str, default='models/',
                       help='Directory containing trained models')
    parser.add_argument('--output', type=str, default='reports/evaluation_report',
                       help='Base path for output files (without extension)')
    parser.add_argument('--zero-day', action='store_true',
                       help='Enable zero-day evaluation mode (cross-dataset)')
    parser.add_argument('--train-data', type=str,
                       help='Path to training data for zero-day mode (optional)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("HYBRID ANOMALY-BASED IDS - EVALUATION")
    print("=" * 80)
    print()
    
    if args.zero_day:
        print("⚠ Zero-day evaluation mode enabled")
        print("  Cross-dataset evaluation for novel attack detection\n")
    
    try:
        # Create logs directory if needed
        os.makedirs('logs', exist_ok=True)
        
        # Load configuration
        config = load_config(args.config)
        
        print("-" * 80)
        print("STEP 1: Load Test Data")
        print("-" * 80)
        
        # Initialize preprocessing
        preprocessing_config = config.get('preprocessing', {})
        preprocessing = PreprocessingPipeline(preprocessing_config)
        
        # Load models first to determine expected input dimensions
        autoencoder, isolation_forest, fusion, model_input_dim, scaler, selected_features = load_models(
            args.model_dir, config
        )

        # Load test data (with optional feature alignment)
        X_test, y_test, attack_labels = load_test_data(
            args.test_data,
            preprocessing,
            expected_features=model_input_dim,
            train_data_path=args.train_data,
            config=config,
            scaler=scaler,
            selected_features=selected_features
        )
        
        print(f"✓ Loaded {X_test.shape[0]} test samples")
        print(f"  - Benign samples: {np.sum(y_test == 0)}")
        print(f"  - Attack samples: {np.sum(y_test == 1)}")
        print(f"  - Features: {X_test.shape[1]}")
        
        print("\n" + "-" * 80)
        print("STEP 2: Load Models")
        print("-" * 80)
        
        print(f"✓ Models loaded from {args.model_dir}")
        print(f"  - Threshold: {fusion.threshold:.6f}")
        
        print("\n" + "-" * 80)
        print("STEP 3: Initialize Alert System")
        print("-" * 80)
        
        # Initialize alert system
        alert_config = {
            'log_path': config.get('log_path', 'logs/anomalies.jsonl'),
            'report_path': config.get('report_path', 'reports/')
        }
        alert_system = HealthcareAlertSystem(alert_config)
        
        print(f"✓ Alert system initialized")
        print(f"  - Log path: {alert_config['log_path']}")
        print(f"  - Report path: {alert_config['report_path']}")
        
        print("\n" + "-" * 80)
        print("STEP 4: Run Evaluation")
        print("-" * 80)
        
        # Run evaluation
        results = run_evaluation(
            X_test, y_test, attack_labels,
            autoencoder, isolation_forest, fusion, alert_system
        )
        
        print(f"✓ Evaluation completed")
        print(f"  - Anomalies detected: {np.sum(results['predictions'])}")
        print(f"  - Detection rate: {np.mean(results['predictions']):.2%}")
        
        print("\n" + "-" * 80)
        print("STEP 5: Save Report")
        print("-" * 80)
        
        # Save evaluation report
        save_evaluation_report(results, args.output)
        
        print(f"✓ Reports saved:")
        print(f"  - JSON: {args.output}.json")
        print(f"  - Text: {args.output}.txt")
        
        print("\n" + "-" * 80)
        print("EVALUATION SUMMARY")
        print("-" * 80)
        
        metrics = results['metrics']
        print(f"\nOverall Performance:")
        print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall:    {metrics['recall']:.4f}")
        print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  - FPR:       {metrics['false_positive_rate']:.4f}")
        if metrics.get('roc_auc') is not None:
            print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\nDeployment Readiness:")
        readiness_lines = metrics['deployment_readiness'].split('\n')
        print(f"  {readiness_lines[0]}")
        
        # Check healthcare criteria
        healthcare_config = config.get('healthcare', {})
        max_fpr = healthcare_config.get('max_fpr', 0.05)
        min_recall = healthcare_config.get('min_recall', 0.90)
        
        if metrics['false_positive_rate'] < max_fpr:
            print(f"  ✓ FPR meets healthcare requirements (<{max_fpr:.2%})")
        else:
            print(f"  ✗ FPR exceeds healthcare requirements (>{max_fpr:.2%})")
        
        if metrics['recall'] > min_recall:
            print(f"  ✓ Recall meets healthcare requirements (>{min_recall:.2%})")
        else:
            print(f"  ✗ Recall below healthcare requirements (<{min_recall:.2%})")
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print()
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        print(f"\n❌ Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
