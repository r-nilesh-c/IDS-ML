"""
Batch inference pipeline for Hybrid Anomaly-Based IDS.

This script:
1. Loads trained models and normalization parameters
2. Processes input data through complete pipeline
3. Generates predictions and anomaly scores
4. Measures and logs inference latency
5. Outputs results to file

Usage:
    python inference.py --input <data_file> --output <results_file> [--config <config_file>]
"""

import argparse
import logging
import os
import sys
import yaml
import pickle
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.preprocessing import PreprocessingPipeline
from src.autoencoder import AutoencoderDetector
from src.isolation_forest import IsolationForestDetector
from src.fusion import FusionModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/inference.log'),
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


def load_models(model_dir: str, input_dim: int, config: dict):
    """
    Load trained models and fusion parameters.
    
    Args:
        model_dir: Directory containing saved models
        input_dim: Number of input features
        config: Configuration dictionary
        
    Returns:
        Tuple of (autoencoder, isolation_forest, fusion)
        
    Raises:
        FileNotFoundError: If model files are missing
        Exception: If models cannot be loaded
    """
    logger.info(f"Loading models from {model_dir}")
    
    # Load autoencoder
    ae_model_path = os.path.join(model_dir, 'autoencoder_best.keras')
    if not os.path.exists(ae_model_path):
        raise FileNotFoundError(f"Autoencoder model not found: {ae_model_path}")
    
    try:
        import tensorflow as tf
        ae_config = config.get('autoencoder', {})
        autoencoder = AutoencoderDetector(input_dim=input_dim, config=ae_config)
        autoencoder.model = tf.keras.models.load_model(ae_model_path)
        logger.info(f"Autoencoder loaded from {ae_model_path}")
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
    
    return autoencoder, isolation_forest, fusion


def load_input_data(input_path: str) -> pd.DataFrame:
    """
    Load input data from CSV file.
    
    Args:
        input_path: Path to input CSV file
        
    Returns:
        DataFrame with input samples
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input file cannot be loaded
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading input data from {input_path}")
    
    try:
        # Try loading with different encodings
        df = None
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(input_path, encoding=encoding, low_memory=False)
                logger.info(f"Successfully loaded {input_path} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Could not load {input_path} with any supported encoding")
        
        logger.info(f"Loaded {len(df)} samples from {input_path}")
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to load input data: {str(e)}")


def preprocess_input(df: pd.DataFrame, preprocessing: PreprocessingPipeline) -> np.ndarray:
    """
    Preprocess input data for inference.
    
    Args:
        df: Input DataFrame
        preprocessing: Preprocessing pipeline instance
        
    Returns:
        Preprocessed feature matrix
        
    Raises:
        ValueError: If preprocessing fails
    """
    logger.info("Preprocessing input data")
    
    try:
        # Clean data (remove duplicates, NaN, inf, non-numeric features)
        df_clean = preprocessing.clean_data(df)
        
        # Extract features (drop label column if present)
        label_cols = ['Label', ' Label', 'label', ' label']
        feature_cols = [col for col in df_clean.columns if col not in label_cols]
        
        X = df_clean[feature_cols].values
        
        logger.info(f"Preprocessed {X.shape[0]} samples with {X.shape[1]} features")
        return X
        
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {str(e)}")


def run_inference(X: np.ndarray, autoencoder: AutoencoderDetector,
                 isolation_forest: IsolationForestDetector,
                 fusion: FusionModule, batch_size: int = 256):
    """
    Run inference on input data.
    
    Args:
        X: Input feature matrix
        autoencoder: Trained autoencoder detector
        isolation_forest: Trained isolation forest detector
        fusion: Fitted fusion module
        batch_size: Batch size for processing (default: 256)
        
    Returns:
        Dictionary containing:
            - predictions: Binary predictions (0/1)
            - anomaly_scores: Combined anomaly scores
            - reconstruction_errors: Autoencoder reconstruction errors
            - isolation_scores: Isolation Forest anomaly scores
            - inference_time_ms: Per-sample inference time in milliseconds
    """
    logger.info(f"Running inference on {X.shape[0]} samples")
    
    start_time = time.time()
    
    # Compute reconstruction errors from autoencoder
    logger.info("Computing autoencoder reconstruction errors...")
    recon_errors = autoencoder.compute_reconstruction_error(X)
    
    # Compute anomaly scores from isolation forest
    logger.info("Computing isolation forest anomaly scores...")
    iso_scores = isolation_forest.compute_anomaly_score(X)
    
    # Normalize scores
    logger.info("Normalizing scores...")
    recon_errors_norm, iso_scores_norm = fusion.normalize_scores(recon_errors, iso_scores)
    
    # Compute combined scores
    logger.info("Computing combined anomaly scores...")
    combined_scores = fusion.compute_combined_score(recon_errors_norm, iso_scores_norm)
    
    # Apply threshold for classification
    logger.info("Applying threshold for classification...")
    predictions = fusion.classify(combined_scores)
    
    end_time = time.time()
    
    # Calculate per-sample inference time
    total_time_ms = (end_time - start_time) * 1000
    per_sample_time_ms = total_time_ms / X.shape[0]
    
    logger.info(f"Inference completed in {total_time_ms:.2f}ms")
    logger.info(f"Per-sample latency: {per_sample_time_ms:.4f}ms")
    logger.info(f"Detected {np.sum(predictions)} anomalies out of {len(predictions)} samples")
    
    return {
        'predictions': predictions,
        'anomaly_scores': combined_scores,
        'reconstruction_errors': recon_errors,
        'isolation_scores': iso_scores,
        'inference_time_ms': per_sample_time_ms
    }


def save_results(results: dict, output_path: str, input_df: pd.DataFrame = None):
    """
    Save inference results to file.
    
    Args:
        results: Dictionary containing inference results
        output_path: Path to output file
        input_df: Optional input DataFrame to include features in output
    """
    logger.info(f"Saving results to {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Prepare output data
    output_data = {
        'prediction': results['predictions'].tolist(),
        'anomaly_score': results['anomaly_scores'].tolist(),
        'reconstruction_error': results['reconstruction_errors'].tolist(),
        'isolation_score': results['isolation_scores'].tolist()
    }
    
    # Create DataFrame
    output_df = pd.DataFrame(output_data)
    
    # Add input features if provided
    if input_df is not None:
        # Get feature columns (exclude label if present)
        label_cols = ['Label', ' Label', 'label', ' label']
        feature_cols = [col for col in input_df.columns if col not in label_cols]
        
        # Add features to output (only for samples that passed preprocessing)
        if len(output_df) == len(input_df):
            for col in feature_cols:
                if col in input_df.columns:
                    output_df[col] = input_df[col].values
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Also save summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(results['predictions']),
        'anomalies_detected': int(np.sum(results['predictions'])),
        'anomaly_rate': float(np.mean(results['predictions'])),
        'avg_anomaly_score': float(np.mean(results['anomaly_scores'])),
        'max_anomaly_score': float(np.max(results['anomaly_scores'])),
        'min_anomaly_score': float(np.min(results['anomaly_scores'])),
        'per_sample_latency_ms': results['inference_time_ms']
    }
    
    summary_path = output_path.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_path}")
    
    return summary


def main():
    """Main inference pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run batch inference with Hybrid Anomaly-Based IDS')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='reports/inference_results.csv',
                       help='Path to output results file')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model-dir', type=str, default='models/',
                       help='Directory containing trained models')
    args = parser.parse_args()
    
    print("=" * 80)
    print("HYBRID ANOMALY-BASED IDS - BATCH INFERENCE")
    print("=" * 80)
    print()
    
    try:
        # Create logs directory if needed
        os.makedirs('logs', exist_ok=True)
        
        # Load configuration
        config = load_config(args.config)
        
        print("-" * 80)
        print("STEP 1: Load Input Data")
        print("-" * 80)
        
        # Load input data
        input_df = load_input_data(args.input)
        print(f"✓ Loaded {len(input_df)} samples from {args.input}")
        
        print("\n" + "-" * 80)
        print("STEP 2: Preprocess Data")
        print("-" * 80)
        
        # Initialize preprocessing
        preprocessing_config = config.get('preprocessing', {})
        preprocessing = PreprocessingPipeline(preprocessing_config)
        
        # Preprocess input
        X = preprocess_input(input_df, preprocessing)
        print(f"✓ Preprocessed {X.shape[0]} samples with {X.shape[1]} features")
        
        print("\n" + "-" * 80)
        print("STEP 3: Load Models")
        print("-" * 80)
        
        # Load trained models
        autoencoder, isolation_forest, fusion = load_models(
            args.model_dir, X.shape[1], config
        )
        print(f"✓ Models loaded from {args.model_dir}")
        print(f"  - Autoencoder: {args.model_dir}/autoencoder_best.keras")
        print(f"  - Isolation Forest: {args.model_dir}/isolation_forest.pkl")
        print(f"  - Fusion parameters: {args.model_dir}/fusion_params.pkl")
        print(f"  - Threshold: {fusion.threshold:.6f}")
        
        print("\n" + "-" * 80)
        print("STEP 4: Run Inference")
        print("-" * 80)
        
        # Run inference
        batch_size = config.get('autoencoder', {}).get('batch_size', 256)
        results = run_inference(X, autoencoder, isolation_forest, fusion, batch_size)
        
        print(f"✓ Inference completed")
        print(f"  - Total samples: {X.shape[0]}")
        print(f"  - Anomalies detected: {np.sum(results['predictions'])}")
        print(f"  - Anomaly rate: {np.mean(results['predictions']):.2%}")
        print(f"  - Per-sample latency: {results['inference_time_ms']:.4f}ms")
        
        print("\n" + "-" * 80)
        print("STEP 5: Save Results")
        print("-" * 80)
        
        # Save results
        summary = save_results(results, args.output, input_df)
        
        print(f"✓ Results saved to {args.output}")
        print(f"✓ Summary saved to {args.output.replace('.csv', '_summary.json')}")
        
        print("\n" + "-" * 80)
        print("INFERENCE SUMMARY")
        print("-" * 80)
        
        print(f"\nInput: {args.input}")
        print(f"Output: {args.output}")
        print(f"\nResults:")
        print(f"  - Total samples: {summary['total_samples']}")
        print(f"  - Anomalies detected: {summary['anomalies_detected']}")
        print(f"  - Anomaly rate: {summary['anomaly_rate']:.2%}")
        print(f"  - Avg anomaly score: {summary['avg_anomaly_score']:.6f}")
        print(f"  - Max anomaly score: {summary['max_anomaly_score']:.6f}")
        print(f"  - Per-sample latency: {summary['per_sample_latency_ms']:.4f}ms")
        
        # Check if latency meets healthcare requirements (<100ms)
        if summary['per_sample_latency_ms'] < 100:
            print(f"\n✓ Latency meets healthcare requirements (<100ms)")
        else:
            print(f"\n⚠ Latency exceeds healthcare requirements (>100ms)")
        
        print("\n" + "=" * 80)
        print("INFERENCE COMPLETE")
        print("=" * 80)
        print()
        
        logger.info("Batch inference completed successfully")
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        print(f"\n❌ Inference failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
