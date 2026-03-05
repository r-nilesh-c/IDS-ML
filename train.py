"""
Main training pipeline for Hybrid Anomaly-Based IDS.

This script:
1. Loads configuration from YAML file
2. Initializes all components (preprocessing, autoencoder, isolation forest, fusion)
3. Executes training pipeline: load data → preprocess → train models → fit threshold
4. Saves trained models and normalization parameters
5. Logs training summary

Usage:
    python train.py --config config/default_config.yaml
"""

import argparse
import logging
import os
import sys
import yaml
import pickle
import random
from datetime import datetime
import numpy as np

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
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        logger.info(f"TensorFlow random seed set to {seed}")
    except ImportError:
        pass
    
    logger.info(f"Random seeds set to {seed} for reproducibility")


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


def save_models(autoencoder: AutoencoderDetector, 
               isolation_forest: IsolationForestDetector,
               fusion: FusionModule,
               save_dir: str):
    """
    Save trained models and fusion parameters.
    
    Args:
        autoencoder: Trained autoencoder detector
        isolation_forest: Trained isolation forest detector
        fusion: Fitted fusion module
        save_dir: Directory to save models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save autoencoder model (already saved during training via checkpoint)
    logger.info(f"Autoencoder model saved to {save_dir}/autoencoder_best.keras")
    
    # Save isolation forest model
    if_path = os.path.join(save_dir, 'isolation_forest.pkl')
    with open(if_path, 'wb') as f:
        pickle.dump(isolation_forest.model, f)
    logger.info(f"Isolation Forest model saved to {if_path}")
    
    # Save fusion parameters
    fusion_params = {
        'weight_autoencoder': fusion.weight_autoencoder,
        'weight_isolation': fusion.weight_isolation,
        'percentile': fusion.percentile,
        'recon_min': fusion.recon_min,
        'recon_max': fusion.recon_max,
        'iso_min': fusion.iso_min,
        'iso_max': fusion.iso_max,
        'threshold': fusion.threshold
    }
    
    fusion_path = os.path.join(save_dir, 'fusion_params.pkl')
    with open(fusion_path, 'wb') as f:
        pickle.dump(fusion_params, f)
    logger.info(f"Fusion parameters saved to {fusion_path}")


def main():
    """Main training pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Hybrid Anomaly-Based IDS')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("HYBRID ANOMALY-BASED IDS - TRAINING PIPELINE")
    print("=" * 80)
    print()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set random seeds for reproducibility
        random_state = config.get('preprocessing', {}).get('random_state', 42)
        set_random_seeds(random_state)
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        print("-" * 80)
        print("STEP 1: Data Preprocessing")
        print("-" * 80)
        
        # Initialize preprocessing pipeline
        preprocessing_config = config.get('preprocessing', {})
        preprocessing = PreprocessingPipeline(preprocessing_config)
        
        # Load real datasets
        logger.info("Preprocessing pipeline initialized")
        
        dataset_paths = config.get('dataset_paths', [])
        
        if not dataset_paths or not all(os.path.exists(p) for p in dataset_paths):
            # Fall back to synthetic data if datasets not available
            logger.warning("Dataset files not found, using synthetic data for demonstration")
            print("\n⚠ Using synthetic data for demonstration")
            print("  To train on real CIC-IDS datasets, provide paths in config file\n")
            
            n_samples = 1000
            n_features = 30
            
            np.random.seed(random_state)
            X_train_benign = np.random.normal(0.5, 0.1, size=(n_samples, n_features))
            X_train_benign = np.clip(X_train_benign, 0, 1)
            
            # Split for validation
            val_split = int(0.8 * n_samples)
            X_train = X_train_benign[:val_split]
            X_val_benign = X_train_benign[val_split:]
            
            n_features = X_train.shape[1]
        else:
            # Load and preprocess real datasets
            logger.info(f"Loading {len(dataset_paths)} dataset files...")
            print(f"\n[OK] Loading {len(dataset_paths)} CIC-IDS dataset files...")
            
            df = preprocessing.load_datasets(dataset_paths)
            logger.info(f"Loaded {len(df)} total samples")
            print(f"  Total samples loaded: {len(df)}")
            
            df_clean = preprocessing.clean_data(df)
            logger.info(f"After cleaning: {len(df_clean)} samples")
            print(f"  After cleaning: {len(df_clean)} samples")
            
            # Remove outliers if enabled
            if preprocessing_config.get('remove_outliers', True):
                outlier_method = preprocessing_config.get('outlier_method', 'iqr')
                outlier_threshold = preprocessing_config.get('outlier_threshold', 3.0)
                df_clean = preprocessing.remove_outliers(
                    df_clean, 
                    method=outlier_method, 
                    threshold=outlier_threshold
                )
                logger.info(f"After outlier removal: {len(df_clean)} samples")
                print(f"  After outlier removal: {len(df_clean)} samples")
            
            benign_df, attack_df = preprocessing.split_benign_attack(df_clean)
            logger.info(f"Benign samples: {len(benign_df)}, Attack samples: {len(attack_df)}")
            print(f"  Benign samples: {len(benign_df)}, Attack samples: {len(attack_df)}")
            
            # Feature selection if enabled
            if preprocessing_config.get('feature_selection', True):
                n_features = preprocessing_config.get('n_features', 30)
                selection_method = preprocessing_config.get('feature_selection_method', 'variance')
                benign_df, attack_df, selected_features = preprocessing.select_features(
                    benign_df, 
                    attack_df, 
                    n_features=n_features,
                    method=selection_method
                )
                logger.info(f"After feature selection: {len(benign_df.columns)-1} features")
                print(f"  After feature selection: {len(benign_df.columns)-1} features")
            
            data_splits = preprocessing.normalize_and_split(benign_df, attack_df)
            
            X_train = data_splits['X_train_benign']
            X_val_benign = data_splits['X_val_benign']
            n_features = X_train.shape[1]
            
            logger.info(f"Training data: {X_train.shape[0]} samples, {n_features} features")
            logger.info(f"Validation data: {X_val_benign.shape[0]} samples")
        
        logger.info(f"Training data: {X_train.shape[0]} samples, {n_features} features")
        logger.info(f"Validation data: {X_val_benign.shape[0]} samples")
        
        print(f"[OK] Data prepared: {X_train.shape[0]} training samples, {X_val_benign.shape[0]} validation samples")
        
        print("\n" + "-" * 80)
        print("STEP 2: Train Autoencoder")
        print("-" * 80)
        
        # Initialize and train autoencoder
        ae_config = config.get('autoencoder', {})
        ae_config['random_state'] = random_state
        ae_config['model_save_path'] = config.get('model_save_path', 'models/')
        
        autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
        autoencoder.build_model(use_dropout=True, dropout_rate=0.2)
        
        logger.info("Training autoencoder...")
        history = autoencoder.train(X_train, X_val_benign)
        
        print(f"[OK] Autoencoder trained in {len(history.history['loss'])} epochs")
        print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")
        
        print("\n" + "-" * 80)
        print("STEP 3: Train Isolation Forest")
        print("-" * 80)
        
        # Initialize and train isolation forest
        if_config = config.get('isolation_forest', {})
        isolation_forest = IsolationForestDetector(if_config)
        
        logger.info("Training Isolation Forest...")
        # Use all benign training data (train + validation) for Isolation Forest
        X_train_benign_full = np.vstack([X_train, X_val_benign])
        isolation_forest.train(X_train_benign_full)
        
        print(f"[OK] Isolation Forest trained on {X_train_benign_full.shape[0]} benign samples")
        
        print("\n" + "-" * 80)
        print("STEP 4: Fit Fusion Threshold")
        print("-" * 80)
        
        # Compute scores on validation set
        logger.info("Computing validation scores for threshold fitting...")
        recon_errors_val = autoencoder.compute_reconstruction_error(X_val_benign)
        iso_scores_val = isolation_forest.compute_anomaly_score(X_val_benign)
        
        # Initialize and fit fusion module
        fusion_config = config.get('fusion', {})
        fusion = FusionModule(fusion_config)
        
        logger.info("Fitting fusion threshold...")
        fusion.fit_threshold(recon_errors_val, iso_scores_val)
        
        print(f"[OK] Fusion threshold fitted: {fusion.threshold:.6f}")
        print(f"  Autoencoder weight: {fusion.weight_autoencoder}")
        print(f"  Isolation Forest weight: {fusion.weight_isolation}")
        print(f"  Percentile: {fusion.percentile}")
        
        print("\n" + "-" * 80)
        print("STEP 5: Save Models")
        print("-" * 80)
        
        # Save all models
        save_dir = config.get('model_save_path', 'models/')
        save_models(autoencoder, isolation_forest, fusion, save_dir)
        
        print(f"[OK] All models saved to {save_dir}")
        
        print("\n" + "-" * 80)
        print("TRAINING SUMMARY")
        print("-" * 80)
        
        print(f"\n[OK] Training completed successfully!")
        print(f"\nModels saved:")
        print(f"  - Autoencoder: {save_dir}/autoencoder_best.keras")
        print(f"  - Isolation Forest: {save_dir}/isolation_forest.pkl")
        print(f"  - Fusion parameters: {save_dir}/fusion_params.pkl")
        
        print(f"\nConfiguration:")
        print(f"  - Random seed: {random_state}")
        print(f"  - Training samples: {X_train.shape[0]}")
        print(f"  - Validation samples: {X_val_benign.shape[0]}")
        print(f"  - Features: {n_features}")
        
        print(f"\nNext steps:")
        print(f"  1. Run evaluation: python evaluate.py")
        print(f"  2. Run inference: python inference.py --input <data_file>")
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print()
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        print(f"\n❌ Training failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
