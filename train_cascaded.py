"""
Training script for Cascaded Hybrid IDS.

This script trains both stages:
Stage 1: Anomaly Detection (Autoencoder + Isolation Forest)
Stage 2: Supervised Classification (Random Forest)

Usage:
    python train_cascaded.py --config config/default_config.yaml
"""

import argparse
import logging
import os
import sys
import yaml
import pickle
import random
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.preprocessing import PreprocessingPipeline
from src.autoencoder import AutoencoderDetector
from src.isolation_forest import IsolationForestDetector
from src.fusion import FusionModule
from src.supervised_classifier import SupervisedClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_cascaded.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        logger.info(f"TensorFlow random seed set to {seed}")
    except ImportError:
        pass
    
    logger.info(f"Random seeds set to {seed}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    return config


def main():
    """Main training pipeline for cascaded system."""
    
    parser = argparse.ArgumentParser(description='Train Cascaded Hybrid IDS')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--optimize-stage2', action='store_true',
                       help='Perform hyperparameter optimization for Stage 2')
    args = parser.parse_args()
    
    print("=" * 80)
    print("CASCADED HYBRID IDS - TRAINING PIPELINE")
    print("=" * 80)
    print()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set random seeds
        random_state = config.get('preprocessing', {}).get('random_state', 42)
        set_random_seeds(random_state)
        
        # Create directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        print("-" * 80)
        print("STEP 1: Data Preprocessing")
        print("-" * 80)
        
        # Initialize preprocessing
        preprocessing_config = config.get('preprocessing', {})
        preprocessing = PreprocessingPipeline(preprocessing_config)
        
        # Load datasets
        dataset_paths = config.get('dataset_paths', [])
        
        if not dataset_paths or not all(os.path.exists(p) for p in dataset_paths):
            raise ValueError("Dataset files not found. Please check config file.")
        
        logger.info(f"Loading {len(dataset_paths)} dataset files...")
        print(f"\n[OK] Loading {len(dataset_paths)} CIC-IDS dataset files...")
        
        df = preprocessing.load_datasets(dataset_paths)
        logger.info(f"Loaded {len(df)} total samples")
        print(f"  Total samples loaded: {len(df)}")
        
        df_clean = preprocessing.clean_data(df)
        logger.info(f"After cleaning: {len(df_clean)} samples")
        print(f"  After cleaning: {len(df_clean)} samples")
        
        benign_df, attack_df = preprocessing.split_benign_attack(df_clean)
        logger.info(f"Benign samples: {len(benign_df)}, Attack samples: {len(attack_df)}")
        print(f"  Benign samples: {len(benign_df)}, Attack samples: {len(attack_df)}")
        
        # Get full dataset for Stage 2
        data_splits = preprocessing.normalize_and_split(benign_df, attack_df)
        
        # Stage 1 data (benign only)
        X_train_benign = data_splits['X_train_benign']
        X_val_benign = data_splits['X_val_benign']
        
        # Stage 2 data (full labeled dataset)
        X_train_full = data_splits['X_train']
        y_train_full = data_splits['y_train']
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        y_test_labels = data_splits['y_test_labels']  # String labels for evaluation
        
        n_features = X_train_benign.shape[1]
        feature_names = [col for col in benign_df.columns if col != 'Label']
        
        logger.info(f"Stage 1 training data: {X_train_benign.shape[0]} benign samples")
        logger.info(f"Stage 1 validation data: {X_val_benign.shape[0]} benign samples")
        logger.info(f"Stage 2 training data: {X_train_full.shape[0]} samples")
        logger.info(f"Test data: {X_test.shape[0]} samples")
        
        print(f"[OK] Data prepared:")
        print(f"  Stage 1: {X_train_benign.shape[0]} benign train, {X_val_benign.shape[0]} benign val")
        print(f"  Stage 2: {X_train_full.shape[0]} train samples (benign + attacks)")
        print(f"  Test: {X_test.shape[0]} samples")
        print(f"  Features: {n_features}")
        
        print("\n" + "-" * 80)
        print("STEP 2: Train Stage 1 - Anomaly Detection")
        print("-" * 80)
        
        # Train Autoencoder
        print("\n[Stage 1.1] Training Autoencoder...")
        ae_config = config.get('autoencoder', {})
        ae_config['random_state'] = random_state
        ae_config['model_save_path'] = config.get('model_save_path', 'models/')
        
        autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
        autoencoder.build_model(use_dropout=True, dropout_rate=0.2)
        
        history = autoencoder.train(X_train_benign, X_val_benign)
        
        print(f"[OK] Autoencoder trained in {len(history.history['loss'])} epochs")
        print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")
        
        # Train Isolation Forest
        print("\n[Stage 1.2] Training Isolation Forest...")
        if_config = config.get('isolation_forest', {})
        isolation_forest = IsolationForestDetector(if_config)
        
        X_train_benign_full = np.vstack([X_train_benign, X_val_benign])
        isolation_forest.train(X_train_benign_full)
        
        print(f"[OK] Isolation Forest trained on {X_train_benign_full.shape[0]} benign samples")
        
        # Fit Fusion Module
        print("\n[Stage 1.3] Fitting Fusion Module...")
        recon_errors_val = autoencoder.compute_reconstruction_error(X_val_benign)
        iso_scores_val = isolation_forest.compute_anomaly_score(X_val_benign)
        
        fusion_config = config.get('fusion', {})
        fusion = FusionModule(fusion_config)
        fusion.fit_threshold(recon_errors_val, iso_scores_val)
        
        print(f"[OK] Fusion threshold fitted: {fusion.threshold:.6f}")
        print(f"  Autoencoder weight: {fusion.weight_autoencoder}")
        print(f"  Isolation Forest weight: {fusion.weight_isolation}")
        print(f"  Percentile: {fusion.percentile}")
        
        print("\n" + "-" * 80)
        print("STEP 3: Train Stage 2 - Supervised Classifier")
        print("-" * 80)
        
        # Train Random Forest Classifier
        print("\n[Stage 2] Training Random Forest Classifier...")
        classifier_config = config.get('supervised_classifier', {})
        classifier_config['random_state'] = random_state
        
        classifier = SupervisedClassifier(classifier_config)
        
        train_metrics = classifier.train(
            X_train_full,
            y_train_full,
            feature_names=feature_names,
            optimize_hyperparameters=args.optimize_stage2
        )
        
        print(f"[OK] Classifier trained successfully")
        print(f"  Training accuracy: {train_metrics['train_accuracy']:.4f}")
        print(f"  Number of classes: {train_metrics['n_classes']}")
        print(f"  Number of features: {train_metrics['n_features']}")
        
        # Evaluate on test set
        print("\n[Stage 2] Evaluating classifier on test set...")
        eval_metrics = classifier.evaluate(X_test, y_test_labels)  # Use string labels
        
        print(f"[OK] Classifier evaluation:")
        print(f"  Test accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"  Macro F1-score: {eval_metrics['classification_report']['macro avg']['f1-score']:.4f}")
        print(f"  Weighted F1-score: {eval_metrics['classification_report']['weighted avg']['f1-score']:.4f}")
        
        # Show per-class metrics
        print(f"\n  Per-class metrics:")
        for cls in classifier.classes_:
            if cls in eval_metrics['classification_report']:
                metrics = eval_metrics['classification_report'][cls]
                print(f"    {cls}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Show top features
        print(f"\n  Top 10 most important features:")
        feature_importance = classifier.get_feature_importance(n=10)
        for i, (feature, importance) in enumerate(feature_importance.items(), 1):
            print(f"    {i}. {feature}: {importance:.4f}")
        
        print("\n" + "-" * 80)
        print("STEP 4: Save Models")
        print("-" * 80)
        
        save_dir = config.get('model_save_path', 'models/')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save Stage 1 models (already saved during training)
        print(f"\n[OK] Stage 1 models saved:")
        print(f"  - Autoencoder: {save_dir}/autoencoder_best.keras")
        
        if_path = os.path.join(save_dir, 'isolation_forest.pkl')
        with open(if_path, 'wb') as f:
            pickle.dump(isolation_forest.model, f)
        print(f"  - Isolation Forest: {if_path}")
        
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
        print(f"  - Fusion parameters: {fusion_path}")
        
        # Save Stage 2 model
        classifier_path = os.path.join(save_dir, 'supervised_classifier.pkl')
        classifier.save(classifier_path)
        print(f"\n[OK] Stage 2 model saved:")
        print(f"  - Supervised Classifier: {classifier_path}")
        
        print("\n" + "-" * 80)
        print("TRAINING SUMMARY")
        print("-" * 80)
        
        print(f"\n[OK] Cascaded Hybrid IDS training completed successfully!")
        
        print(f"\nStage 1 (Anomaly Detection):")
        print(f"  - Autoencoder: {len(history.history['loss'])} epochs, loss={history.history['val_loss'][-1]:.6f}")
        print(f"  - Isolation Forest: {X_train_benign_full.shape[0]} samples")
        print(f"  - Threshold: {fusion.threshold:.6f} (percentile={fusion.percentile})")
        
        print(f"\nStage 2 (Supervised Classification):")
        print(f"  - Random Forest: {train_metrics['n_classes']} classes")
        print(f"  - Training accuracy: {train_metrics['train_accuracy']:.4f}")
        print(f"  - Test accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"  - Macro F1-score: {eval_metrics['classification_report']['macro avg']['f1-score']:.4f}")
        
        print(f"\nDataset:")
        print(f"  - Training samples: {X_train_full.shape[0]}")
        print(f"  - Test samples: {X_test.shape[0]}")
        print(f"  - Features: {n_features}")
        
        print(f"\nNext steps:")
        print(f"  1. Run evaluation: python evaluate_cascaded.py")
        print(f"  2. Run inference: python inference_cascaded.py --input <data_file>")
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print()
        
        logger.info("Cascaded training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        print(f"\n[FAIL] Training failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
