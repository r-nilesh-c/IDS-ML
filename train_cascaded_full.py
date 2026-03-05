#!/usr/bin/env python3
"""
Cascaded Hybrid IDS Training - Full Dataset Version

This script trains the cascaded hybrid IDS using BOTH CIC-IDS2017 and CIC-IDS2018 datasets
to improve attack detection coverage, especially for attack types that were missed in the
initial evaluation (Bot, Infiltration, Web attacks, etc.).
"""

import os
import sys
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
import random
import tensorflow as tf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import PreprocessingPipeline
from autoencoder import AutoencoderDetector
from isolation_forest import IsolationForestDetector
from fusion import FusionModule
from supervised_classifier import SupervisedClassifier
from utils import load_config, set_random_seeds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline for cascaded system using full dataset."""
    
    parser = argparse.ArgumentParser(description='Train Cascaded Hybrid IDS on Full Dataset')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--optimize-stage2', action='store_true',
                       help='Perform hyperparameter optimization for Stage 2')
    args = parser.parse_args()
    
    print("=" * 80)
    print("CASCADED HYBRID IDS - FULL DATASET TRAINING")
    print("=" * 80)
    print("Training on BOTH CIC-IDS2017 and CIC-IDS2018 datasets")
    print("Expected improvements:")
    print("  - Better Bot/Botnet detection")
    print("  - Better Infiltration detection") 
    print("  - Better Web attack detection")
    print("  - Better Brute force detection")
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
        print("STEP 1: Data Preprocessing - Full Dataset")
        print("-" * 80)
        
        # Initialize preprocessing
        preprocessing_config = config.get('preprocessing', {})
        preprocessing = PreprocessingPipeline(preprocessing_config)
        
        # Define full dataset paths (2017 + 2018)
        dataset_paths_2017 = [
            'dataset/cic-ids2017/Monday-WorkingHours.pcap_ISCX.csv',
            'dataset/cic-ids2017/Tuesday-WorkingHours.pcap_ISCX.csv',
            'dataset/cic-ids2017/Wednesday-workingHours.pcap_ISCX.csv',
            'dataset/cic-ids2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'dataset/cic-ids2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'dataset/cic-ids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'dataset/cic-ids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'dataset/cic-ids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
        ]
        
        dataset_paths_2018 = [
            'dataset/cic-ids2018/Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet',
            'dataset/cic-ids2018/Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet',
            'dataset/cic-ids2018/DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet',
            'dataset/cic-ids2018/DDoS2-Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet',
            'dataset/cic-ids2018/DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet',
            'dataset/cic-ids2018/DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter.parquet',
            'dataset/cic-ids2018/Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet',
            'dataset/cic-ids2018/Infil2-Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet',
            'dataset/cic-ids2018/Web1-Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet',
            'dataset/cic-ids2018/Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter.parquet'
        ]
        
        # Check which files exist
        available_2017 = [p for p in dataset_paths_2017 if os.path.exists(p)]
        available_2018 = [p for p in dataset_paths_2018 if os.path.exists(p)]
        
        print(f"Available 2017 files: {len(available_2017)}/{len(dataset_paths_2017)}")
        print(f"Available 2018 files: {len(available_2018)}/{len(dataset_paths_2018)}")
        
        if not available_2017 and not available_2018:
            raise ValueError("No dataset files found!")
        
        # Load datasets separately to handle potential column differences
        dataframes = []
        
        if available_2017:
            print(f"\n[Loading] CIC-IDS2017 dataset ({len(available_2017)} files)...")
            df_2017 = preprocessing.load_datasets(available_2017)
            print(f"  2017 dataset: {len(df_2017)} samples")
            dataframes.append(df_2017)
        
        if available_2018:
            print(f"\n[Loading] CIC-IDS2018 dataset ({len(available_2018)} files)...")
            df_2018 = preprocessing.load_datasets(available_2018)
            print(f"  2018 dataset: {len(df_2018)} samples")
            dataframes.append(df_2018)
        
        # Combine datasets if we have both
        if len(dataframes) == 2:
            print(f"\n[Combining] Merging 2017 and 2018 datasets...")
            
            # Check column compatibility
            cols_2017 = set(df_2017.columns)
            cols_2018 = set(df_2018.columns)
            common_cols = cols_2017.intersection(cols_2018)
            
            print(f"  2017 columns: {len(cols_2017)}")
            print(f"  2018 columns: {len(cols_2018)}")
            print(f"  Common columns: {len(common_cols)}")
            
            if len(common_cols) < 50:  # Arbitrary threshold
                print(f"  WARNING: Limited column overlap, using intersection")
                # Use only common columns
                df_2017 = df_2017[list(common_cols)]
                df_2018 = df_2018[list(common_cols)]
            
            # Combine datasets
            df = pd.concat([df_2017, df_2018], ignore_index=True)
            print(f"  Combined dataset: {len(df)} samples")
        else:
            df = dataframes[0]
            print(f"  Using single dataset: {len(df)} samples")
        
        logger.info(f"Loaded {len(df)} total samples")
        
        # Clean and preprocess
        df_clean = preprocessing.clean_data(df)
        logger.info(f"After cleaning: {len(df_clean)} samples")
        print(f"  After cleaning: {len(df_clean)} samples")
        
        benign_df, attack_df = preprocessing.split_benign_attack(df_clean)
        logger.info(f"Benign samples: {len(benign_df)}, Attack samples: {len(attack_df)}")
        print(f"  Benign samples: {len(benign_df)}, Attack samples: {len(attack_df)}")
        
        # Show attack type distribution
        if len(attack_df) > 0:
            attack_counts = attack_df['Label'].value_counts()
            print(f"\n  Attack type distribution:")
            for attack_type, count in attack_counts.items():
                print(f"    {attack_type}: {count:,} samples")
        
        # Select top features for consistency across training and inference
        print("\n[STEP 1.5] Feature Selection")
        print("-" * 80)
        benign_df_selected, attack_df_selected, selected_features = preprocessing.select_features(
            benign_df, attack_df, 
            n_features=12,  # Fixed to 12 features for consistent model architecture
            method='variance'
        )
        print(f"[OK] Selected {len(selected_features)} features:")
        print(f"     {selected_features}")
        
        # Get full dataset for Stage 2 using SELECTED features
        data_splits = preprocessing.normalize_and_split(benign_df_selected, attack_df_selected)
        
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
        feature_names = selected_features  # Use selected feature names for consistency
        
        logger.info(f"Stage 1 training data: {X_train_benign.shape[0]} benign samples")
        logger.info(f"Stage 1 validation data: {X_val_benign.shape[0]} benign samples")
        logger.info(f"Stage 2 training data: {X_train_full.shape[0]} samples")
        logger.info(f"Test data: {X_test.shape[0]} samples")
        
        print(f"\n[OK] Data prepared:")
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
        
        # CRITICAL: Save selected feature names for consistent inference
        # This ensures live monitoring and evaluation use the SAME 12 features
        selected_features_path = os.path.join(save_dir, 'selected_features.pkl')
        with open(selected_features_path, 'wb') as f:
            pickle.dump({'selected_features': selected_features, 'n_features': len(selected_features)}, f)
        print(f"  - Selected Features Metadata: {selected_features_path}")
        print(f"    Features: {selected_features}")
        
        # Save scaler for live monitoring (prevent unnecessary CIC dataset reprocessing)
        scaler = data_splits['scaler']
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  - Scaler: {scaler_path}")
        
        print("\n" + "-" * 80)
        print("TRAINING SUMMARY")
        print("-" * 80)
        
        print(f"\n[OK] Cascaded Hybrid IDS training completed successfully!")
        print(f"     Trained on FULL DATASET (2017 + 2018)")
        
        print(f"\nDataset Coverage:")
        if available_2017:
            print(f"  - CIC-IDS2017: {len(available_2017)} files")
        if available_2018:
            print(f"  - CIC-IDS2018: {len(available_2018)} files")
        
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
        print(f"  1. Run fast evaluation: python evaluate_cascaded_fast.py")
        print(f"  2. Compare with previous results")
        print(f"  3. Check if healthcare requirements are now met")
        
        print("\n" + "=" * 80)
        print("FULL DATASET TRAINING COMPLETE")
        print("=" * 80)
        print()
        
        logger.info("Cascaded training pipeline (full dataset) completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        print(f"\n[FAIL] Training failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()