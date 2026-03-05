"""
Train cascaded IDS on multimodal (network + medical) data.

This script performs full multimodal training:
- Stage 1: Autoencoder + Isolation Forest + Fusion
- Stage 2: Supervised classifier

Usage:
  conda run -n hybrid-ids python train_cascaded_multimodal.py \
      --train-data data/multimodal/multimodal_train.csv \
      --holdout-data data/multimodal/multimodal_holdout.csv
"""

import argparse
import logging
import os
import pickle
import random
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.autoencoder import AutoencoderDetector
from src.fusion import FusionModule
from src.isolation_forest import IsolationForestDetector
from src.preprocessing import PreprocessingPipeline
from src.supervised_classifier import SupervisedClassifier


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/training_multimodal.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except ImportError:
        pass


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_csv_with_fallback(path: str) -> pd.DataFrame:
    for enc in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Failed to read CSV: {path}')


def standardize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if str(col).strip().lower() == 'label':
            if col != 'Label':
                return df.rename(columns={col: 'Label'})
            return df
    raise ValueError('No label column found in dataset')


def preprocess_train_data(
    train_data_path: str,
    preprocessing: PreprocessingPipeline,
    n_features: int,
    selection_method: str,
) -> Tuple[Dict, list]:
    train_df = load_csv_with_fallback(train_data_path)
    train_df = standardize_label_column(train_df)
    train_df_clean = preprocessing.clean_data(train_df)

    benign_df, attack_df = preprocessing.split_benign_attack(train_df_clean)

    benign_df_selected, attack_df_selected, selected_features = preprocessing.select_features(
        benign_df,
        attack_df,
        n_features=n_features,
        method=selection_method,
    )

    data_splits = preprocessing.normalize_and_split(benign_df_selected, attack_df_selected)
    return data_splits, selected_features


def preprocess_holdout_data(
    holdout_data_path: str,
    preprocessing: PreprocessingPipeline,
    selected_features: list,
    scaler,
) -> Tuple[np.ndarray, np.ndarray]:
    holdout_df = load_csv_with_fallback(holdout_data_path)
    holdout_df = standardize_label_column(holdout_df)
    holdout_df_clean = preprocessing.clean_data(holdout_df)

    missing = [f for f in selected_features if f not in holdout_df_clean.columns]
    if missing:
        raise ValueError(f'Holdout missing selected features: {missing[:10]}')

    X = holdout_df_clean[selected_features].values
    X_scaled = scaler.transform(X)
    y = (holdout_df_clean['Label'].astype(str).str.upper() != 'BENIGN').astype(int).values
    return X_scaled, y


def main() -> None:
    parser = argparse.ArgumentParser(description='Train cascaded IDS on multimodal data')
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--train-data', type=str, default='data/multimodal/multimodal_train.csv')
    parser.add_argument('--holdout-data', type=str, default='data/multimodal/multimodal_holdout.csv')
    parser.add_argument('--output-dir', type=str, default='models/multimodal')
    parser.add_argument('--n-features', type=int, default=36)
    parser.add_argument('--feature-method', type=str, default='statistical', choices=['variance', 'statistical'])
    parser.add_argument('--optimize-stage2', action='store_true')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    preprocessing_config = dict(config.get('preprocessing', {}))
    preprocessing_config['feature_selection'] = True
    preprocessing_config['n_features'] = args.n_features
    preprocessing_config['feature_selection_method'] = args.feature_method

    random_state = preprocessing_config.get('random_state', 42)
    set_random_seeds(random_state)

    preprocessing = PreprocessingPipeline(preprocessing_config)

    print('=' * 80)
    print('MULTIMODAL CASCADED TRAINING')
    print('=' * 80)

    data_splits, selected_features = preprocess_train_data(
        train_data_path=args.train_data,
        preprocessing=preprocessing,
        n_features=args.n_features,
        selection_method=args.feature_method,
    )

    X_train_benign = data_splits['X_train_benign']
    X_val_benign = data_splits['X_val_benign']
    X_train_full = data_splits['X_train']
    y_train_full = data_splits['y_train']
    X_internal_test = data_splits['X_test']
    y_internal_test = data_splits['y_test']

    n_features = X_train_benign.shape[1]

    print(f'Train benign: {X_train_benign.shape}')
    print(f'Train full: {X_train_full.shape}')
    print(f'Selected features ({len(selected_features)}): {selected_features[:12]}...')

    ae_config = dict(config.get('autoencoder', {}))
    ae_config['random_state'] = random_state
    ae_config['model_save_path'] = args.output_dir

    autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
    autoencoder.build_model(use_dropout=True, dropout_rate=0.2)
    autoencoder.train(X_train_benign, X_val_benign)

    if_config = dict(config.get('isolation_forest', {}))
    isolation_forest = IsolationForestDetector(if_config)
    isolation_forest.train(np.vstack([X_train_benign, X_val_benign]))

    recon_val = autoencoder.compute_reconstruction_error(X_val_benign)
    iso_val = isolation_forest.compute_anomaly_score(X_val_benign)

    fusion = FusionModule(config.get('fusion', {}))
    fusion.fit_threshold(recon_val, iso_val)

    classifier = SupervisedClassifier(config.get('supervised_classifier', {}))
    classifier.train(
        X_train_full,
        y_train_full,
        feature_names=selected_features,
        optimize_hyperparameters=args.optimize_stage2,
    )

    # Quick internal binary sanity check using Stage 2 only
    internal_pred, _ = classifier.predict(X_internal_test)
    internal_pred_bin = (np.asarray(internal_pred).astype(str) != 'BENIGN').astype(int)
    internal_acc = float((internal_pred_bin == y_internal_test).mean())

    # Optional holdout check with selected features and saved scaler
    holdout_metrics = {}
    if os.path.exists(args.holdout_data):
        X_holdout, y_holdout = preprocess_holdout_data(
            holdout_data_path=args.holdout_data,
            preprocessing=preprocessing,
            selected_features=selected_features,
            scaler=data_splits['scaler'],
        )
        holdout_pred, _ = classifier.predict(X_holdout)
        holdout_pred_bin = (np.asarray(holdout_pred).astype(str) != 'BENIGN').astype(int)
        holdout_acc = float((holdout_pred_bin == y_holdout).mean())
        holdout_metrics = {
            'holdout_samples': int(len(y_holdout)),
            'holdout_accuracy_stage2_binary': holdout_acc,
        }

    # Save artifacts
    with open(os.path.join(args.output_dir, 'isolation_forest.pkl'), 'wb') as f:
        pickle.dump(isolation_forest.model, f)

    fusion_params = {
        'weight_autoencoder': fusion.weight_autoencoder,
        'weight_isolation': fusion.weight_isolation,
        'percentile': fusion.percentile,
        'recon_min': fusion.recon_min,
        'recon_max': fusion.recon_max,
        'iso_min': fusion.iso_min,
        'iso_max': fusion.iso_max,
        'threshold': fusion.threshold,
    }
    with open(os.path.join(args.output_dir, 'fusion_params.pkl'), 'wb') as f:
        pickle.dump(fusion_params, f)

    classifier.save(os.path.join(args.output_dir, 'supervised_classifier.pkl'))

    with open(os.path.join(args.output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(data_splits['scaler'], f)

    with open(os.path.join(args.output_dir, 'selected_features.pkl'), 'wb') as f:
        pickle.dump({'selected_features': selected_features, 'n_features': len(selected_features)}, f)

    summary = {
        'train_data': args.train_data,
        'holdout_data': args.holdout_data,
        'output_dir': args.output_dir,
        'selected_feature_count': len(selected_features),
        'fusion_threshold': float(fusion.threshold),
        'internal_stage2_binary_accuracy': internal_acc,
        **holdout_metrics,
    }

    summary_path = os.path.join('reports', 'multimodal_training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('-' * 80)
    print('Training completed')
    print(f'Artifacts: {args.output_dir}')
    print(f'Summary:   {summary_path}')
    print(summary)


if __name__ == '__main__':
    main()
