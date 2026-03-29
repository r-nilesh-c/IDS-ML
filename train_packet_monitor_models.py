"""
Train Cascaded IDS models specifically for the Live Packet Monitor.

This script trains on a reduced feature set that matches what the packet monitor
can extract from live network packets (12 flow-level features).

Usage:
  # Train on all CSV files, hold out port-scan file automatically
  python train_packet_monitor_models.py --dataset-dir dataset/cic-ids2017

  # Train on specific files
  python train_packet_monitor_models.py \
      --train-data dataset/cic-ids2017/Monday-WorkingHours.pcap_ISCX.csv \
      --holdout-data dataset/cic-ids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

This will create models/packet_monitor/ with trained models suitable for live monitoring.

Key changes vs original:
  - Stage 2 always trains on [fused_score + 12 raw features] (13 dims)
  - stage2_feature_mode saved to selected_features.pkl for live monitor
  - Smart two-gate holdout decision: S2_high OR (S1 AND S2_medium)
  - SMOTE applied automatically when attack ratio < 5%
  - --stage2-candidate-percentile filters Stage 2 training to ambiguous flows
  - Threshold configs read from cascaded_ids.stage2 in default_config.yaml
"""

import argparse
import logging
import os
import pickle
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.preprocessing import PreprocessingPipeline
from src.autoencoder import AutoencoderDetector
from src.isolation_forest import IsolationForestDetector
from src.fusion import FusionModule
from src.supervised_classifier import SupervisedClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_packet_monitor.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# The ONLY features the live packet monitor can extract from live packets
PACKET_MONITOR_FEATURES = [
    'Fwd Packets/s',
    'Active Mean',
    'FIN Flag Count',
    'Fwd IAT Total',
    'Fwd PSH Flags',
    'Bwd IAT Total',
    'Subflow Fwd Packets',
    'Fwd Avg Bytes/Bulk',
    'Bwd Packet Length Max',
    'Idle Mean',
    'Flow Bytes/s',
    'Bwd Avg Bulk Rate',
]


# Config helpers

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


def resolve_fusion_config(config: Dict) -> Dict:
    """Resolve Stage-1 fusion config.

    Precedence:
      1) cascaded_ids.stage1.threshold_percentile / fusion_weights
      2) fusion.percentile / fusion weights
    """
    fusion_cfg = dict(config.get('fusion', {}))
    cascaded_cfg = config.get('cascaded_ids', {}) or {}
    stage1_cfg = cascaded_cfg.get('stage1', {}) or {}

    stage1_percentile = stage1_cfg.get('threshold_percentile')
    if stage1_percentile is not None:
        old = fusion_cfg.get('percentile')
        if old is not None and float(old) != float(stage1_percentile):
            logger.warning(
                "Config percentile mismatch: fusion.percentile=%s vs "
                "cascaded_ids.stage1.threshold_percentile=%s. "
                "Using cascaded value.",
                old, stage1_percentile,
            )
        fusion_cfg['percentile'] = stage1_percentile

    fusion_weights = stage1_cfg.get('fusion_weights')
    if isinstance(fusion_weights, dict):
        if fusion_weights.get('autoencoder') is not None:
            fusion_cfg['weight_autoencoder'] = fusion_weights['autoencoder']
        if fusion_weights.get('isolation') is not None:
            fusion_cfg['weight_isolation'] = fusion_weights['isolation']

    return fusion_cfg


def resolve_stage2_attack_thresholds(config: Dict) -> Tuple[float, float]:
    """Return (high_threshold, medium_threshold) for Stage-2 attack probability."""
    cascaded_cfg = config.get('cascaded_ids', {}) or {}
    stage2_cfg = cascaded_cfg.get('stage2', {}) or {}

    # Backward compat: single threshold -> use for both
    legacy = stage2_cfg.get('attack_probability_threshold')
    if legacy is not None:
        t = float(legacy)
        if not 0.0 < t < 1.0:
            raise ValueError('attack_probability_threshold must be in (0, 1)')
        return t, t

    high = float(stage2_cfg.get('attack_probability_threshold_high', 0.45))
    medium = float(stage2_cfg.get('attack_probability_threshold_medium', 0.25))

    if not 0.0 < high < 1.0:
        raise ValueError('threshold_high must be in (0, 1)')
    if not 0.0 < medium < 1.0:
        raise ValueError('threshold_medium must be in (0, 1)')
    if medium > high:
        raise ValueError('threshold_medium must be <= threshold_high')

    return high, medium


# Data helpers

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
            return df.rename(columns={col: 'Label'}) if col != 'Label' else df
    raise ValueError('No label column found in dataset')


def get_available_features(df: pd.DataFrame, target_features: List[str]) -> List[str]:
    df_cols_norm = {str(c).strip(): c for c in df.columns}
    available = []
    for target in target_features:
        t_norm = target.strip().lower()
        for col_norm, col_actual in df_cols_norm.items():
            if col_norm.lower() == t_norm:
                available.append(col_actual)
                break
    return available


def discover_csv_files(dataset_dir: str) -> List[str]:
    csv_files: List[str] = []
    for root, _, files in os.walk(dataset_dir):
        for name in files:
            if name.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, name))
    return sorted(csv_files)


def resolve_csv_path(path: str, dataset_dir: str, arg_name: str) -> str:
    """Resolve CSV path from absolute, relative, or bare filename inputs.

    Resolution order:
      1) as provided (absolute/relative to CWD)
      2) under dataset_dir preserving relative subpath
      3) recursive basename match inside dataset_dir
    """
    if path is None:
        raise ValueError(f'{arg_name} cannot be None')

    raw = str(path).strip().strip('"').strip("'")
    if not raw:
        raise ValueError(f'{arg_name} cannot be empty')

    # 1) direct resolution
    direct = os.path.abspath(raw)
    if os.path.isfile(direct):
        return direct

    dataset_root = os.path.abspath(dataset_dir)

    # 2) dataset-relative resolution
    ds_joined = os.path.abspath(os.path.join(dataset_root, raw))
    if os.path.isfile(ds_joined):
        return ds_joined

    # 3) recursive basename lookup in dataset dir
    basename = os.path.basename(raw).lower()
    matches: List[str] = []
    for root, _, files in os.walk(dataset_root):
        for name in files:
            if name.lower() == basename:
                matches.append(os.path.join(root, name))

    if len(matches) == 1:
        logger.info('Resolved %s by filename match: %s -> %s', arg_name, raw, matches[0])
        return os.path.abspath(matches[0])

    if len(matches) > 1:
        raise FileNotFoundError(
            f"{arg_name} '{raw}' matched multiple files under '{dataset_root}'. "
            f"Please pass a more specific path. Matches: {matches}"
        )

    raise FileNotFoundError(
        f"{arg_name} not found: '{raw}'. Checked CWD, '{dataset_root}', and recursive filename lookup."
    )


def choose_default_holdout(csv_files: List[str]) -> str:
    preferred = [
        p for p in csv_files
        if 'friday-workinghours-afternoon-portscan' in os.path.basename(p).lower()
    ]
    return preferred[0] if preferred else csv_files[-1]


def load_and_merge_csvs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = load_csv_with_fallback(path)
        df = standardize_label_column(df)
        frames.append(df)
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def preprocess_for_packet_monitor_df(
    train_df: pd.DataFrame,
    preprocessing: PreprocessingPipeline,
) -> Tuple[Dict, List[str]]:
    """Clean and split a training DataFrame using packet monitor features."""
    available = get_available_features(train_df, PACKET_MONITOR_FEATURES)
    missing = set(PACKET_MONITOR_FEATURES) - {c.strip() for c in available}
    if missing:
        logger.warning('Missing packet monitor features: %s', missing)

    selected = available
    logger.info('Selected %d features: %s', len(selected), selected)

    df_clean = preprocessing.clean_data(train_df[selected + ['Label']])
    benign_df, attack_df = preprocessing.split_benign_attack(df_clean)
    logger.info('Benign: %d  Attack: %d', len(benign_df), len(attack_df))

    data_splits = preprocessing.normalize_and_split(
        benign_df[selected + ['Label']],
        attack_df[selected + ['Label']],
    )
    return data_splits, selected


# Stage 2 feature builder - always fused_score + 12 raw features

def build_stage2_features(fused_scores: np.ndarray, X_raw: np.ndarray) -> np.ndarray:
    """Stack [fused_score | raw_features] -> shape (n, 13)."""
    return np.column_stack([fused_scores.reshape(-1, 1), X_raw])


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train Cascaded IDS models for Live Packet Monitor'
    )
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--train-data', type=str, default=None,
                        help='Single training CSV. If omitted, all CSVs under --dataset-dir are used.')
    parser.add_argument('--holdout-data', type=str, default=None,
                        help='Holdout CSV. Required when --train-data is used.')
    parser.add_argument('--dataset-dir', type=str, default='dataset',
                        help='Root folder to search for CSVs when --train-data is omitted.')
    parser.add_argument('--output-dir', type=str, default='models/packet_monitor')
    parser.add_argument(
        '--stage2-label-mode',
        type=str,
        choices=['binary', 'multiclass'],
        default='multiclass',
        help='Stage-2 training labels: multiclass enables attack-type prediction; binary uses BENIGN vs ATTACK.',
    )
    parser.add_argument(
        '--stage2-candidate-percentile', type=float, default=None,
        help='Loose Stage-1 percentile (e.g. 85) to filter Stage-2 training to ambiguous flows.',
    )
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    preprocessing_cfg = dict(config.get('preprocessing', {}))
    random_state = preprocessing_cfg.get('random_state', 42)
    set_random_seeds(random_state)

    preprocessing = PreprocessingPipeline(preprocessing_cfg)

    # Load data
    if args.train_data:
        if not args.holdout_data:
            raise ValueError('--holdout-data is required when --train-data is used')
        train_path = resolve_csv_path(args.train_data, args.dataset_dir, '--train-data')
        holdout_path = resolve_csv_path(args.holdout_data, args.dataset_dir, '--holdout-data')

        train_df = load_csv_with_fallback(train_path)
        train_df = standardize_label_column(train_df)
        training_source_desc = train_path
    else:
        csv_files = discover_csv_files(args.dataset_dir)
        if not csv_files:
            raise ValueError(f'No CSV files found under: {args.dataset_dir}')

        if args.holdout_data:
            holdout_path = resolve_csv_path(args.holdout_data, args.dataset_dir, '--holdout-data')
        else:
            holdout_path = choose_default_holdout(csv_files)

        holdout_abs = os.path.abspath(holdout_path)
        training_files = [p for p in csv_files if os.path.abspath(p) != holdout_abs]
        if not training_files:
            raise ValueError('No training CSV files remain after excluding holdout.')

        logger.info('Auto mode: %d training files, holdout=%s', len(training_files), holdout_path)
        train_df = load_and_merge_csvs(training_files)
        training_source_desc = f'{len(training_files)} CSV files under {args.dataset_dir}'

    print('=' * 80)
    print('PACKET MONITOR CASCADED TRAINING')
    print('=' * 80)
    print(f'  Training data : {training_source_desc}')
    print(f'  Holdout data  : {holdout_path}')
    print(f'  Output dir    : {args.output_dir}')
    print(f'  Features      : {len(PACKET_MONITOR_FEATURES)} packet monitor features + fused score for Stage 2')
    print('=' * 80)

    # Preprocess
    data_splits, selected_features = preprocess_for_packet_monitor_df(train_df, preprocessing)

    X_train_benign = data_splits['X_train_benign']
    X_val_benign = data_splits['X_val_benign']
    X_train = data_splits['X_train']
    y_train_binary = data_splits['y_train_binary']
    y_train_labels = data_splits['y_train']

    logger.info('Benign train=%d  val=%d  full train=%d',
                len(X_train_benign), len(X_val_benign), len(X_train))

    # Stage 1: Autoencoder
    print('\n[Stage 1.1] Training Autoencoder...')
    ae_config = config.get('autoencoder', {})
    autoencoder = AutoencoderDetector(input_dim=X_train_benign.shape[1], config=ae_config)
    autoencoder.build_model(use_dropout=True, dropout_rate=0.2)
    history = autoencoder.train(X_train_benign, X_val_benign)
    print(f'  Epochs: {len(history.history["loss"])}  '
          f'val_loss: {history.history["val_loss"][-1]:.6f}')

    # Stage 1: Isolation Forest
    print('\n[Stage 1.2] Training Isolation Forest...')
    iso_forest = IsolationForestDetector(config=config.get('isolation_forest', {}))
    X_train_benign_all = np.vstack([X_train_benign, X_val_benign])
    iso_forest.train(X_train_benign_all)
    print(f'  Trained on {len(X_train_benign_all)} benign samples')

    # Stage 1: Fusion threshold
    print('\n[Stage 1.3] Fitting Fusion Module...')
    fusion_config = resolve_fusion_config(config)
    fusion = FusionModule(config=fusion_config)

    ae_val = autoencoder.compute_reconstruction_error(X_val_benign)
    iso_val = iso_forest.compute_anomaly_score(X_val_benign)
    fusion.fit_threshold(ae_val, iso_val)

    threshold = float(fusion.threshold)
    print(f'  Threshold: {threshold:.6f}  (percentile={fusion.percentile})')
    print(f'  AE weight: {fusion.weight_autoencoder}  IF weight: {fusion.weight_isolation}')

    # Stage 2: Build features [fused_score | raw_12]
    print('\n[Stage 2] Building Stage 2 training features...')

    ae_full = autoencoder.compute_reconstruction_error(X_train)
    iso_full = iso_forest.compute_anomaly_score(X_train)
    fused_full = fusion.compute_combined_score(ae_full, iso_full)

    # Always fused_plus_raw - 13 features total
    X_stage2_all = build_stage2_features(fused_full, X_train)
    if args.stage2_label_mode == 'multiclass':
        y_stage2_all = y_train_labels
        logger.info('Stage-2 label mode: multiclass (BENIGN + specific attack types)')
    else:
        y_stage2_all = y_train_binary
        logger.info('Stage-2 label mode: binary (BENIGN vs ATTACK)')

    feature_names_stage2 = ['fused_score'] + list(selected_features)
    logger.info('Stage 2 input shape: %s', X_stage2_all.shape)

    # Optional: filter to ambiguous flows only (Stage 1 uncertain region)
    if args.stage2_candidate_percentile is not None:
        p = float(args.stage2_candidate_percentile)
        loose_thresh = float(np.percentile(fused_full, p))
        candidates = fused_full > loose_thresh
        candidate_count = int(np.sum(candidates))

        if candidate_count > 0 and np.unique(y_stage2_all[candidates]).size >= 2:
            X_stage2 = X_stage2_all[candidates]
            y_stage2 = y_stage2_all[candidates]
            logger.info(
                'Stage-2 candidate filter: percentile=%.1f  threshold=%.6f  '
                'candidates=%d/%d',
                p, loose_thresh, candidate_count, len(fused_full),
            )
        else:
            logger.warning('Candidate filter produced single class or 0 samples - using full set.')
            X_stage2 = X_stage2_all
            y_stage2 = y_stage2_all
    else:
        X_stage2 = X_stage2_all
        y_stage2 = y_stage2_all

    y_stage2_series = pd.Series(y_stage2)
    benign_mask = y_stage2_series.astype(str).str.upper() == 'BENIGN'
    if args.stage2_label_mode == 'binary':
        benign_count = int(np.sum(np.asarray(y_stage2) == 0))
        attack_count = int(np.sum(np.asarray(y_stage2) == 1))
    else:
        benign_count = int(np.sum(benign_mask))
        attack_count = int(len(y_stage2_series) - benign_count)

    print(f'\n  Stage 2 training distribution:')
    print(f'    Benign : {benign_count}')
    print(f'    Attack : {attack_count}')
    print(f'    Ratio  : {attack_count / max(1, benign_count + attack_count):.4f}')

    if args.stage2_label_mode == 'multiclass':
        attack_breakdown = y_stage2_series[~benign_mask].value_counts().head(10)
        print('    Top attack labels:')
        for label, count in attack_breakdown.items():
            print(f'      - {label}: {int(count)}')

    # Auto-SMOTE when attacks are rare
    attack_ratio = attack_count / max(1, benign_count + attack_count)
    if attack_ratio < 0.05:
        try:
            from imblearn.over_sampling import SMOTE
            logger.info('Attack ratio %.4f < 0.05 - applying SMOTE', attack_ratio)
            X_stage2, y_stage2 = SMOTE(random_state=random_state).fit_resample(X_stage2, y_stage2)
            logger.info(
                'After SMOTE: total=%d  benign=%d  attack=%d',
                len(X_stage2), int(np.sum(y_stage2 == 0)), int(np.sum(y_stage2 == 1)),
            )
        except Exception as e:
            logger.warning('SMOTE failed - proceeding without it: %s', e)

    # Stage 2: Train RF
    print('\n[Stage 2] Training Random Forest...')
    classifier_cfg = dict(config.get('supervised_classifier', {}))
    classifier_cfg['random_state'] = random_state
    supervised = SupervisedClassifier(classifier_cfg)
    supervised.train(X_stage2, y_stage2, feature_names=feature_names_stage2)
    print(f'  Training complete  input_dim={X_stage2.shape[1]}')

    # Show top features
    try:
        fi = supervised.get_feature_importance(n=10)
        print('\n  Top 10 features by importance:')
        for i, (feat, imp) in enumerate(fi.items(), 1):
            print(f'    {i:2d}. {feat:<30s} {imp:.4f}')
    except Exception:
        pass

    # Save models
    print(f'\n[Save] Writing models to {args.output_dir}...')

    autoencoder.model.save(os.path.join(args.output_dir, 'autoencoder_best.keras'))

    with open(os.path.join(args.output_dir, 'isolation_forest.pkl'), 'wb') as f:
        pickle.dump(iso_forest.model, f)

    with open(os.path.join(args.output_dir, 'fusion_params.pkl'), 'wb') as f:
        pickle.dump({
            'weight_autoencoder': fusion.weight_autoencoder,
            'weight_isolation': fusion.weight_isolation,
            'percentile': fusion.percentile,
            'recon_min': fusion.recon_min,
            'recon_max': fusion.recon_max,
            'iso_min': fusion.iso_min,
            'iso_max': fusion.iso_max,
            'threshold': fusion.threshold,
        }, f)

    supervised.save(os.path.join(args.output_dir, 'supervised_classifier.pkl'))

    with open(os.path.join(args.output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(data_splits['scaler'], f)

    # Save feature metadata - live monitor reads this to build Stage 2 input
    with open(os.path.join(args.output_dir, 'selected_features.pkl'), 'wb') as f:
        pickle.dump({
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'feature_names_stage2': feature_names_stage2,
            'stage2_input_dim': X_stage2.shape[1],
            'stage2_uses_fused': True,
            'stage2_feature_mode': 'fused_plus_raw',
            'stage2_label_mode': args.stage2_label_mode,
            'stage2_classes': sorted(pd.Series(y_stage2).astype(str).unique().tolist()),
        }, f)

    print('  Models saved successfully.')

    # Holdout evaluation
    print(f'\n[Eval] Loading holdout: {holdout_path}')
    holdout_df = load_csv_with_fallback(holdout_path)
    holdout_df = standardize_label_column(holdout_df)

    avail_holdout = get_available_features(holdout_df, selected_features)
    if len(avail_holdout) < len(selected_features):
        logger.warning('Holdout missing features - using %d/%d', len(avail_holdout), len(selected_features))
        eval_features = avail_holdout
    else:
        eval_features = selected_features

    holdout_clean = preprocessing.clean_data(holdout_df[eval_features + ['Label']])
    X_holdout = data_splits['scaler'].transform(holdout_clean[eval_features].values)
    y_holdout = (holdout_clean['Label'].astype(str).str.upper() != 'BENIGN').astype(int).values

    ae_h = autoencoder.compute_reconstruction_error(X_holdout)
    iso_h = iso_forest.compute_anomaly_score(X_holdout)
    fused_h = fusion.compute_combined_score(ae_h, iso_h)

    X_holdout_s2 = build_stage2_features(fused_h, X_holdout)

    preds_stage1 = fused_h > threshold
    _, stage2_proba = supervised.predict(X_holdout_s2)

    # Extract attack confidence column
    class_labels = getattr(supervised.model, 'classes_', np.array([0, 1]))
    if hasattr(stage2_proba, 'ndim') and stage2_proba.ndim == 2:
        attack_idx = None
        for idx, cls in enumerate(class_labels):
            if isinstance(cls, (int, np.integer, float, np.floating)) and int(cls) == 1:
                attack_idx = idx
                break
            if str(cls).strip().upper() in {'ATTACK', 'MALICIOUS'}:
                attack_idx = idx
                break

        if attack_idx is None:
            benign_idx = next(
                (i for i, c in enumerate(class_labels)
                 if (isinstance(c, (int, np.integer)) and int(c) == 0)
                 or str(c).strip().upper() == 'BENIGN'),
                None,
            )
            if benign_idx is not None and stage2_proba.shape[1] > 1:
                attack_indices = [i for i in range(stage2_proba.shape[1]) if i != benign_idx]
                confidences = np.max(stage2_proba[:, attack_indices], axis=1)
            else:
                confidences = np.max(stage2_proba, axis=1)
        else:
            confidences = stage2_proba[:, attack_idx]
    else:
        confidences = np.asarray(stage2_proba).reshape(-1)

    # Two-gate decision
    high_t, med_t = resolve_stage2_attack_thresholds(config)
    logger.info('Stage-2 thresholds - high=%.3f  medium=%.3f', high_t, med_t)

    stage2_high = confidences > high_t
    stage2_med = confidences > med_t

    # ATTACK if: S2 very confident (regardless of S1)
    #         OR S1 flagged AND S2 moderately agrees
    final_preds = stage2_high | (preds_stage1 & stage2_med)

    from sklearn.metrics import (
        confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,
    )

    cm = confusion_matrix(y_holdout, final_preds)
    tn, fp, fn, tp = cm.ravel()

    print('\n' + '=' * 80)
    print('HOLDOUT EVALUATION')
    print('=' * 80)
    print(f'True Negatives  : {tn}')
    print(f'False Positives : {fp}')
    print(f'False Negatives : {fn}')
    print(f'True Positives  : {tp}')
    print(f'Precision       : {precision_score(y_holdout, final_preds, zero_division=0):.4f}')
    print(f'Recall          : {recall_score(y_holdout, final_preds, zero_division=0):.4f}')
    print(f'F1-Score        : {f1_score(y_holdout, final_preds, zero_division=0):.4f}')

    if len(np.unique(y_holdout)) > 1:
        try:
            print(f'ROC-AUC         : {roc_auc_score(y_holdout, confidences):.4f}')
        except Exception as e:
            logger.debug('ROC-AUC failed: %s', e)

    # Diagnostics
    s1_pos = int(np.sum(preds_stage1))
    s1_tp = int(np.sum(preds_stage1 & (y_holdout == 1)))
    s1_fp = int(np.sum(preds_stage1 & (y_holdout == 0)))
    s2h_pos = int(np.sum(stage2_high))
    s2h_tp = int(np.sum(stage2_high & (y_holdout == 1)))
    s2h_fp = int(np.sum(stage2_high & (y_holdout == 0)))
    s2m_pos = int(np.sum(stage2_med))
    s2m_tp = int(np.sum(stage2_med & (y_holdout == 1)))
    s2m_fp = int(np.sum(stage2_med & (y_holdout == 0)))

    print('-' * 80)
    print('DIAGNOSTICS')
    print('-' * 80)
    print(f'Stage 1 positives      : {s1_pos}/{len(y_holdout)}  | TP={s1_tp}  FP={s1_fp}')
    print(f'Stage 2 HIGH positives : {s2h_pos}/{len(y_holdout)}  | TP={s2h_tp}  FP={s2h_fp}')
    print(f'Stage 2 MED positives  : {s2m_pos}/{len(y_holdout)}  | TP={s2m_tp}  FP={s2m_fp}')

    if s1_pos > 0:
        s2_on_s1 = stage2_med[preds_stage1]
        y_on_s1 = y_holdout[preds_stage1]
        gate_tp = int(np.sum(s2_on_s1 & (y_on_s1 == 1)))
        gate_fp = int(np.sum(s2_on_s1 & (y_on_s1 == 0)))
        print(
            f'Stage 2 on S1 subset   : {int(np.sum(s2_on_s1))}/{s1_pos}'
            f'  | TP={gate_tp}  FP={gate_fp}'
        )

    # Attack coverage breakdown
    total_attacks = int(np.sum(y_holdout == 1))
    s1_missed = int(np.sum((preds_stage1 == 0) & (y_holdout == 1)))
    s2_recovered = int(np.sum(stage2_high & (preds_stage1 == 0) & (y_holdout == 1)))
    print('-' * 80)
    print(f'Total attacks in holdout : {total_attacks}')
    print(f'Missed by Stage 1        : {s1_missed}  ({s1_missed / max(1, total_attacks) * 100:.1f}%)')
    print(f'Recovered by S2 HIGH     : {s2_recovered}')
    print(f'Final TP                 : {tp}  ({tp / max(1, total_attacks) * 100:.1f}% recall)')
    print('=' * 80)

    logger.info('Training complete. Models saved to %s', args.output_dir)
    logger.info('Run live monitor with: python live_packet_monitor.py --model-dir %s', args.output_dir)


if __name__ == '__main__':
    main()