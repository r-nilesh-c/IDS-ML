"""
Live Monitoring for Cascaded IDS (Stage 1 + Stage 2).

This script supports:
1. Single-file cascaded inference for a CSV window
2. Continuous folder watching for incoming CSV windows
3. Real-time anomaly logging (JSONL)
4. Optional scaler bootstrapping from configured training datasets

Usage examples:
    python live_monitor_cascaded.py --input-file data/window.csv
    python live_monitor_cascaded.py --watch-dir data/live --poll-seconds 5
    python live_monitor_cascaded.py --watch-dir data/live --bootstrap-scaler
"""

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.autoencoder import AutoencoderDetector
from src.cascaded_detector import CascadedDetector
from src.fusion import FusionModule
from src.isolation_forest import IsolationForestDetector
from src.multimodal_validation import MultimodalValidator
from src.preprocessing import PreprocessingPipeline
from src.supervised_classifier import SupervisedClassifier


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_monitor_cascaded.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


LABEL_COLS = ['Label', ' label', 'label', ' Label']


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def bootstrap_and_save_scaler(config: Dict, scaler_path: str) -> None:
    preprocessing = PreprocessingPipeline(config.get('preprocessing', {}))
    dataset_paths = [p for p in config.get('dataset_paths', []) if os.path.exists(p)]

    if not dataset_paths:
        raise ValueError('No dataset paths available to bootstrap scaler')

    logger.info(f'Bootstrapping scaler from {len(dataset_paths)} configured datasets')
    df = preprocessing.load_datasets(dataset_paths)
    df_clean = preprocessing.clean_data(df)
    benign_df, attack_df = preprocessing.split_benign_attack(df_clean)
    splits = preprocessing.normalize_and_split(benign_df, attack_df)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(splits['scaler'], f)

    logger.info(f'Scaler bootstrapped and saved to {scaler_path}')


def resolve_model_file(model_dir: str, requested_file: str, fallback_files: List[str], model_label: str) -> str:
    requested_path = os.path.join(model_dir, requested_file)
    if os.path.exists(requested_path):
        return requested_path

    for fallback in fallback_files:
        fallback_path = os.path.join(model_dir, fallback)
        if os.path.exists(fallback_path):
            logger.warning(
                f"Requested {model_label} file '{requested_file}' not found. "
                f"Using '{fallback}' from model directory instead."
            )
            return fallback_path

    tried = [requested_file] + fallback_files
    raise FileNotFoundError(
        f"Could not find {model_label} file in '{model_dir}'. Tried: {tried}"
    )


def load_models(
    model_dir: str,
    input_dim: int,
    config: Dict,
    fusion_params_file: str,
    classifier_file: str,
):
    import tensorflow as tf

    ae_path = os.path.join(model_dir, 'autoencoder_best.keras')
    if_path = os.path.join(model_dir, 'isolation_forest.pkl')
    fusion_path = resolve_model_file(
        model_dir,
        fusion_params_file,
        ['fusion_params.pkl', 'fusion_params_balanced_30.0p.pkl'],
        'fusion parameters',
    )
    clf_path = resolve_model_file(
        model_dir,
        classifier_file,
        ['supervised_classifier.pkl', 'supervised_classifier_balanced_30.0p.pkl'],
        'supervised classifier',
    )

    autoencoder = AutoencoderDetector(input_dim=input_dim, config=config.get('autoencoder', {}))
    autoencoder.model = tf.keras.models.load_model(ae_path)

    isolation_forest = IsolationForestDetector(config.get('isolation_forest', {}))
    with open(if_path, 'rb') as f:
        isolation_forest.model = pickle.load(f)

    fusion = FusionModule(config.get('fusion', {}))
    with open(fusion_path, 'rb') as f:
        fusion_params = pickle.load(f)

    fusion.recon_min = fusion_params['recon_min']
    fusion.recon_max = fusion_params['recon_max']
    fusion.iso_min = fusion_params['iso_min']
    fusion.iso_max = fusion_params['iso_max']
    fusion.threshold = fusion_params['threshold']

    classifier = SupervisedClassifier(config.get('supervised_classifier', {}))
    classifier.load(clf_path)

    return autoencoder, isolation_forest, fusion, classifier


def prepare_features(
    df: pd.DataFrame, 
    preprocessing: PreprocessingPipeline, 
    selected_features: List[str] = None,
    scaler=None,
    return_cleaned_df: bool = False,
) -> pd.DataFrame:
    """Prepare features for inference with explicit feature selection.
    
    Args:
        df: Raw input DataFrame (may have 78 or other number of features)
        preprocessing: Preprocessing pipeline for data cleaning
        selected_features: List of feature names to extract (e.g., the saved 12 features)
        scaler: Optional StandardScaler for normalization
        
    Returns:
        pd.DataFrame with selected features only, scaled if scaler provided.
        If return_cleaned_df=True, returns tuple (X_df, df_clean)
    """
    df_clean = preprocessing.clean_data(df)
    
    # If selected features are provided, extract ONLY those features
    if selected_features is not None:
        # Verify all selected features exist in cleaned data
        available_cols = [c for c in df_clean.columns if c not in LABEL_COLS]
        missing = [f for f in selected_features if f not in available_cols]
        if missing:
            raise ValueError(
                f"Selected features not found in input data. Missing: {missing}. "
                f"Available: {available_cols}"
            )
        X_df = df_clean[selected_features].copy()
        logger.info(f"Extracted {len(selected_features)} selected features for inference")
    else:
        # Fallback: use all non-label features (backward compatibility)
        feature_cols = [c for c in df_clean.columns if c not in LABEL_COLS]
        X_df = df_clean[feature_cols].copy()
        logger.warning(
            f"No selected features provided; using all {len(feature_cols)} features. "
            "This may cause dimension mismatch. Consider loading selected_features.pkl"
        )
    
    # Apply scaling if provided
    if scaler is not None:
        expected_features = scaler.n_features_in_
        if X_df.shape[1] != expected_features:
            raise ValueError(
                f"Feature dimension mismatch: X_df has {X_df.shape[1]} features "
                f"but scaler expects {expected_features}. Ensure selected_features "
                f"are loaded and used consistently."
            )
        X_scaled = scaler.transform(X_df)
        X_df = pd.DataFrame(X_scaled, columns=X_df.columns, index=X_df.index)
    
    if return_cleaned_df:
        return X_df, df_clean
    return X_df


def apply_multimodal_post_validation(
    cleaned_df: pd.DataFrame,
    results: List[Dict],
    validator: Optional[MultimodalValidator],
) -> List[Dict]:
    """Augment network IDS results with multimodal (medical + network) validation."""
    if validator is None or len(results) == 0:
        return results

    network_scores = np.array([float(r.get('anomaly_score', 0.0) or 0.0) for r in results], dtype=float)
    network_predictions = [str(r.get('prediction', 'BENIGN')) for r in results]

    mm_df = validator.validate_dataframe(
        cleaned_df,
        network_scores=network_scores,
        network_predictions=network_predictions,
    )

    if len(mm_df) != len(results):
        raise ValueError(
            f"Multimodal validation length mismatch: results={len(results)}, multimodal={len(mm_df)}"
        )

    enriched_results = []
    for idx, result in enumerate(results):
        mm_row = mm_df.iloc[idx]
        multimodal_alert = bool(mm_row['multimodal_alert'])
        network_prediction = str(result.get('prediction', 'BENIGN')).upper()

        enriched = dict(result)
        enriched['medical_risk_score'] = float(mm_row['medical_risk_score'])
        enriched['combined_risk_score'] = float(mm_row['combined_risk_score'])
        enriched['cross_modal_mismatch'] = bool(mm_row['cross_modal_mismatch'])
        enriched['multimodal_alert'] = multimodal_alert
        enriched['multimodal_reason'] = str(mm_row['multimodal_reason'])
        enriched['multimodal_prediction'] = 'ATTACK' if multimodal_alert else network_prediction
        enriched['multimodal_escalated'] = (network_prediction != 'ATTACK') and multimodal_alert
        enriched_results.append(enriched)

    return enriched_results


def run_cascaded_inference(detector: CascadedDetector, X_df: pd.DataFrame) -> List[Dict]:
    X = X_df.values
    return detector.predict_batch(X)


def log_anomalies(anomaly_log_path: str, source_file: str, results: List[Dict]) -> int:
    os.makedirs(os.path.dirname(anomaly_log_path), exist_ok=True)
    anomaly_count = 0

    with open(anomaly_log_path, 'a', encoding='utf-8') as log_file:
        for idx, res in enumerate(results):
            network_attack = res.get('prediction') == 'ATTACK'
            multimodal_attack = res.get('multimodal_prediction') == 'ATTACK'
            if not (network_attack or multimodal_attack):
                continue

            anomaly_count += 1
            entry = {
                'timestamp': datetime.now().isoformat(),
                'source_file': source_file,
                'sample_index': idx,
                'prediction': res.get('prediction'),
                'multimodal_prediction': res.get('multimodal_prediction', res.get('prediction')),
                'attack_type': res.get('attack_type'),
                'stage': res.get('stage'),
                'anomaly_score': res.get('anomaly_score'),
                'confidence': res.get('confidence'),
                'medical_risk_score': res.get('medical_risk_score'),
                'combined_risk_score': res.get('combined_risk_score'),
                'cross_modal_mismatch': res.get('cross_modal_mismatch'),
                'multimodal_reason': res.get('multimodal_reason'),
            }
            log_file.write(json.dumps(entry) + '\n')

    return anomaly_count


def save_results(output_path: str, results: List[Dict], source_file: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []

    for idx, res in enumerate(results):
        rows.append({
            'source_file': source_file,
            'sample_index': idx,
            'prediction': res.get('prediction'),
            'multimodal_prediction': res.get('multimodal_prediction', res.get('prediction')),
            'attack_type': res.get('attack_type'),
            'stage': res.get('stage'),
            'anomaly_score': res.get('anomaly_score'),
            'confidence': res.get('confidence'),
            'latency_ms': res.get('latency_ms'),
            'medical_risk_score': res.get('medical_risk_score'),
            'combined_risk_score': res.get('combined_risk_score'),
            'cross_modal_mismatch': res.get('cross_modal_mismatch'),
            'multimodal_alert': res.get('multimodal_alert'),
            'multimodal_escalated': res.get('multimodal_escalated'),
            'multimodal_reason': res.get('multimodal_reason'),
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)


def process_file(
    file_path: str,
    detector: CascadedDetector,
    preprocessing: PreprocessingPipeline,
    selected_features: List[str],
    scaler,
    output_dir: str,
    anomaly_log_path: str,
    mini_batch_size: int,
    multimodal_validator: Optional[MultimodalValidator] = None,
) -> Dict:
    started_at = time.time()
    df = pd.read_csv(file_path, low_memory=False)
    X_df, cleaned_df = prepare_features(
        df,
        preprocessing,
        selected_features=selected_features,
        scaler=scaler,
        return_cleaned_df=True,
    )

    results: List[Dict] = []
    n_rows = len(X_df)
    for batch_start in range(0, n_rows, mini_batch_size):
        batch_end = min(batch_start + mini_batch_size, n_rows)
        batch_df = X_df.iloc[batch_start:batch_end]
        batch_results = run_cascaded_inference(detector, batch_df)
        results.extend(batch_results)

    results = apply_multimodal_post_validation(cleaned_df, results, multimodal_validator)
    anomalies = log_anomalies(anomaly_log_path, os.path.basename(file_path), results)

    out_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_predictions.csv"
    out_path = os.path.join(output_dir, out_name)
    save_results(out_path, results, os.path.basename(file_path))

    elapsed = time.time() - started_at
    total = len(results)

    summary = {
        'file': file_path,
        'samples': total,
        'anomalies': anomalies,
        'anomaly_rate': anomalies / total if total else 0.0,
        'elapsed_s': elapsed,
        'out_path': out_path,
    }
    return summary


def is_insufficient_samples_error(error: Exception) -> bool:
    return 'Insufficient samples after cleaning' in str(error)


def build_detector(config: Dict, model_dir: str, sample_feature_dim: int, fusion_params_file: str, classifier_file: str) -> CascadedDetector:
    autoencoder, isolation_forest, fusion, classifier = load_models(
        model_dir=model_dir,
        input_dim=sample_feature_dim,
        config=config,
        fusion_params_file=fusion_params_file,
        classifier_file=classifier_file,
    )

    detector = CascadedDetector(config)
    detector.load_stage1(autoencoder, isolation_forest, fusion, fusion.threshold)
    detector.load_stage2(classifier)
    return detector


def main():
    parser = argparse.ArgumentParser(description='Live cascaded IDS monitoring')
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--fusion-params-file', type=str, default='fusion_params.pkl')
    parser.add_argument('--classifier-file', type=str, default='supervised_classifier.pkl')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-file', type=str)
    group.add_argument('--watch-dir', type=str)

    parser.add_argument('--output-dir', type=str, default='reports/live')
    parser.add_argument('--anomaly-log', type=str, default='logs/live_anomalies.jsonl')
    parser.add_argument('--poll-seconds', type=int, default=5)
    parser.add_argument('--mini-batch-size', type=int, default=256)
    parser.add_argument('--archive-dir', type=str, default='reports/live/processed')

    parser.add_argument('--scaler-path', type=str, default='models/scaler.pkl')
    parser.add_argument('--bootstrap-scaler', action='store_true')

    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.archive_dir, exist_ok=True)

    config = load_config(args.config)
    preprocessing = PreprocessingPipeline(config.get('preprocessing', {}))
    multimodal_cfg = config.get('multimodal_validation', {})
    multimodal_enabled = bool(multimodal_cfg.get('enabled', False))
    multimodal_validator = MultimodalValidator(multimodal_cfg) if multimodal_enabled else None
    if multimodal_enabled:
        logger.info('Multimodal validation enabled (medical + network post-validation)')
    
    # Load selected features for consistent inference
    selected_features_path = os.path.join(args.model_dir, 'selected_features.pkl')
    selected_features = None
    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'rb') as f:
            features_metadata = pickle.load(f)
            selected_features = features_metadata.get('selected_features', None)
            n_features = features_metadata.get('n_features', 0)
        logger.info(f'Loaded {n_features} selected features from {selected_features_path}')
        logger.info(f'Features: {selected_features}')
    else:
        logger.warning(
            f'selected_features.pkl not found at {selected_features_path}. '
            'Live monitoring may encounter feature dimension mismatches. '
            'Please retrain models with the updated training script.'
        )

    if args.bootstrap_scaler:
        # Only bootstrap if explicitly requested
        logger.info("Bootstrapping scaler from configured datasets (--bootstrap-scaler flag)")
        bootstrap_and_save_scaler(config, args.scaler_path)
    elif not os.path.exists(args.scaler_path):
        # Scaler not found and not bootstrapping
        logger.error(f"Scaler not found at {args.scaler_path}")
        logger.error("Use one of:")
        logger.error("  1. Ensure training script saved scaler.pkl (run: python train_cascaded_full.py)")
        logger.error("  2. Pass --bootstrap-scaler flag to generate scaler from training data")
        raise FileNotFoundError(
            f"Scaler required but not found at {args.scaler_path}. "
            "Run training or use --bootstrap-scaler flag."
        )

    scaler = None
    if os.path.exists(args.scaler_path):
        with open(args.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f'Loaded scaler from {args.scaler_path}')
    else:
        logger.warning('Scaler not found; proceeding without scaling')

    if args.input_file:
        df_preview = pd.read_csv(args.input_file, low_memory=False)
        X_preview = prepare_features(df_preview, preprocessing, selected_features=selected_features, scaler=scaler)
        detector = build_detector(config, args.model_dir, X_preview.shape[1], args.fusion_params_file, args.classifier_file)

        summary = process_file(
            file_path=args.input_file,
            detector=detector,
            preprocessing=preprocessing,
            selected_features=selected_features,
            scaler=scaler,
            output_dir=args.output_dir,
            anomaly_log_path=args.anomaly_log,
            mini_batch_size=args.mini_batch_size,
            multimodal_validator=multimodal_validator,
        )
        print(json.dumps(summary, indent=2))
        return

    processed: Set[str] = set()
    detector: Optional[CascadedDetector] = None

    print('Watching for CSV windows... Press Ctrl+C to stop.')
    while True:
        try:
            files = sorted(
                [
                    os.path.join(args.watch_dir, f)
                    for f in os.listdir(args.watch_dir)
                    if f.lower().endswith('.csv')
                ]
            )

            for file_path in files:
                if file_path in processed:
                    continue

                if detector is None:
                    try:
                        df_preview = pd.read_csv(file_path, low_memory=False)
                        X_preview = prepare_features(
                            df_preview,
                            preprocessing,
                            selected_features=selected_features,
                            scaler=scaler,
                        )
                        detector = build_detector(
                            config,
                            args.model_dir,
                            X_preview.shape[1],
                            args.fusion_params_file,
                            args.classifier_file,
                        )
                    except ValueError as preview_error:
                        if is_insufficient_samples_error(preview_error):
                            logger.warning(
                                f"Skipping file with insufficient samples: {file_path} | {preview_error}"
                            )
                            archived_small = os.path.join(args.archive_dir, os.path.basename(file_path))
                            shutil.move(file_path, archived_small)
                            processed.add(file_path)
                            continue
                        raise

                try:
                    summary = process_file(
                        file_path=file_path,
                        detector=detector,
                        preprocessing=preprocessing,
                        selected_features=selected_features,
                        scaler=scaler,
                        output_dir=args.output_dir,
                        anomaly_log_path=args.anomaly_log,
                        mini_batch_size=args.mini_batch_size,
                        multimodal_validator=multimodal_validator,
                    )
                except ValueError as process_error:
                    if is_insufficient_samples_error(process_error):
                        logger.warning(
                            f"Skipping file with insufficient samples: {file_path} | {process_error}"
                        )
                        archived_small = os.path.join(args.archive_dir, os.path.basename(file_path))
                        shutil.move(file_path, archived_small)
                        processed.add(file_path)
                        continue
                    raise
                print(json.dumps(summary))

                archived = os.path.join(args.archive_dir, os.path.basename(file_path))
                shutil.move(file_path, archived)
                processed.add(file_path)

            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            print('\nStopping live monitor.')
            break


if __name__ == '__main__':
    main()
