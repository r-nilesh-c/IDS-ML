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

import pandas as pd
import yaml

# Silence TensorFlow C++ backend logs in live monitor output.
# 3 hides INFO/WARNING/ERROR messages that are often noisy but non-fatal in this script.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))


# ===== ANSI Color Codes for Terminal Output =====
class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    
    @staticmethod
    def success(text: str) -> str:
        """Green success text"""
        return f"{Colors.GREEN}{Colors.BOLD}{text}{Colors.RESET}"
    
    @staticmethod
    def error(text: str) -> str:
        """Red error/alert text"""
        return f"{Colors.RED}{Colors.BOLD}{text}{Colors.RESET}"
    
    @staticmethod
    def warning(text: str) -> str:
        """Yellow warning text"""
        return f"{Colors.YELLOW}{Colors.BOLD}{text}{Colors.RESET}"
    
    @staticmethod
    def info(text: str) -> str:
        """Cyan info text"""
        return f"{Colors.CYAN}{text}{Colors.RESET}"
    
    @staticmethod
    def header(text: str) -> str:
        """Bold white header"""
        return f"{Colors.WHITE}{Colors.BOLD}{text}{Colors.RESET}"


def cprint(message: str = '') -> None:
    """Print immediately so monitoring output is visible in real time."""
    print(message, flush=True)

from src.autoencoder import AutoencoderDetector
from src.cascaded_detector import CascadedDetector
from src.fusion import FusionModule
from src.isolation_forest import IsolationForestDetector
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


def normalize_attack_type(result: Dict) -> str:
    """Return a human-readable network attack type and avoid raw NaN/None values."""
    pred = str(result.get('prediction', 'BENIGN')).upper()
    if pred == 'BENIGN':
        # Keep benign records explicit to avoid ATTACK/BENIGN semantic confusion.
        return 'N/A'

    raw = result.get('attack_type')
    if raw is not None and not pd.isna(raw):
        text = str(raw).strip()
        if text and text.lower() != 'nan':
            if text.strip().upper() in {'BENIGN', '0', 'N/A'}:
                return 'Network Attack'
            return text

    return 'Network Attack'


# ===== Terminal Alert Functions =====
def print_system_ok(timestamp: str = None) -> None:
    """Print green 'System OK' status banner"""
    ts = timestamp or datetime.now().strftime("%H:%M:%S")
    cprint(f"\n{Colors.success('✓ SYSTEM OK')} | {Colors.info(ts)} | No threats detected")


def print_threat_alert(filename: str, anomaly_count: int, total_samples: int, 
                       anomaly_rate: float, elapsed_s: float) -> None:
    """Print prominent threat detection alert"""
    rate_pct = f"{anomaly_rate*100:.1f}%"
    cprint(f"\n{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD} ⚠️  THREATS DETECTED ⚠️ {Colors.RESET}")
    cprint(f"{Colors.error('═'*70)}")
    cprint(f"  File    : {Colors.header(filename)}")
    cprint(f"  Threats : {Colors.error(f'{anomaly_count}/{total_samples}')} samples flagged ({rate_pct})")
    cprint(f"  Time    : {Colors.info(f'{elapsed_s:.2f}s')}")
    cprint(f"{Colors.error('═'*70)}\n")


def print_anomaly_details(results: List[Dict], source_file: str) -> None:
    """Print detailed information for each detected anomaly"""
    anomalies = [r for r in results if r.get('prediction') == 'ATTACK']
    
    if not anomalies:
        return
    
    cprint(f"{Colors.warning('ANOMALY DETAILS:')}\n")
    for idx, res in enumerate(anomalies, 1):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        sample_idx = res.get('sample_index', 0)
        
        # Threat type and stage
        attack_type = normalize_attack_type(res)
        stage = res.get('stage', 'N/A')
        network_pred = res.get('prediction', 'N/A')
        
        # Risk scores
        anomaly_score = res.get('anomaly_score', 0.0)
        confidence = res.get('confidence', 0.0)
        
        cprint(f"  [{Colors.error(f'Threat #{idx}')}] Sample #{sample_idx} @ {timestamp}")
        cprint(f"    Type      : {Colors.error(attack_type)}")
        cprint(f"    Stage     : {stage}")
        cprint(f"    Network   : {Colors.error(network_pred)} (anomaly_score={anomaly_score:.3f}, confidence={confidence:.3f})")
        cprint()


def print_status_banner() -> None:
    """Print the monitoring status banner"""
    cprint('\n' + '='*70)
    cprint(Colors.header('    HEALTHCARE IDS — CASCADED LIVE MONITOR'))
    cprint('='*70)
    cprint(f'  {Colors.success("✓ Ready")} | Watching for incoming CSV windows...')
    cprint(f'  Press Ctrl+C to stop monitoring\n')
    cprint('='*70 + '\n')


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_fusion_config(config: Dict) -> Dict:
    """Resolve Stage-1 fusion settings with explicit precedence.

    Precedence for runtime defaults:
    1) cascaded_ids.stage1.threshold_percentile / fusion_weights
    2) fusion.percentile / fusion weights

    Note: runtime threshold ultimately comes from saved fusion artifact when present.
    """
    fusion_cfg = dict(config.get('fusion', {}))
    cascaded_cfg = config.get('cascaded_ids', {}) if isinstance(config.get('cascaded_ids', {}), dict) else {}
    stage1_cfg = cascaded_cfg.get('stage1', {}) if isinstance(cascaded_cfg.get('stage1', {}), dict) else {}

    stage1_percentile = stage1_cfg.get('threshold_percentile')
    if stage1_percentile is not None:
        old = fusion_cfg.get('percentile')
        if old is not None and float(old) != float(stage1_percentile):
            logger.warning(
                "Config percentile mismatch detected: fusion.percentile=%s, "
                "cascaded_ids.stage1.threshold_percentile=%s. "
                "Using cascaded_ids.stage1.threshold_percentile as runtime default.",
                old,
                stage1_percentile,
            )
        fusion_cfg['percentile'] = stage1_percentile

    fusion_weights = stage1_cfg.get('fusion_weights')
    if isinstance(fusion_weights, dict):
        ae_w = fusion_weights.get('autoencoder')
        iso_w = fusion_weights.get('isolation')
        if ae_w is not None:
            fusion_cfg['weight_autoencoder'] = ae_w
        if iso_w is not None:
            fusion_cfg['weight_isolation'] = iso_w

    return fusion_cfg


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

    fusion = FusionModule(resolve_fusion_config(config))
    with open(fusion_path, 'rb') as f:
        fusion_params = pickle.load(f)

    artifact_percentile = fusion_params.get('percentile')
    if artifact_percentile is not None and float(artifact_percentile) != float(fusion.percentile):
        logger.warning(
            "Fusion percentile mismatch between config (%s) and artifact (%s). "
            "Runtime threshold comes from artifact.",
            fusion.percentile,
            artifact_percentile,
        )

    if artifact_percentile is not None:
        fusion.percentile = artifact_percentile

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


def run_cascaded_inference(detector: CascadedDetector, X_df: pd.DataFrame) -> List[Dict]:
    X = X_df.values
    return detector.predict_batch(X)


def log_anomalies(anomaly_log_path: str, source_file: str, results: List[Dict]) -> int:
    os.makedirs(os.path.dirname(anomaly_log_path), exist_ok=True)
    anomaly_count = 0

    with open(anomaly_log_path, 'a', encoding='utf-8') as log_file:
        for idx, res in enumerate(results):
            if res.get('prediction') != 'ATTACK':
                continue

            anomaly_count += 1
            atk_type = normalize_attack_type(res)
            entry = {
                'timestamp': datetime.now().isoformat(),
                'source_file': source_file,
                'sample_index': idx,
                'prediction': res.get('prediction'),
                'attack_type': atk_type,
                'stage': res.get('stage'),
                'anomaly_score': round(res.get('anomaly_score') or 0, 4),
                'confidence': round(res.get('confidence') or 0, 4),
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
            'attack_type': normalize_attack_type(res),
            'stage': res.get('stage'),
            'anomaly_score': res.get('anomaly_score'),
            'confidence': res.get('confidence'),
            'latency_ms': res.get('latency_ms'),
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
) -> Dict:
    started_at = time.time()
    df = pd.read_csv(file_path, low_memory=False)
    X_df = prepare_features(
        df,
        preprocessing,
        selected_features=selected_features,
        scaler=scaler,
    )

    results: List[Dict] = []
    n_rows = len(X_df)
    for batch_start in range(0, n_rows, mini_batch_size):
        batch_end = min(batch_start + mini_batch_size, n_rows)
        batch_df = X_df.iloc[batch_start:batch_end]
        batch_results = run_cascaded_inference(detector, batch_df)
        results.extend(batch_results)

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


def is_recoverable_window_error(error: Exception) -> bool:
    """Return True if the file-level error should be skipped and archived."""
    text = str(error).lower()
    markers = [
        'insufficient samples after cleaning',
        'feature dimension mismatch',
        'selected features not found',
        'stage 2 feature dimension mismatch',
        'found array with 0 sample',
        'at least one array or dtype is required',
    ]
    return any(marker in text for marker in markers)


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
    parser.add_argument('--status-seconds', type=int, default=5)
    parser.add_argument('--mini-batch-size', type=int, default=256)
    parser.add_argument('--archive-dir', type=str, default='reports/live/processed')
    parser.add_argument('--failed-dir', type=str, default='reports/live/failed')
    parser.add_argument(
        '--verbose-model-logs',
        action='store_true',
        help='Show detailed INFO logs from model internals (autoencoder, tensorflow, etc.)',
    )

    parser.add_argument('--scaler-path', type=str, default='models/scaler.pkl')
    parser.add_argument('--bootstrap-scaler', action='store_true')

    args = parser.parse_args()

    if not args.verbose_model_logs:
        noisy_loggers = [
            '__main__',
            'src.preprocessing',
            'src.autoencoder',
            'src.cascaded_detector',
            'src.isolation_forest',
            'tensorflow',
            'absl',
        ]
        for noisy_name in noisy_loggers:
            logging.getLogger(noisy_name).setLevel(logging.WARNING)

    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.archive_dir, exist_ok=True)
    os.makedirs(args.failed_dir, exist_ok=True)

    config = load_config(args.config)
    preprocessing = PreprocessingPipeline(config.get('preprocessing', {}))
    
    # Load selected features for consistent inference
    selected_features_path = os.path.join(args.model_dir, 'selected_features.pkl')
    selected_features = None
    if os.path.exists(selected_features_path):
        try:
            with open(selected_features_path, 'rb') as f:
                features_metadata = pickle.load(f)
                selected_features = features_metadata.get('selected_features', None)
                n_features = features_metadata.get('n_features', 0)
            logger.info(f'Loaded {n_features} selected features from {selected_features_path}')
            logger.info(f'Features: {selected_features}')
        except (EOFError, pickle.UnpicklingError, AttributeError, OSError) as feature_load_error:
            logger.error(f'Failed to load selected features metadata: {feature_load_error}')
            logger.error('Proceeding without selected feature metadata (dimension mismatch risk).')
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
        try:
            with open(args.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f'Loaded scaler from {args.scaler_path}')
        except (EOFError, pickle.UnpicklingError, AttributeError, OSError) as scaler_error:
            logger.error(f'Failed to load scaler from {args.scaler_path}: {scaler_error}')
            logger.error('Regenerate scaler with --bootstrap-scaler or retrain models.')
            sys.exit(1)
    else:
        logger.warning('Scaler not found; proceeding without scaling')

    if args.input_file:
        try:
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
            )
            print(json.dumps(summary, indent=2))
            return
        except Exception as input_error:
            logger.exception(f'Failed to process input file {args.input_file}: {input_error}')
            sys.exit(1)

    processed: Set[str] = set()
    detector: Optional[CascadedDetector] = None
    last_ok_status = time.time()

    print_status_banner()
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
                        if is_recoverable_window_error(preview_error):
                            logger.warning(
                                f"Skipping file with insufficient samples: {file_path} | {preview_error}"
                            )
                            archived_small = os.path.join(args.failed_dir, os.path.basename(file_path))
                            shutil.move(file_path, archived_small)
                            processed.add(file_path)
                            continue
                        raise
                    except Exception as preview_error:
                        logger.exception(f"Failed to initialize detector from {file_path}: {preview_error}")
                        failed_path = os.path.join(args.failed_dir, os.path.basename(file_path))
                        shutil.move(file_path, failed_path)
                        processed.add(file_path)
                        continue

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
                    )
                except ValueError as process_error:
                    if is_recoverable_window_error(process_error):
                        logger.warning(
                            f"Skipping file due to recoverable processing error: {file_path} | {process_error}"
                        )
                        archived_small = os.path.join(args.failed_dir, os.path.basename(file_path))
                        shutil.move(file_path, archived_small)
                        processed.add(file_path)
                        continue
                    raise
                except Exception as process_error:
                    logger.exception(f"Unexpected processing error for {file_path}: {process_error}")
                    failed_path = os.path.join(args.failed_dir, os.path.basename(file_path))
                    shutil.move(file_path, failed_path)
                    processed.add(file_path)
                    continue
                
                # Extract summary data
                n_attacks = summary.get('anomalies', 0)
                n_total = summary.get('samples', 0)
                rate = summary.get('anomaly_rate', 0)
                elapsed = summary.get('elapsed_s', 0)
                fname = os.path.basename(file_path)
                
                # Log JSON summary
                logger.info(json.dumps(summary))
                
                # Display colored alert or OK status
                if n_attacks > 0:
                    print_threat_alert(fname, n_attacks, n_total, rate, elapsed)
                    # Also print detailed anomaly information
                    try:
                        # Re-read results from the predictions CSV for detailed display
                        out_csv = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_predictions.csv")
                        if os.path.exists(out_csv):
                            pred_df = pd.read_csv(out_csv)
                            results_detail = pred_df.to_dict(orient='records')
                            print_anomaly_details(results_detail, fname)
                    except Exception as e:
                        logger.debug(f"Could not load predictions for detailed display: {e}")
                    last_ok_status = time.time()
                else:
                    print_system_ok(datetime.now().strftime("%H:%M:%S"))
                    last_ok_status = time.time()

                archived = os.path.join(args.archive_dir, os.path.basename(file_path))
                shutil.move(file_path, archived)
                processed.add(file_path)

            # Periodically print "System OK" status if no files processed recently
            current_time = time.time()
            if (current_time - last_ok_status) > args.status_seconds and len(files) == 0:
                print_system_ok(datetime.now().strftime("%H:%M:%S"))
                last_ok_status = current_time

            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            cprint(f'\n{Colors.warning("Stopping live monitor...")}\n')
            break
        except Exception as loop_error:
            logger.exception(f"Live monitor loop error: {loop_error}")
            time.sleep(max(1, args.poll_seconds))


if __name__ == '__main__':
    main()
