"""
Compare network-only cascaded predictions vs multimodal-validated predictions
on a holdout multimodal dataset.

Usage:
  conda run -n hybrid-ids python evaluate_multimodal_comparison.py \
      --data data/multimodal/multimodal_holdout.csv \
      --model-dir models/multimodal
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.autoencoder import AutoencoderDetector
from src.cascaded_detector import CascadedDetector
from src.fusion import FusionModule
from src.isolation_forest import IsolationForestDetector
from src.multimodal_validation import MultimodalValidator
from src.preprocessing import PreprocessingPipeline
from src.supervised_classifier import SupervisedClassifier


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_csv(path: str) -> pd.DataFrame:
    for enc in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Failed to load CSV: {path}')


def standardize_label(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if str(col).strip().lower() == 'label':
            if col != 'Label':
                return df.rename(columns={col: 'Label'})
            return df
    raise ValueError('No label column found')


def load_models(model_dir: str, config: Dict, input_dim: int):
    import tensorflow as tf

    autoencoder = AutoencoderDetector(input_dim=input_dim, config=config.get('autoencoder', {}))
    autoencoder.model = tf.keras.models.load_model(os.path.join(model_dir, 'autoencoder_best.keras'))

    isolation_forest = IsolationForestDetector(config.get('isolation_forest', {}))
    with open(os.path.join(model_dir, 'isolation_forest.pkl'), 'rb') as f:
        isolation_forest.model = pickle.load(f)

    fusion = FusionModule(config.get('fusion', {}))
    with open(os.path.join(model_dir, 'fusion_params.pkl'), 'rb') as f:
        fusion_params = pickle.load(f)
    fusion.recon_min = fusion_params['recon_min']
    fusion.recon_max = fusion_params['recon_max']
    fusion.iso_min = fusion_params['iso_min']
    fusion.iso_max = fusion_params['iso_max']
    fusion.threshold = fusion_params['threshold']

    classifier = SupervisedClassifier(config.get('supervised_classifier', {}))
    classifier.load(os.path.join(model_dir, 'supervised_classifier.pkl'))

    detector = CascadedDetector(config)
    detector.load_stage1(autoencoder, isolation_forest, fusion, fusion.threshold)
    detector.load_stage2(classifier)

    return detector


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'false_positive_rate': fpr,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def prepare_eval_data(df: pd.DataFrame, preprocessing: PreprocessingPipeline, model_dir: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    with open(os.path.join(model_dir, 'selected_features.pkl'), 'rb') as f:
        selected_meta = pickle.load(f)
    selected_features = selected_meta['selected_features']

    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    cleaned = preprocessing.clean_data(df)
    y_true = (cleaned['Label'].astype(str).str.upper() != 'BENIGN').astype(int).values

    missing = [c for c in selected_features if c not in cleaned.columns]
    if missing:
        raise ValueError(f'Missing selected features in eval data: {missing[:10]}')

    X_df = cleaned[selected_features].copy()
    X = scaler.transform(X_df)
    return cleaned, X, y_true


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate network-only vs multimodal comparison')
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--data', type=str, default='data/multimodal/multimodal_holdout.csv')
    parser.add_argument('--model-dir', type=str, default='models/multimodal')
    parser.add_argument('--output-json', type=str, default='reports/multimodal_comparison_metrics.json')
    parser.add_argument('--output-md', type=str, default='reports/MULTIMODAL_COMPARISON_REPORT.md')
    parser.add_argument('--max-samples', type=int, default=0, help='Optional cap for faster debug runs (0 = all samples)')
    parser.add_argument('--mini-batch-size', type=int, default=256, help='Batch size for cascaded inference progress')
    parser.add_argument('--progress-every', type=int, default=1000, help='Print progress every N processed samples')
    args = parser.parse_args()

    started = time.time()
    print('[1/6] Loading configuration and preprocessing...', flush=True)
    config = load_config(args.config)
    preprocessing = PreprocessingPipeline(config.get('preprocessing', {}))

    print(f'[2/6] Loading evaluation data: {args.data}', flush=True)
    raw_df = load_csv(args.data)
    raw_df = standardize_label(raw_df)

    print('[3/6] Preparing features (clean + select + scale)...', flush=True)
    cleaned_df, X, y_true = prepare_eval_data(raw_df, preprocessing, args.model_dir)

    if args.max_samples and args.max_samples > 0:
        n = min(args.max_samples, len(y_true))
        cleaned_df = cleaned_df.iloc[:n].copy()
        X = X[:n]
        y_true = y_true[:n]
        print(f'      Using capped sample count: {n}', flush=True)

    print(f'      Samples ready: {len(y_true)}', flush=True)

    print('[4/6] Loading trained models...', flush=True)
    detector = load_models(args.model_dir, config, X.shape[1])

    print('[5/6] Running cascaded inference...', flush=True)
    total = len(X)
    results = []
    progress_step = max(1, args.progress_every)
    mini_batch_size = max(1, args.mini_batch_size)
    infer_start = time.time()

    for batch_start in range(0, total, mini_batch_size):
        batch_end = min(batch_start + mini_batch_size, total)
        batch_results = detector.predict_batch(X[batch_start:batch_end])
        results.extend(batch_results)

        processed = batch_end
        if processed % progress_step == 0 or processed == total:
            elapsed = time.time() - infer_start
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = total - processed
            eta = (remaining / rate) if rate > 0 else 0.0
            print(
                f'      Progress: {processed}/{total} ({processed/total:.1%}) | '
                f'rate={rate:.1f} samples/s | eta={eta:.1f}s',
                flush=True,
            )

    network_pred = np.array([1 if r.get('prediction') == 'ATTACK' else 0 for r in results], dtype=int)
    network_scores = np.array([float(r.get('anomaly_score', 0.0) or 0.0) for r in results], dtype=float)

    print('[6/6] Applying multimodal validation + writing reports...', flush=True)
    validator = MultimodalValidator(config.get('multimodal_validation', {}))
    mm_df = validator.validate_dataframe(
        cleaned_df,
        network_scores=network_scores,
        network_predictions=[r.get('prediction', 'BENIGN') for r in results],
    )
    multimodal_pred = mm_df['multimodal_alert'].astype(int).values

    network_metrics = compute_binary_metrics(y_true, network_pred)
    multimodal_metrics = compute_binary_metrics(y_true, multimodal_pred)

    comparison = {
        'dataset': args.data,
        'samples': int(len(y_true)),
        'network_only': network_metrics,
        'multimodal': multimodal_metrics,
        'delta': {
            'recall': float(multimodal_metrics['recall'] - network_metrics['recall']),
            'false_positive_rate': float(multimodal_metrics['false_positive_rate'] - network_metrics['false_positive_rate']),
            'f1': float(multimodal_metrics['f1'] - network_metrics['f1']),
        },
        'escalated_samples': int(((network_pred == 0) & (multimodal_pred == 1)).sum()),
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)

    md = []
    md.append('# Multimodal Comparison Report')
    md.append('')
    md.append(f"- Dataset: `{args.data}`")
    md.append(f"- Samples: {comparison['samples']}")
    md.append('')
    md.append('## Network-only Metrics')
    md.append(f"- Accuracy: {network_metrics['accuracy']:.4f}")
    md.append(f"- Precision: {network_metrics['precision']:.4f}")
    md.append(f"- Recall: {network_metrics['recall']:.4f}")
    md.append(f"- F1: {network_metrics['f1']:.4f}")
    md.append(f"- FPR: {network_metrics['false_positive_rate']:.4f}")
    md.append('')
    md.append('## Multimodal Metrics')
    md.append(f"- Accuracy: {multimodal_metrics['accuracy']:.4f}")
    md.append(f"- Precision: {multimodal_metrics['precision']:.4f}")
    md.append(f"- Recall: {multimodal_metrics['recall']:.4f}")
    md.append(f"- F1: {multimodal_metrics['f1']:.4f}")
    md.append(f"- FPR: {multimodal_metrics['false_positive_rate']:.4f}")
    md.append('')
    md.append('## Delta (Multimodal - Network)')
    md.append(f"- Recall delta: {comparison['delta']['recall']:+.4f}")
    md.append(f"- F1 delta: {comparison['delta']['f1']:+.4f}")
    md.append(f"- FPR delta: {comparison['delta']['false_positive_rate']:+.4f}")
    md.append(f"- Escalated samples: {comparison['escalated_samples']}")

    with open(args.output_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')

    print('=' * 80)
    print('MULTIMODAL COMPARISON COMPLETE')
    print('=' * 80)
    print(f'Total runtime: {time.time() - started:.2f}s')
    print(json.dumps(comparison, indent=2))
    print(f'Report: {args.output_md}')


if __name__ == '__main__':
    main()
