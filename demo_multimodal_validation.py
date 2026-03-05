"""
Multimodal validation demo for healthcare IDS.

Demonstrates how medical-signal plausibility can escalate suspicious cases
that look benign in network-only IDS output.

Usage:
    conda run -n hybrid-ids python demo_multimodal_validation.py
"""

import json
import numpy as np
import pandas as pd

from src.multimodal_validation import MultimodalValidator


def build_demo_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 24

    df = pd.DataFrame({
        'hr': np.random.normal(78, 6, n),
        'spo2': np.random.normal(97, 1.0, n),
        'temp': np.random.normal(36.8, 0.2, n),
        'sys': np.random.normal(122, 8, n),
        'dia': np.random.normal(78, 5, n),
        'rr': np.random.normal(16, 2, n),
    })

    tamper_idx = [8, 9, 16, 17]
    df.loc[tamper_idx, 'spo2'] = [76, 74, 79, 77]
    df.loc[tamper_idx, 'hr'] = [238, 244, 232, 241]
    df.loc[tamper_idx, 'temp'] = [42.6, 43.0, 42.3, 42.8]
    df.loc[tamper_idx, 'sys'] = [244, 251, 239, 247]
    df.loc[tamper_idx, 'dia'] = [152, 158, 149, 154]
    df.loc[tamper_idx, 'rr'] = [43, 46, 41, 44]

    return df


def build_network_outputs(n: int):
    network_scores = np.random.uniform(0.08, 0.35, n)
    network_predictions = ['BENIGN'] * n

    # Simulate that network IDS catches only one obvious case
    network_scores[9] = 0.82
    network_predictions[9] = 'ATTACK'

    return network_scores, network_predictions


def main():
    df = build_demo_data()
    network_scores, network_predictions = build_network_outputs(len(df))

    validator = MultimodalValidator({
        'enabled': True,
        'network_weight': 0.55,
        'medical_weight': 0.45,
        'medical_alert_threshold': 0.50,
        'combined_alert_threshold': 0.60,
    })

    mm = validator.validate_dataframe(
        df,
        network_scores=network_scores,
        network_predictions=network_predictions,
    )

    out = df.copy()
    out['network_score'] = network_scores
    out['network_prediction'] = network_predictions
    out['medical_risk_score'] = mm['medical_risk_score']
    out['combined_risk_score'] = mm['combined_risk_score']
    out['cross_modal_mismatch'] = mm['cross_modal_mismatch']
    out['multimodal_alert'] = mm['multimodal_alert']
    out['multimodal_reason'] = mm['multimodal_reason']

    network_attacks = int((out['network_prediction'] == 'ATTACK').sum())
    multimodal_attacks = int(out['multimodal_alert'].sum())
    escalations = int(((out['network_prediction'] == 'BENIGN') & (out['multimodal_alert'])).sum())

    print('=' * 80)
    print('MULTIMODAL VALIDATION DEMO')
    print('=' * 80)
    print(f'Network-only alerts: {network_attacks}')
    print(f'Multimodal alerts:   {multimodal_attacks}')
    print(f'Escalated by medical checks: {escalations}')
    print('-' * 80)

    interesting = out[out['multimodal_alert'] | out['cross_modal_mismatch']].copy()
    print(interesting[['network_prediction', 'network_score', 'medical_risk_score', 'combined_risk_score',
                       'cross_modal_mismatch', 'multimodal_alert', 'multimodal_reason']].to_string(index=True))

    payload = {
        'network_only_alerts': network_attacks,
        'multimodal_alerts': multimodal_attacks,
        'medical_escalations': escalations,
    }
    print('-' * 80)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
