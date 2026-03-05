"""Unit tests for multimodal validation module."""

import numpy as np
import pandas as pd

from src.multimodal_validation import MultimodalValidator


def test_detect_signal_columns_aliases():
    df = pd.DataFrame({
        'hr': [80, 82],
        'spo2': [98, 97],
        'temp': [36.8, 36.9],
    })

    validator = MultimodalValidator({'enabled': True})
    mapping = validator.detect_signal_columns(df)

    assert mapping['heart_rate'] == 'hr'
    assert mapping['spo2'] == 'spo2'
    assert mapping['temperature'] == 'temp'


def test_validate_dataframe_cross_modal_escalation():
    df = pd.DataFrame({
        'hr': [78, 242, 80],
        'spo2': [98, 74, 97],
        'temp': [36.9, 43.0, 36.8],
        'sys': [122, 252, 121],
        'dia': [78, 156, 79],
        'rr': [16, 45, 15],
    })

    network_scores = np.array([0.15, 0.18, 0.12])
    network_predictions = ['BENIGN', 'BENIGN', 'BENIGN']

    validator = MultimodalValidator({
        'enabled': True,
        'medical_alert_threshold': 0.5,
        'combined_alert_threshold': 0.6,
    })

    mm = validator.validate_dataframe(df, network_scores=network_scores, network_predictions=network_predictions)

    assert len(mm) == 3
    assert bool(mm.loc[1, 'cross_modal_mismatch']) is True
    assert bool(mm.loc[1, 'multimodal_alert']) is True
    assert 'cross_modal_mismatch' in mm.loc[1, 'multimodal_reason']


def test_validate_dataframe_no_medical_columns_safe_default():
    df = pd.DataFrame({'feature_a': [1, 2, 3], 'feature_b': [4, 5, 6]})
    validator = MultimodalValidator({'enabled': True})

    mm = validator.validate_dataframe(df)

    assert np.allclose(mm['medical_risk_score'].values, 0.0)
    assert mm['multimodal_alert'].sum() == 0
