"""
Multimodal validation for healthcare IDS outputs.

This module adds medical-signal plausibility checks and cross-modal
consistency checks on top of existing network IDS outputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class MultimodalValidator:
    """Validate IDS decisions using medical signals + network context."""

    DEFAULT_SIGNAL_ALIASES = {
        'heart_rate': ['heart_rate', 'hr', 'pulse_rate', 'pr'],
        'spo2': ['spo2', 'blood_oxygen', 'oxygen_saturation'],
        'temperature': ['temperature', 'temp', 'body_temp'],
        'systolic_bp': ['systolic_bp', 'sys', 'bp_sys', 'blood_pressure_sys'],
        'diastolic_bp': ['diastolic_bp', 'dia', 'bp_dia', 'blood_pressure_dia'],
        'respiration_rate': ['respiration_rate', 'rr', 'resp_rate'],
    }

    DEFAULT_SIGNAL_RULES = {
        'heart_rate': {'min': 35.0, 'max': 220.0, 'max_delta': 35.0},
        'spo2': {'min': 80.0, 'max': 100.0, 'max_delta': 6.0},
        'temperature': {'min': 34.0, 'max': 42.0, 'max_delta': 1.2},
        'systolic_bp': {'min': 70.0, 'max': 220.0, 'max_delta': 40.0},
        'diastolic_bp': {'min': 40.0, 'max': 140.0, 'max_delta': 25.0},
        'respiration_rate': {'min': 6.0, 'max': 40.0, 'max_delta': 10.0},
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.enabled = cfg.get('enabled', True)
        self.medical_weight = float(cfg.get('medical_weight', 0.45))
        self.network_weight = float(cfg.get('network_weight', 0.55))
        self.medical_alert_threshold = float(cfg.get('medical_alert_threshold', 0.5))
        self.combined_alert_threshold = float(cfg.get('combined_alert_threshold', 0.6))

        aliases = cfg.get('signal_aliases', {})
        self.signal_aliases = {
            key: aliases.get(key, defaults)
            for key, defaults in self.DEFAULT_SIGNAL_ALIASES.items()
        }

        rules = cfg.get('signal_rules', {})
        self.signal_rules = {
            key: {**defaults, **rules.get(key, {})}
            for key, defaults in self.DEFAULT_SIGNAL_RULES.items()
        }

    def detect_signal_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Map canonical signal names to actual dataframe column names."""
        column_lookup = {c.lower().strip(): c for c in df.columns}
        mapping: Dict[str, str] = {}

        for canonical, aliases in self.signal_aliases.items():
            for alias in aliases:
                match = column_lookup.get(alias.lower().strip())
                if match is not None:
                    mapping[canonical] = match
                    break

        return mapping

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        network_scores: Optional[np.ndarray] = None,
        network_predictions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute multimodal risk and alert decisions.

        Returns a DataFrame with columns:
        - medical_risk_score
        - combined_risk_score
        - cross_modal_mismatch
        - multimodal_alert
        - multimodal_reason
        """
        n_samples = len(df)
        if n_samples == 0:
            return pd.DataFrame(
                {
                    'medical_risk_score': [],
                    'combined_risk_score': [],
                    'cross_modal_mismatch': [],
                    'multimodal_alert': [],
                    'multimodal_reason': [],
                }
            )

        if not self.enabled:
            return pd.DataFrame(
                {
                    'medical_risk_score': np.zeros(n_samples),
                    'combined_risk_score': np.zeros(n_samples),
                    'cross_modal_mismatch': np.zeros(n_samples, dtype=bool),
                    'multimodal_alert': np.zeros(n_samples, dtype=bool),
                    'multimodal_reason': ['disabled'] * n_samples,
                }
            )

        signal_columns = self.detect_signal_columns(df)
        medical_score = np.zeros(n_samples, dtype=float)
        score_denominator = 0.0
        reasons: List[List[str]] = [[] for _ in range(n_samples)]

        for signal_name, rule in self.signal_rules.items():
            col_name = signal_columns.get(signal_name)
            if col_name is None:
                continue

            series = pd.to_numeric(df[col_name], errors='coerce')
            series = series.replace([np.inf, -np.inf], np.nan)

            out_of_range = ((series < rule['min']) | (series > rule['max']) | series.isna()).astype(float)
            medical_score += 0.7 * out_of_range.values
            score_denominator += 0.7

            for idx in np.where(out_of_range.values > 0)[0]:
                reasons[idx].append(f'{signal_name}_out_of_range')

            max_delta = float(rule.get('max_delta', 0.0))
            if max_delta > 0:
                deltas = series.diff().abs().fillna(0.0)
                abrupt = (deltas > max_delta).astype(float)
                medical_score += 0.3 * abrupt.values
                score_denominator += 0.3

                for idx in np.where(abrupt.values > 0)[0]:
                    reasons[idx].append(f'{signal_name}_abrupt_change')

        if score_denominator > 0:
            medical_score = np.clip(medical_score / score_denominator, 0.0, 1.0)
        else:
            medical_score = np.zeros(n_samples, dtype=float)
            for idx in range(n_samples):
                reasons[idx].append('no_medical_signal_columns')

        if network_scores is None:
            network_scores_arr = np.zeros(n_samples, dtype=float)
        else:
            network_scores_arr = np.asarray(network_scores, dtype=float)
            if len(network_scores_arr) != n_samples:
                raise ValueError(
                    f'network_scores length mismatch: expected {n_samples}, got {len(network_scores_arr)}'
                )
            network_scores_arr = np.clip(network_scores_arr, 0.0, 1.0)

        if network_predictions is None:
            network_attack = np.zeros(n_samples, dtype=bool)
        else:
            if len(network_predictions) != n_samples:
                raise ValueError(
                    f'network_predictions length mismatch: expected {n_samples}, got {len(network_predictions)}'
                )
            network_attack = np.array([str(p).upper() == 'ATTACK' for p in network_predictions], dtype=bool)

        cross_modal_mismatch = (~network_attack) & (medical_score >= self.medical_alert_threshold)

        combined = (
            self.network_weight * network_scores_arr +
            self.medical_weight * medical_score
        )

        multimodal_alert = network_attack | (combined >= self.combined_alert_threshold) | cross_modal_mismatch

        reason_text = []
        for idx in range(n_samples):
            row_reasons = list(reasons[idx])
            if cross_modal_mismatch[idx]:
                row_reasons.append('cross_modal_mismatch')
            if network_attack[idx]:
                row_reasons.append('network_attack')
            reason_text.append(';'.join(row_reasons) if row_reasons else 'none')

        return pd.DataFrame(
            {
                'medical_risk_score': medical_score,
                'combined_risk_score': np.clip(combined, 0.0, 1.0),
                'cross_modal_mismatch': cross_modal_mismatch,
                'multimodal_alert': multimodal_alert,
                'multimodal_reason': reason_text,
            }
        )
