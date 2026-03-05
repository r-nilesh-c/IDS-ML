"""
Unit and property-based tests for HealthcareAlertSystem class.
"""

import pytest
import numpy as np
import json
import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.alert_system import HealthcareAlertSystem


class TestHealthcareAlertSystemInit:
    """Test HealthcareAlertSystem initialization."""
    
    def test_init_basic(self):
        """Test basic initialization with valid parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'log_path': os.path.join(tmpdir, 'logs', 'anomalies.jsonl'),
                'report_path': os.path.join(tmpdir, 'reports')
            }
            
            alert_system = HealthcareAlertSystem(config)
            
            assert alert_system.log_path == config['log_path']
            assert alert_system.report_path == config['report_path']
            
            # Verify directories were created
            assert os.path.exists(os.path.dirname(config['log_path']))
            assert os.path.exists(config['report_path'])
    
    def test_init_with_defaults(self):
        """Test initialization with minimal config (using defaults)."""
        config = {}
        
        alert_system = HealthcareAlertSystem(config)
        
        assert alert_system.log_path == 'logs/anomalies.jsonl'
        assert alert_system.report_path == 'reports/'


class TestHealthcareAlertSystemLogAnomaly:
    """Test HealthcareAlertSystem log_anomaly method."""
    
    def test_log_anomaly_basic(self):
        """Test basic anomaly logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'log_path': os.path.join(tmpdir, 'anomalies.jsonl'),
                'report_path': tmpdir
            }
            
            alert_system = HealthcareAlertSystem(config)
            
            # Log an anomaly
            timestamp = "2024-01-15T10:30:45Z"
            flow_features = {
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.50",
                "protocol": "TCP",
                "flow_duration": 1234
            }
            anomaly_score = 0.87
            prediction = 1
            
            alert_system.log_anomaly(timestamp, flow_features, anomaly_score, prediction)
            
            # Verify log file was created and contains entry
            assert os.path.exists(config['log_path'])
            
            with open(config['log_path'], 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                
                entry = json.loads(lines[0])
                assert entry['timestamp'] == timestamp
                assert entry['anomaly_score'] == anomaly_score
                assert entry['prediction'] == prediction
                assert entry['flow_features'] == flow_features
    
    def test_log_anomaly_multiple_entries(self):
        """Test logging multiple anomalies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'log_path': os.path.join(tmpdir, 'anomalies.jsonl'),
                'report_path': tmpdir
            }
            
            alert_system = HealthcareAlertSystem(config)
            
            # Log multiple anomalies
            for i in range(5):
                alert_system.log_anomaly(
                    f"2024-01-15T10:30:{i:02d}Z",
                    {"flow_id": i},
                    0.5 + i * 0.1,
                    1
                )
            
            # Verify all entries were logged
            with open(config['log_path'], 'r') as f:
                lines = f.readlines()
                assert len(lines) == 5
    
    def test_log_anomaly_empty_timestamp_raises_error(self):
        """Test that empty timestamp raises ValueError."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        with pytest.raises(ValueError, match="timestamp cannot be empty"):
            alert_system.log_anomaly("", {}, 0.5, 1)
    
    def test_log_anomaly_none_flow_features_raises_error(self):
        """Test that None flow_features raises ValueError."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        with pytest.raises(ValueError, match="flow_features cannot be None"):
            alert_system.log_anomaly("2024-01-15T10:30:45Z", None, 0.5, 1)
    
    def test_log_anomaly_invalid_score_raises_error(self):
        """Test that invalid anomaly_score raises ValueError."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        with pytest.raises(ValueError, match="anomaly_score must be numeric"):
            alert_system.log_anomaly("2024-01-15T10:30:45Z", {}, "invalid", 1)
    
    def test_log_anomaly_nan_score_raises_error(self):
        """Test that NaN anomaly_score raises ValueError."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        with pytest.raises(ValueError, match="anomaly_score must be finite"):
            alert_system.log_anomaly("2024-01-15T10:30:45Z", {}, np.nan, 1)
    
    def test_log_anomaly_invalid_prediction_raises_error(self):
        """Test that invalid prediction raises ValueError."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        with pytest.raises(ValueError, match="prediction must be 0 or 1"):
            alert_system.log_anomaly("2024-01-15T10:30:45Z", {}, 0.5, 2)


class TestHealthcareAlertSystemGenerateEvaluationReport:
    """Test HealthcareAlertSystem generate_evaluation_report method."""
    
    def test_generate_evaluation_report_basic(self):
        """Test basic evaluation report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'log_path': os.path.join(tmpdir, 'anomalies.jsonl'),
                'report_path': tmpdir
            }
            
            alert_system = HealthcareAlertSystem(config)
            
            # Generate synthetic data
            np.random.seed(42)
            n_samples = 100
            y_true = np.array([0] * 50 + [1] * 50)
            y_pred = y_true.copy()
            y_pred[:5] = 1 - y_pred[:5]  # Add some errors
            y_scores = np.random.rand(n_samples)
            
            # Generate report
            metrics = alert_system.generate_evaluation_report(y_true, y_pred, y_scores)
            
            # Verify required metrics are present
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            assert 'macro_f1_score' in metrics
            assert 'false_positive_rate' in metrics
            assert 'roc_auc' in metrics
            assert 'confusion_matrix' in metrics
            
            # Verify confusion matrix structure
            cm = metrics['confusion_matrix']
            assert 'true_negative' in cm
            assert 'false_positive' in cm
            assert 'false_negative' in cm
            assert 'true_positive' in cm
            
            # Verify visualization paths
            assert 'roc_curve_path' in metrics
            assert 'pr_curve_path' in metrics
            assert 'confusion_matrix_path' in metrics
            
            # Verify files were created
            assert os.path.exists(metrics['roc_curve_path'])
            assert os.path.exists(metrics['pr_curve_path'])
            assert os.path.exists(metrics['confusion_matrix_path'])
    
    def test_generate_evaluation_report_empty_y_true_raises_error(self):
        """Test that empty y_true raises ValueError."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        with pytest.raises(ValueError, match="y_true cannot be empty"):
            alert_system.generate_evaluation_report(np.array([]), np.array([1]), np.array([0.5]))
    
    def test_generate_evaluation_report_mismatched_lengths_raises_error(self):
        """Test that mismatched input lengths raise ValueError."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        with pytest.raises(ValueError, match="Input lengths must match"):
            alert_system.generate_evaluation_report(
                np.array([0, 1]),
                np.array([0, 1, 0]),
                np.array([0.5, 0.7])
            )
    
    def test_generate_evaluation_report_invalid_y_true_raises_error(self):
        """Test that y_true with values other than 0/1 raises ValueError."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        with pytest.raises(ValueError, match="y_true must contain only 0 or 1"):
            alert_system.generate_evaluation_report(
                np.array([0, 1, 2]),
                np.array([0, 1, 1]),
                np.array([0.5, 0.7, 0.9])
            )
    
    def test_generate_evaluation_report_with_attack_labels(self):
        """Test report generation with attack labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'log_path': os.path.join(tmpdir, 'anomalies.jsonl'),
                'report_path': tmpdir
            }
            
            alert_system = HealthcareAlertSystem(config)
            
            # Generate synthetic data with attack labels
            np.random.seed(42)
            y_true = np.array([0, 0, 1, 1, 1, 1])
            y_pred = np.array([0, 0, 1, 1, 0, 1])
            y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.95])
            attack_labels = ['benign', 'benign', 'DoS', 'DoS', 'PortScan', 'PortScan']
            
            # Generate report
            metrics = alert_system.generate_evaluation_report(
                y_true, y_pred, y_scores, attack_labels
            )
            
            # Verify per-class metrics are present
            assert 'per_class_metrics' in metrics
            assert len(metrics['per_class_metrics']) > 0


class TestHealthcareAlertSystemAssessDeploymentReadiness:
    """Test HealthcareAlertSystem assess_deployment_readiness method."""
    
    def test_assess_deployment_readiness_ready(self):
        """Test deployment readiness assessment when criteria are met."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        # Metrics that meet criteria
        metrics = {
            'false_positive_rate': 0.03,  # < 5%
            'recall': 0.95  # > 90%
        }
        
        assessment = alert_system.assess_deployment_readiness(metrics)
        
        assert "READY FOR DEPLOYMENT" in assessment
        assert "meets healthcare deployment criteria" in assessment
    
    def test_assess_deployment_readiness_not_ready_high_fpr(self):
        """Test deployment readiness when FPR is too high."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        # Metrics with high FPR
        metrics = {
            'false_positive_rate': 0.08,  # > 5%
            'recall': 0.95  # > 90%
        }
        
        assessment = alert_system.assess_deployment_readiness(metrics)
        
        assert "NOT READY FOR DEPLOYMENT" in assessment
        assert "False Positive Rate" in assessment
    
    def test_assess_deployment_readiness_not_ready_low_recall(self):
        """Test deployment readiness when recall is too low."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        # Metrics with low recall
        metrics = {
            'false_positive_rate': 0.03,  # < 5%
            'recall': 0.85  # < 90%
        }
        
        assessment = alert_system.assess_deployment_readiness(metrics)
        
        assert "NOT READY FOR DEPLOYMENT" in assessment
        assert "Recall" in assessment
    
    def test_assess_deployment_readiness_missing_metric_raises_error(self):
        """Test that missing required metrics raise ValueError."""
        config = {}
        alert_system = HealthcareAlertSystem(config)
        
        # Missing recall
        metrics = {'false_positive_rate': 0.03}
        
        with pytest.raises(ValueError, match="Required metric 'recall' not found"):
            alert_system.assess_deployment_readiness(metrics)


class TestHealthcareAlertSystemPropertyBased:
    """Property-based tests for HealthcareAlertSystem using Hypothesis."""
    
    def test_property_14_anomaly_logging_completeness(self):
        """
        Property 14: Anomaly Logging Completeness
        
        Validates Requirement 6.5: Anomaly logging
        
        Property: For any sample classified as anomalous (prediction = 1),
        a log entry should be created containing timestamp, flow features,
        anomaly score, and prediction.
        
        **Validates: Requirements 6.5**
        """
        from hypothesis import given, settings, strategies as st
        
        @given(
            n_anomalies=st.integers(min_value=1, max_value=20),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=50, deadline=None)
        def property_test(n_anomalies, data_seed):
            with tempfile.TemporaryDirectory() as tmpdir:
                config = {
                    'log_path': os.path.join(tmpdir, 'anomalies.jsonl'),
                    'report_path': tmpdir
                }
                
                alert_system = HealthcareAlertSystem(config)
                
                # Generate and log anomalies
                np.random.seed(data_seed)
                for i in range(n_anomalies):
                    timestamp = f"2024-01-15T10:30:{i:02d}Z"
                    flow_features = {"flow_id": i, "value": float(np.random.rand())}
                    anomaly_score = float(np.random.rand())
                    prediction = 1
                    
                    alert_system.log_anomaly(timestamp, flow_features, anomaly_score, prediction)
                
                # Verify all anomalies were logged
                with open(config['log_path'], 'r') as f:
                    lines = f.readlines()
                    assert len(lines) == n_anomalies, \
                        f"Expected {n_anomalies} log entries, got {len(lines)}"
                    
                    # Verify each entry has all required fields
                    for i, line in enumerate(lines):
                        entry = json.loads(line)
                        
                        assert 'timestamp' in entry, f"Entry {i} missing timestamp"
                        assert 'flow_features' in entry, f"Entry {i} missing flow_features"
                        assert 'anomaly_score' in entry, f"Entry {i} missing anomaly_score"
                        assert 'prediction' in entry, f"Entry {i} missing prediction"
                        
                        assert entry['prediction'] == 1, \
                            f"Entry {i} has prediction={entry['prediction']}, expected 1"
        
        # Run the property test
        property_test()
    
    def test_property_15_roc_auc_computation_correctness(self):
        """
        Property 15: ROC-AUC Computation Correctness
        
        Validates Requirements 6.7, 8.4: ROC-AUC computation
        
        Property: For any set of true labels and predicted scores,
        the computed ROC-AUC should match sklearn.metrics.roc_auc_score.
        
        **Validates: Requirements 6.7, 8.4**
        """
        from hypothesis import given, settings, strategies as st
        from sklearn.metrics import roc_auc_score
        
        @given(
            n_samples=st.integers(min_value=20, max_value=100),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=50, deadline=None)
        def property_test(n_samples, data_seed):
            with tempfile.TemporaryDirectory() as tmpdir:
                config = {
                    'log_path': os.path.join(tmpdir, 'anomalies.jsonl'),
                    'report_path': tmpdir
                }
                
                alert_system = HealthcareAlertSystem(config)
                
                # Generate synthetic data with both classes
                np.random.seed(data_seed)
                n_positive = n_samples // 2
                n_negative = n_samples - n_positive
                
                y_true = np.array([0] * n_negative + [1] * n_positive)
                y_scores = np.random.rand(n_samples).astype(np.float64)
                y_pred = (y_scores > 0.5).astype(int)
                
                # Generate report
                metrics = alert_system.generate_evaluation_report(y_true, y_pred, y_scores)
                
                # Compute expected ROC-AUC using sklearn
                expected_roc_auc = roc_auc_score(y_true, y_scores)
                
                # Property assertion
                assert metrics['roc_auc'] is not None, "ROC-AUC should not be None"
                assert np.isclose(metrics['roc_auc'], expected_roc_auc, rtol=1e-10, atol=1e-12), \
                    f"ROC-AUC mismatch: got {metrics['roc_auc']}, expected {expected_roc_auc}"
        
        # Run the property test
        property_test()
    
    def test_property_16_confusion_matrix_correctness(self):
        """
        Property 16: Confusion Matrix Correctness
        
        Validates Requirement 6.9: Confusion matrix computation
        
        Property: For any set of true labels and predictions,
        the confusion matrix should correctly count TP, TN, FP, FN.
        
        **Validates: Requirements 6.9**
        """
        from hypothesis import given, settings, strategies as st
        
        @given(
            n_samples=st.integers(min_value=20, max_value=100),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=50, deadline=None)
        def property_test(n_samples, data_seed):
            with tempfile.TemporaryDirectory() as tmpdir:
                config = {
                    'log_path': os.path.join(tmpdir, 'anomalies.jsonl'),
                    'report_path': tmpdir
                }
                
                alert_system = HealthcareAlertSystem(config)
                
                # Generate synthetic data
                np.random.seed(data_seed)
                y_true = np.random.randint(0, 2, size=n_samples)
                y_pred = np.random.randint(0, 2, size=n_samples)
                y_scores = np.random.rand(n_samples).astype(np.float64)
                
                # Generate report
                metrics = alert_system.generate_evaluation_report(y_true, y_pred, y_scores)
                
                # Manually compute confusion matrix counts
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                tp = np.sum((y_true == 1) & (y_pred == 1))
                
                # Property assertions
                cm = metrics['confusion_matrix']
                assert cm['true_negative'] == tn, \
                    f"TN mismatch: got {cm['true_negative']}, expected {tn}"
                assert cm['false_positive'] == fp, \
                    f"FP mismatch: got {cm['false_positive']}, expected {fp}"
                assert cm['false_negative'] == fn, \
                    f"FN mismatch: got {cm['false_negative']}, expected {fn}"
                assert cm['true_positive'] == tp, \
                    f"TP mismatch: got {cm['true_positive']}, expected {tp}"
                
                # Verify total equals n_samples
                total = tn + fp + fn + tp
                assert total == n_samples, \
                    f"Confusion matrix total {total} doesn't match n_samples {n_samples}"
        
        # Run the property test
        property_test()
    
    def test_property_18_accuracy_computation_correctness(self):
        """
        Property 18: Accuracy Computation Correctness
        
        Validates Requirement 8.1: Accuracy computation
        
        Property: For any set of true labels and predictions,
        accuracy should equal (TP + TN) / (TP + TN + FP + FN).
        
        **Validates: Requirements 8.1**
        """
        from hypothesis import given, settings, strategies as st
        
        @given(
            n_samples=st.integers(min_value=20, max_value=100),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=50, deadline=None)
        def property_test(n_samples, data_seed):
            with tempfile.TemporaryDirectory() as tmpdir:
                config = {
                    'log_path': os.path.join(tmpdir, 'anomalies.jsonl'),
                    'report_path': tmpdir
                }
                
                alert_system = HealthcareAlertSystem(config)
                
                # Generate synthetic data
                np.random.seed(data_seed)
                y_true = np.random.randint(0, 2, size=n_samples)
                y_pred = np.random.randint(0, 2, size=n_samples)
                y_scores = np.random.rand(n_samples).astype(np.float64)
                
                # Generate report
                metrics = alert_system.generate_evaluation_report(y_true, y_pred, y_scores)
                
                # Manually compute accuracy
                cm = metrics['confusion_matrix']
                tn = cm['true_negative']
                fp = cm['false_positive']
                fn = cm['false_negative']
                tp = cm['true_positive']
                
                expected_accuracy = (tp + tn) / (tp + tn + fp + fn)
                
                # Property assertion
                assert np.isclose(metrics['accuracy'], expected_accuracy, rtol=1e-10, atol=1e-12), \
                    f"Accuracy mismatch: got {metrics['accuracy']}, expected {expected_accuracy}"
        
        # Run the property test
        property_test()
    
    def test_property_19_macro_f1_score_computation_correctness(self):
        """
        Property 19: Macro F1-Score Computation Correctness
        
        Validates Requirement 8.2: Macro F1-score computation
        
        Property: For any set of true labels and predictions,
        macro F1-score should equal the unweighted mean of per-class F1-scores.
        
        **Validates: Requirements 8.2**
        """
        from hypothesis import given, settings, strategies as st
        from sklearn.metrics import f1_score
        
        @given(
            n_samples=st.integers(min_value=20, max_value=100),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=50, deadline=None)
        def property_test(n_samples, data_seed):
            with tempfile.TemporaryDirectory() as tmpdir:
                config = {
                    'log_path': os.path.join(tmpdir, 'anomalies.jsonl'),
                    'report_path': tmpdir
                }
                
                alert_system = HealthcareAlertSystem(config)
                
                # Generate synthetic data
                np.random.seed(data_seed)
                y_true = np.random.randint(0, 2, size=n_samples)
                y_pred = np.random.randint(0, 2, size=n_samples)
                y_scores = np.random.rand(n_samples).astype(np.float64)
                
                # Generate report
                metrics = alert_system.generate_evaluation_report(y_true, y_pred, y_scores)
                
                # Compute expected macro F1-score using sklearn
                expected_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                
                # Property assertion
                assert np.isclose(metrics['macro_f1_score'], expected_macro_f1, rtol=1e-10, atol=1e-12), \
                    f"Macro F1-score mismatch: got {metrics['macro_f1_score']}, expected {expected_macro_f1}"
        
        # Run the property test
        property_test()
    
    def test_property_20_false_positive_rate_computation_correctness(self):
        """
        Property 20: False Positive Rate Computation Correctness
        
        Validates Requirement 8.3: FPR computation
        
        Property: For any set of true labels and predictions,
        FPR should equal FP / (FP + TN) for the benign class.
        
        **Validates: Requirements 8.3**
        """
        from hypothesis import given, settings, strategies as st
        
        @given(
            n_samples=st.integers(min_value=20, max_value=100),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=50, deadline=None)
        def property_test(n_samples, data_seed):
            with tempfile.TemporaryDirectory() as tmpdir:
                config = {
                    'log_path': os.path.join(tmpdir, 'anomalies.jsonl'),
                    'report_path': tmpdir
                }
                
                alert_system = HealthcareAlertSystem(config)
                
                # Generate synthetic data
                np.random.seed(data_seed)
                y_true = np.random.randint(0, 2, size=n_samples)
                y_pred = np.random.randint(0, 2, size=n_samples)
                y_scores = np.random.rand(n_samples).astype(np.float64)
                
                # Generate report
                metrics = alert_system.generate_evaluation_report(y_true, y_pred, y_scores)
                
                # Manually compute FPR
                cm = metrics['confusion_matrix']
                fp = cm['false_positive']
                tn = cm['true_negative']
                
                if (fp + tn) > 0:
                    expected_fpr = fp / (fp + tn)
                else:
                    expected_fpr = 0.0
                
                # Property assertion
                assert np.isclose(metrics['false_positive_rate'], expected_fpr, rtol=1e-10, atol=1e-12), \
                    f"FPR mismatch: got {metrics['false_positive_rate']}, expected {expected_fpr}"
        
        # Run the property test
        property_test()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
