"""
Unit tests for IsolationForestDetector class.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.isolation_forest import IsolationForestDetector


class TestIsolationForestDetectorInit:
    """Test IsolationForestDetector initialization (Task 5.1)."""
    
    def test_init_basic(self):
        """Test basic initialization with valid parameters."""
        config = {
            'n_estimators': 100,
            'max_samples': 256,
            'contamination': 'auto',
            'random_state': 42,
            'n_jobs': -1
        }
        
        detector = IsolationForestDetector(config)
        
        assert detector.n_estimators == 100
        assert detector.max_samples == 256
        assert detector.contamination == 'auto'
        assert detector.random_state == 42
        assert detector.n_jobs == -1
        assert detector.model is None  # Not trained yet
    
    def test_init_with_defaults(self):
        """Test initialization with minimal config (using defaults)."""
        config = {}
        
        detector = IsolationForestDetector(config)
        
        assert detector.n_estimators == 100  # default
        assert detector.max_samples == 256  # default
        assert detector.contamination == 'auto'  # default
        assert detector.random_state == 42  # default
        assert detector.n_jobs == -1  # default
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        config = {
            'n_estimators': 200,
            'max_samples': 512,
            'contamination': 0.1,
            'random_state': 123,
            'n_jobs': 4
        }
        
        detector = IsolationForestDetector(config)
        
        assert detector.n_estimators == 200
        assert detector.max_samples == 512
        assert detector.contamination == 0.1
        assert detector.random_state == 123
        assert detector.n_jobs == 4
    
    def test_init_max_samples_auto(self):
        """Test initialization with max_samples='auto'."""
        config = {
            'max_samples': 'auto',
            'random_state': 42
        }
        
        detector = IsolationForestDetector(config)
        assert detector.max_samples == 'auto'
    
    def test_init_invalid_n_estimators(self):
        """Test that invalid n_estimators raises ValueError."""
        config = {'n_estimators': 0}
        
        with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
            IsolationForestDetector(config)
        
        config = {'n_estimators': -10}
        with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
            IsolationForestDetector(config)
        
        config = {'n_estimators': 'invalid'}
        with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
            IsolationForestDetector(config)
    
    def test_init_invalid_max_samples(self):
        """Test that invalid max_samples raises ValueError."""
        config = {'max_samples': 0}
        
        with pytest.raises(ValueError, match="max_samples must be 'auto' or a positive integer"):
            IsolationForestDetector(config)
        
        config = {'max_samples': -100}
        with pytest.raises(ValueError, match="max_samples must be 'auto' or a positive integer"):
            IsolationForestDetector(config)
    
    def test_init_invalid_contamination(self):
        """Test that invalid contamination raises ValueError."""
        config = {'contamination': 0.6}  # Must be < 0.5
        
        with pytest.raises(ValueError, match="contamination must be in range"):
            IsolationForestDetector(config)
        
        config = {'contamination': 0}  # Must be > 0
        with pytest.raises(ValueError, match="contamination must be in range"):
            IsolationForestDetector(config)
        
        config = {'contamination': 'invalid'}
        with pytest.raises(ValueError, match="contamination must be 'auto' or a number"):
            IsolationForestDetector(config)
    
    def test_init_invalid_random_state(self):
        """Test that invalid random_state raises ValueError."""
        config = {'random_state': -1}
        
        with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
            IsolationForestDetector(config)
        
        config = {'random_state': 'invalid'}
        with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
            IsolationForestDetector(config)
    
    def test_init_invalid_n_jobs(self):
        """Test that invalid n_jobs raises ValueError."""
        config = {'n_jobs': 'invalid'}
        
        with pytest.raises(ValueError, match="n_jobs must be an integer"):
            IsolationForestDetector(config)


class TestIsolationForestDetectorTraining:
    """Test IsolationForestDetector training method (Task 5.2)."""
    
    def test_train_basic(self):
        """Test basic training with valid benign data."""
        config = {
            'n_estimators': 50,
            'max_samples': 100,
            'random_state': 42,
            'n_jobs': 1
        }
        
        detector = IsolationForestDetector(config)
        
        # Generate synthetic benign data
        np.random.seed(42)
        X_train = np.random.normal(0.5, 0.1, size=(200, 20))
        
        # Train
        detector.train(X_train)
        
        # Verify model is trained
        assert detector.model is not None
        assert hasattr(detector.model, 'estimators_')
        assert detector.model.n_features_in_ == 20
    
    def test_train_validates_benign_only(self):
        """Test that training logs indicate benign-only training."""
        config = {'random_state': 42, 'n_jobs': 1}
        
        detector = IsolationForestDetector(config)
        
        # Generate benign data
        np.random.seed(42)
        X_train = np.random.rand(100, 15)
        
        # Train (should complete without error)
        detector.train(X_train)
        
        assert detector.model is not None
    
    def test_train_empty_data_raises_error(self):
        """Test that empty training data raises ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        with pytest.raises(ValueError, match="X_train cannot be empty"):
            detector.train(np.array([]))
    
    def test_train_none_data_raises_error(self):
        """Test that None training data raises ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        with pytest.raises(ValueError, match="X_train cannot be empty"):
            detector.train(None)
    
    def test_train_wrong_dimensions_raises_error(self):
        """Test that wrong input dimensions raise ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        # 1D array
        X_train_1d = np.random.rand(100)
        with pytest.raises(ValueError, match="X_train must be 2D array"):
            detector.train(X_train_1d)
        
        # 3D array
        X_train_3d = np.random.rand(10, 20, 5)
        with pytest.raises(ValueError, match="X_train must be 2D array"):
            detector.train(X_train_3d)
    
    def test_train_nan_values_raises_error(self):
        """Test that NaN values in training data raise ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        X_train = np.random.rand(100, 20)
        X_train[10, 5] = np.nan
        
        with pytest.raises(ValueError, match="X_train contains NaN or infinite values"):
            detector.train(X_train)
    
    def test_train_inf_values_raises_error(self):
        """Test that infinite values in training data raise ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        X_train = np.random.rand(100, 20)
        X_train[10, 5] = np.inf
        
        with pytest.raises(ValueError, match="X_train contains NaN or infinite values"):
            detector.train(X_train)
    
    def test_train_insufficient_samples_raises_error(self):
        """Test that insufficient samples raise ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        # Only 1 sample
        X_train = np.random.rand(1, 20)
        
        with pytest.raises(ValueError, match="X_train must have at least 2 samples"):
            detector.train(X_train)
    
    def test_train_various_feature_dimensions(self):
        """Test training with various feature dimensions."""
        config = {'random_state': 42, 'n_jobs': 1}
        
        for n_features in [5, 10, 20, 50, 100]:
            detector = IsolationForestDetector(config)
            X_train = np.random.rand(200, n_features)
            
            detector.train(X_train)
            
            assert detector.model is not None
            assert detector.model.n_features_in_ == n_features
    
    def test_train_various_sample_sizes(self):
        """Test training with various sample sizes."""
        config = {'random_state': 42, 'n_jobs': 1}
        
        for n_samples in [10, 50, 100, 500, 1000]:
            detector = IsolationForestDetector(config)
            X_train = np.random.rand(n_samples, 20)
            
            detector.train(X_train)
            
            assert detector.model is not None
    
    def test_train_reproducibility_with_seed(self):
        """Test that same seed produces reproducible training."""
        config = {'random_state': 42, 'n_jobs': 1}
        
        # Generate training data
        np.random.seed(42)
        X_train = np.random.rand(200, 20)
        
        # Train first detector
        detector1 = IsolationForestDetector(config)
        detector1.train(X_train)
        scores1 = detector1.compute_anomaly_score(X_train[:10])
        
        # Train second detector with same seed
        detector2 = IsolationForestDetector(config)
        detector2.train(X_train)
        scores2 = detector2.compute_anomaly_score(X_train[:10])
        
        # Scores should be identical
        np.testing.assert_array_almost_equal(scores1, scores2, decimal=10)


class TestIsolationForestDetectorAnomalyScore:
    """Test IsolationForestDetector anomaly score computation (Task 5.3)."""
    
    def test_compute_anomaly_score_basic(self):
        """Test basic anomaly score computation."""
        config = {'random_state': 42, 'n_jobs': 1}
        
        detector = IsolationForestDetector(config)
        
        # Train on benign data
        np.random.seed(42)
        X_train = np.random.normal(0.5, 0.1, size=(200, 20))
        detector.train(X_train)
        
        # Compute scores on test data
        X_test = np.random.normal(0.5, 0.1, size=(50, 20))
        scores = detector.compute_anomaly_score(X_test)
        
        # Verify output shape and type
        assert scores.shape == (50,)
        assert scores.dtype in [np.float32, np.float64]
    
    def test_compute_anomaly_score_higher_for_anomalies(self):
        """Test that anomaly scores are higher for anomalous samples."""
        config = {'random_state': 42, 'n_jobs': 1}
        
        detector = IsolationForestDetector(config)
        
        # Train on benign data (centered around 0.5)
        np.random.seed(42)
        X_train = np.random.normal(0.5, 0.1, size=(500, 20))
        detector.train(X_train)
        
        # Test on benign data (similar distribution)
        X_benign = np.random.normal(0.5, 0.1, size=(100, 20))
        scores_benign = detector.compute_anomaly_score(X_benign)
        
        # Test on anomalous data (very different distribution)
        X_anomaly = np.random.normal(5.0, 2.0, size=(100, 20))
        scores_anomaly = detector.compute_anomaly_score(X_anomaly)
        
        # Anomalous samples should have higher average scores
        assert np.mean(scores_anomaly) > np.mean(scores_benign)
    
    def test_compute_anomaly_score_negates_decision_function(self):
        """Test that scores are negated decision_function values."""
        config = {'random_state': 42, 'n_jobs': 1}
        
        detector = IsolationForestDetector(config)
        
        # Train
        np.random.seed(42)
        X_train = np.random.rand(200, 20)
        detector.train(X_train)
        
        # Compute scores
        X_test = np.random.rand(50, 20)
        anomaly_scores = detector.compute_anomaly_score(X_test)
        
        # Get decision function directly
        decision_scores = detector.model.decision_function(X_test)
        
        # Verify negation
        np.testing.assert_array_almost_equal(anomaly_scores, -decision_scores)
    
    def test_compute_anomaly_score_without_training_raises_error(self):
        """Test that computing scores without training raises ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        X_test = np.random.rand(50, 20)
        
        with pytest.raises(ValueError, match="Model must be trained before computing anomaly scores"):
            detector.compute_anomaly_score(X_test)
    
    def test_compute_anomaly_score_empty_data_raises_error(self):
        """Test that empty test data raises ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        # Train first
        X_train = np.random.rand(100, 20)
        detector.train(X_train)
        
        # Try with empty data
        with pytest.raises(ValueError, match="X cannot be empty"):
            detector.compute_anomaly_score(np.array([]))
    
    def test_compute_anomaly_score_none_data_raises_error(self):
        """Test that None test data raises ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        # Train first
        X_train = np.random.rand(100, 20)
        detector.train(X_train)
        
        # Try with None
        with pytest.raises(ValueError, match="X cannot be empty"):
            detector.compute_anomaly_score(None)
    
    def test_compute_anomaly_score_wrong_dimensions_raises_error(self):
        """Test that wrong input dimensions raise ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        # Train
        X_train = np.random.rand(100, 20)
        detector.train(X_train)
        
        # 1D array
        X_test_1d = np.random.rand(50)
        with pytest.raises(ValueError, match="X must be 2D array"):
            detector.compute_anomaly_score(X_test_1d)
        
        # 3D array
        X_test_3d = np.random.rand(10, 20, 5)
        with pytest.raises(ValueError, match="X must be 2D array"):
            detector.compute_anomaly_score(X_test_3d)
    
    def test_compute_anomaly_score_nan_values_raises_error(self):
        """Test that NaN values in test data raise ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        # Train
        X_train = np.random.rand(100, 20)
        detector.train(X_train)
        
        # Test with NaN
        X_test = np.random.rand(50, 20)
        X_test[10, 5] = np.nan
        
        with pytest.raises(ValueError, match="X contains NaN or infinite values"):
            detector.compute_anomaly_score(X_test)
    
    def test_compute_anomaly_score_inf_values_raises_error(self):
        """Test that infinite values in test data raise ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        # Train
        X_train = np.random.rand(100, 20)
        detector.train(X_train)
        
        # Test with inf
        X_test = np.random.rand(50, 20)
        X_test[10, 5] = np.inf
        
        with pytest.raises(ValueError, match="X contains NaN or infinite values"):
            detector.compute_anomaly_score(X_test)
    
    def test_compute_anomaly_score_wrong_feature_dimension_raises_error(self):
        """Test that mismatched feature dimensions raise ValueError."""
        config = {'random_state': 42}
        detector = IsolationForestDetector(config)
        
        # Train with 20 features
        X_train = np.random.rand(100, 20)
        detector.train(X_train)
        
        # Test with 15 features
        X_test = np.random.rand(50, 15)
        
        with pytest.raises(ValueError, match="X has 15 features, but model was trained with 20 features"):
            detector.compute_anomaly_score(X_test)
    
    def test_compute_anomaly_score_various_batch_sizes(self):
        """Test anomaly score computation with various batch sizes."""
        config = {'random_state': 42, 'n_jobs': 1}
        
        detector = IsolationForestDetector(config)
        
        # Train
        np.random.seed(42)
        X_train = np.random.rand(200, 20)
        detector.train(X_train)
        
        # Test with various batch sizes
        for n_samples in [1, 10, 50, 100, 500, 1000]:
            X_test = np.random.rand(n_samples, 20)
            scores = detector.compute_anomaly_score(X_test)
            
            assert scores.shape == (n_samples,)
            assert np.all(np.isfinite(scores))
    
    def test_compute_anomaly_score_returns_array(self):
        """Test that compute_anomaly_score returns numpy array."""
        config = {'random_state': 42, 'n_jobs': 1}
        
        detector = IsolationForestDetector(config)
        
        # Train
        X_train = np.random.rand(100, 20)
        detector.train(X_train)
        
        # Compute scores
        X_test = np.random.rand(50, 20)
        scores = detector.compute_anomaly_score(X_test)
        
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1


class TestIsolationForestDetectorIntegration:
    """Integration tests for IsolationForestDetector."""
    
    def test_full_pipeline(self):
        """Test complete pipeline: init -> train -> score."""
        config = {
            'n_estimators': 100,
            'max_samples': 256,
            'contamination': 'auto',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Initialize
        detector = IsolationForestDetector(config)
        assert detector.model is None
        
        # Train on benign data
        np.random.seed(42)
        X_train = np.random.normal(0.5, 0.1, size=(500, 30))
        detector.train(X_train)
        assert detector.model is not None
        
        # Compute scores on benign test data
        X_test_benign = np.random.normal(0.5, 0.1, size=(100, 30))
        scores_benign = detector.compute_anomaly_score(X_test_benign)
        assert scores_benign.shape == (100,)
        
        # Compute scores on anomalous test data
        X_test_anomaly = np.random.normal(5.0, 2.0, size=(100, 30))
        scores_anomaly = detector.compute_anomaly_score(X_test_anomaly)
        assert scores_anomaly.shape == (100,)
        
        # Anomalies should have higher scores
        assert np.mean(scores_anomaly) > np.mean(scores_benign)
    
    def test_parallel_processing(self):
        """Test that n_jobs=-1 enables parallel processing."""
        config = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1  # Use all cores
        }
        
        detector = IsolationForestDetector(config)
        
        # Train on larger dataset
        np.random.seed(42)
        X_train = np.random.rand(1000, 50)
        detector.train(X_train)
        
        # Compute scores
        X_test = np.random.rand(500, 50)
        scores = detector.compute_anomaly_score(X_test)
        
        assert scores.shape == (500,)
        assert np.all(np.isfinite(scores))
    
    def test_configuration_persistence(self):
        """Test that configuration is preserved throughout lifecycle."""
        config = {
            'n_estimators': 150,
            'max_samples': 512,
            'contamination': 0.05,
            'random_state': 123,
            'n_jobs': 4
        }
        
        detector = IsolationForestDetector(config)
        
        # Verify config before training
        assert detector.n_estimators == 150
        assert detector.max_samples == 512
        assert detector.contamination == 0.05
        assert detector.random_state == 123
        assert detector.n_jobs == 4
        
        # Train
        X_train = np.random.rand(200, 20)
        detector.train(X_train)
        
        # Verify config after training
        assert detector.n_estimators == 150
        assert detector.max_samples == 512
        assert detector.contamination == 0.05
        assert detector.random_state == 123
        assert detector.n_jobs == 4
        
        # Compute scores
        X_test = np.random.rand(50, 20)
        detector.compute_anomaly_score(X_test)
        
        # Verify config after scoring
        assert detector.n_estimators == 150
        assert detector.max_samples == 512


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



class TestIsolationForestDetectorPropertyBased:
    """Property-based tests for IsolationForestDetector using Hypothesis."""
    
    def test_property_9_anomaly_score_output_scalar(self):
        """
        Property 9: Isolation Forest Anomaly Score Output
        
        Validates Requirement 4.4: Output anomaly score per sample
        
        Property: For any valid input data, anomaly scores must be:
        - Scalar value per sample (1D array with length = n_samples)
        - Finite (not NaN or inf)
        - Numeric (float type)
        
        **Validates: Requirements 4.4**
        """
        from hypothesis import given, settings, strategies as st
        
        @given(
            n_samples=st.integers(min_value=10, max_value=200),
            n_features=st.integers(min_value=5, max_value=50),
            n_estimators=st.integers(min_value=10, max_value=100),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=100, deadline=None)
        def property_test(n_samples, n_features, n_estimators, data_seed):
            # Configure detector
            config = {
                'n_estimators': n_estimators,
                'max_samples': min(256, n_samples),
                'contamination': 'auto',
                'random_state': 42,
                'n_jobs': 1  # Single thread for property test
            }
            
            # Create detector
            detector = IsolationForestDetector(config)
            
            # Generate random training data
            np.random.seed(data_seed)
            X_train = np.random.rand(n_samples, n_features).astype(np.float32)
            
            # Train model
            detector.train(X_train)
            
            # Generate test data
            X_test = np.random.rand(n_samples, n_features).astype(np.float32)
            
            # Compute anomaly scores
            scores = detector.compute_anomaly_score(X_test)
            
            # Property assertions
            assert scores.shape == (n_samples,), \
                f"Expected shape ({n_samples},), got {scores.shape}"
            
            assert scores.ndim == 1, \
                f"Anomaly scores must be 1D array (scalar per sample), got {scores.ndim}D"
            
            assert np.all(np.isfinite(scores)), \
                f"Anomaly scores must be finite, found NaN or inf"
            
            assert scores.dtype in [np.float32, np.float64], \
                f"Anomaly scores must be floating point, got {scores.dtype}"
            
            # Verify each element is a scalar (not an array)
            for i, score in enumerate(scores):
                assert np.isscalar(score) or (isinstance(score, np.ndarray) and score.shape == ()), \
                    f"Score at index {i} is not a scalar: {score}"
        
        # Run the property test
        property_test()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
