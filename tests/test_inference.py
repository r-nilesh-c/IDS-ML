"""
Unit tests for inference pipelines (batch and streaming).

Tests cover:
- Batch inference with various batch sizes
- Streaming inference latency
- Error handling for invalid inputs
- Model loading and initialization
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import pickle
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import PreprocessingPipeline
from src.autoencoder import AutoencoderDetector
from src.isolation_forest import IsolationForestDetector
from src.fusion import FusionModule


class TestBatchInference:
    """Test batch inference functionality."""
    
    @pytest.fixture
    def setup_models(self):
        """Create and train minimal models for testing."""
        n_samples = 100
        n_features = 10
        
        # Generate synthetic benign data
        np.random.seed(42)
        X_train = np.random.normal(0.5, 0.1, size=(n_samples, n_features))
        X_train = np.clip(X_train, 0, 1)
        
        X_val = np.random.normal(0.5, 0.1, size=(20, n_features))
        X_val = np.clip(X_val, 0, 1)
        
        # Create and train autoencoder
        ae_config = {
            'encoding_dim': 5,
            'learning_rate': 0.001,
            'epochs': 2,
            'batch_size': 32,
            'early_stopping_patience': 1,
            'use_gpu': False,
            'mixed_precision': False,
            'random_state': 42
        }
        
        autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
        autoencoder.build_model(use_dropout=False)
        autoencoder.train(X_train, X_val)
        
        # Create and train isolation forest
        if_config = {
            'n_estimators': 10,
            'max_samples': 50,
            'contamination': 'auto',
            'random_state': 42,
            'n_jobs': 1
        }
        
        isolation_forest = IsolationForestDetector(if_config)
        isolation_forest.train(X_train)
        
        # Create and fit fusion module
        fusion_config = {
            'weight_autoencoder': 0.5,
            'weight_isolation': 0.5,
            'percentile': 95
        }
        
        fusion = FusionModule(fusion_config)
        
        # Compute validation scores and fit threshold
        recon_errors_val = autoencoder.compute_reconstruction_error(X_val)
        iso_scores_val = isolation_forest.compute_anomaly_score(X_val)
        fusion.fit_threshold(recon_errors_val, iso_scores_val)
        
        return {
            'autoencoder': autoencoder,
            'isolation_forest': isolation_forest,
            'fusion': fusion,
            'n_features': n_features
        }
    
    def test_batch_inference_single_sample(self, setup_models):
        """Test batch inference with a single sample."""
        models = setup_models
        
        # Create test sample
        X_test = np.random.normal(0.5, 0.1, size=(1, models['n_features']))
        X_test = np.clip(X_test, 0, 1)
        
        # Run inference
        recon_errors = models['autoencoder'].compute_reconstruction_error(X_test)
        iso_scores = models['isolation_forest'].compute_anomaly_score(X_test)
        
        recon_norm, iso_norm = models['fusion'].normalize_scores(recon_errors, iso_scores)
        combined_scores = models['fusion'].compute_combined_score(recon_norm, iso_norm)
        predictions = models['fusion'].classify(combined_scores)
        
        # Verify outputs
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]
        assert len(combined_scores) == 1
        assert 0 <= combined_scores[0] <= 1
    
    def test_batch_inference_multiple_samples(self, setup_models):
        """Test batch inference with multiple samples."""
        models = setup_models
        
        # Create test samples
        batch_sizes = [10, 50, 100]
        
        for batch_size in batch_sizes:
            X_test = np.random.normal(0.5, 0.1, size=(batch_size, models['n_features']))
            X_test = np.clip(X_test, 0, 1)
            
            # Run inference
            recon_errors = models['autoencoder'].compute_reconstruction_error(X_test)
            iso_scores = models['isolation_forest'].compute_anomaly_score(X_test)
            
            recon_norm, iso_norm = models['fusion'].normalize_scores(recon_errors, iso_scores)
            combined_scores = models['fusion'].compute_combined_score(recon_norm, iso_norm)
            predictions = models['fusion'].classify(combined_scores)
            
            # Verify outputs
            assert len(predictions) == batch_size
            assert all(p in [0, 1] for p in predictions)
            assert len(combined_scores) == batch_size
            assert all(0 <= s <= 1 for s in combined_scores)
    
    def test_batch_inference_with_anomalies(self, setup_models):
        """Test batch inference with anomalous samples."""
        models = setup_models
        
        # Create benign samples
        X_benign = np.random.normal(0.5, 0.1, size=(50, models['n_features']))
        X_benign = np.clip(X_benign, 0, 1)
        
        # Create anomalous samples (different distribution)
        X_anomaly = np.random.uniform(0, 1, size=(50, models['n_features']))
        
        # Combine
        X_test = np.vstack([X_benign, X_anomaly])
        
        # Run inference
        recon_errors = models['autoencoder'].compute_reconstruction_error(X_test)
        iso_scores = models['isolation_forest'].compute_anomaly_score(X_test)
        
        recon_norm, iso_norm = models['fusion'].normalize_scores(recon_errors, iso_scores)
        combined_scores = models['fusion'].compute_combined_score(recon_norm, iso_norm)
        predictions = models['fusion'].classify(combined_scores)
        
        # Verify that some anomalies are detected
        assert np.sum(predictions) > 0, "Should detect at least some anomalies"
        assert len(predictions) == 100
    
    def test_batch_inference_latency(self, setup_models):
        """Test that batch inference meets latency requirements."""
        import time
        
        models = setup_models
        
        # Create test samples
        X_test = np.random.normal(0.5, 0.1, size=(100, models['n_features']))
        X_test = np.clip(X_test, 0, 1)
        
        # Measure inference time
        start_time = time.time()
        
        recon_errors = models['autoencoder'].compute_reconstruction_error(X_test)
        iso_scores = models['isolation_forest'].compute_anomaly_score(X_test)
        recon_norm, iso_norm = models['fusion'].normalize_scores(recon_errors, iso_scores)
        combined_scores = models['fusion'].compute_combined_score(recon_norm, iso_norm)
        predictions = models['fusion'].classify(combined_scores)
        
        end_time = time.time()
        
        # Calculate per-sample latency
        total_time_ms = (end_time - start_time) * 1000
        per_sample_latency = total_time_ms / len(X_test)
        
        # Verify latency is reasonable (should be well under 100ms per sample)
        assert per_sample_latency < 100, f"Per-sample latency {per_sample_latency:.2f}ms exceeds 100ms"
    
    def test_model_save_and_load(self, setup_models, tmp_path):
        """Test saving and loading models for inference."""
        models = setup_models
        
        # Save models
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        
        # Save autoencoder
        ae_path = model_dir / "autoencoder_best.keras"
        models['autoencoder'].model.save(ae_path)
        
        # Save isolation forest
        if_path = model_dir / "isolation_forest.pkl"
        with open(if_path, 'wb') as f:
            pickle.dump(models['isolation_forest'].model, f)
        
        # Save fusion parameters
        fusion_params = {
            'weight_autoencoder': models['fusion'].weight_autoencoder,
            'weight_isolation': models['fusion'].weight_isolation,
            'percentile': models['fusion'].percentile,
            'recon_min': models['fusion'].recon_min,
            'recon_max': models['fusion'].recon_max,
            'iso_min': models['fusion'].iso_min,
            'iso_max': models['fusion'].iso_max,
            'threshold': models['fusion'].threshold
        }
        
        fusion_path = model_dir / "fusion_params.pkl"
        with open(fusion_path, 'wb') as f:
            pickle.dump(fusion_params, f)
        
        # Load models
        import tensorflow as tf
        
        ae_config = {'encoding_dim': 5, 'learning_rate': 0.001, 'epochs': 2,
                    'batch_size': 32, 'use_gpu': False, 'random_state': 42}
        autoencoder_loaded = AutoencoderDetector(input_dim=models['n_features'], config=ae_config)
        autoencoder_loaded.model = tf.keras.models.load_model(ae_path)
        
        with open(if_path, 'rb') as f:
            if_model_loaded = pickle.load(f)
        
        if_config = {'n_estimators': 10, 'random_state': 42}
        isolation_forest_loaded = IsolationForestDetector(if_config)
        isolation_forest_loaded.model = if_model_loaded
        
        with open(fusion_path, 'rb') as f:
            fusion_params_loaded = pickle.load(f)
        
        fusion_config = {'weight_autoencoder': 0.5, 'weight_isolation': 0.5, 'percentile': 95}
        fusion_loaded = FusionModule(fusion_config)
        fusion_loaded.recon_min = fusion_params_loaded['recon_min']
        fusion_loaded.recon_max = fusion_params_loaded['recon_max']
        fusion_loaded.iso_min = fusion_params_loaded['iso_min']
        fusion_loaded.iso_max = fusion_params_loaded['iso_max']
        fusion_loaded.threshold = fusion_params_loaded['threshold']
        
        # Test inference with loaded models
        X_test = np.random.normal(0.5, 0.1, size=(10, models['n_features']))
        X_test = np.clip(X_test, 0, 1)
        
        # Original models
        recon_orig = models['autoencoder'].compute_reconstruction_error(X_test)
        iso_orig = models['isolation_forest'].compute_anomaly_score(X_test)
        
        # Loaded models
        recon_loaded = autoencoder_loaded.compute_reconstruction_error(X_test)
        iso_loaded = isolation_forest_loaded.compute_anomaly_score(X_test)
        
        # Verify outputs are similar (may not be exactly equal due to floating point)
        np.testing.assert_allclose(recon_orig, recon_loaded, rtol=1e-5)
        np.testing.assert_allclose(iso_orig, iso_loaded, rtol=1e-5)


class TestStreamingInference:
    """Test streaming inference functionality."""
    
    @pytest.fixture
    def setup_streaming_engine(self):
        """Create streaming inference engine for testing."""
        n_samples = 100
        n_features = 10
        
        # Generate synthetic benign data
        np.random.seed(42)
        X_train = np.random.normal(0.5, 0.1, size=(n_samples, n_features))
        X_train = np.clip(X_train, 0, 1)
        
        X_val = np.random.normal(0.5, 0.1, size=(20, n_features))
        X_val = np.clip(X_val, 0, 1)
        
        # Create and train autoencoder
        ae_config = {
            'encoding_dim': 5,
            'learning_rate': 0.001,
            'epochs': 2,
            'batch_size': 32,
            'early_stopping_patience': 1,
            'use_gpu': False,
            'mixed_precision': False,
            'random_state': 42
        }
        
        autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
        autoencoder.build_model(use_dropout=False)
        autoencoder.train(X_train, X_val)
        
        # Create and train isolation forest
        if_config = {
            'n_estimators': 10,
            'max_samples': 50,
            'contamination': 'auto',
            'random_state': 42,
            'n_jobs': 1
        }
        
        isolation_forest = IsolationForestDetector(if_config)
        isolation_forest.train(X_train)
        
        # Create and fit fusion module
        fusion_config = {
            'weight_autoencoder': 0.5,
            'weight_isolation': 0.5,
            'percentile': 95
        }
        
        fusion = FusionModule(fusion_config)
        
        # Compute validation scores and fit threshold
        recon_errors_val = autoencoder.compute_reconstruction_error(X_val)
        iso_scores_val = isolation_forest.compute_anomaly_score(X_val)
        fusion.fit_threshold(recon_errors_val, iso_scores_val)
        
        # Import StreamingInferenceEngine
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from stream_inference import StreamingInferenceEngine
        
        engine = StreamingInferenceEngine(autoencoder, isolation_forest, fusion, mini_batch_size=1)
        
        return {
            'engine': engine,
            'n_features': n_features
        }
    
    def test_streaming_single_sample(self, setup_streaming_engine):
        """Test streaming inference with single sample."""
        engine_data = setup_streaming_engine
        engine = engine_data['engine']
        
        # Create test sample
        X_test = np.random.normal(0.5, 0.1, size=(engine_data['n_features'],))
        X_test = np.clip(X_test, 0, 1)
        
        # Process sample
        result = engine.process_sample(X_test)
        
        # Verify output
        assert 'prediction' in result
        assert 'anomaly_score' in result
        assert 'reconstruction_error' in result
        assert 'isolation_score' in result
        assert 'latency_ms' in result
        
        assert result['prediction'] in [0, 1]
        assert 0 <= result['anomaly_score'] <= 1
        assert result['latency_ms'] > 0
    
    def test_streaming_multiple_samples(self, setup_streaming_engine):
        """Test streaming inference with multiple samples."""
        engine_data = setup_streaming_engine
        engine = engine_data['engine']
        
        # Process multiple samples
        n_samples = 50
        for i in range(n_samples):
            X_test = np.random.normal(0.5, 0.1, size=(engine_data['n_features'],))
            X_test = np.clip(X_test, 0, 1)
            
            result = engine.process_sample(X_test)
            
            assert result['prediction'] in [0, 1]
            assert 0 <= result['anomaly_score'] <= 1
        
        # Check statistics
        stats = engine.get_statistics()
        assert stats['total_samples'] == n_samples
        assert stats['avg_latency_ms'] > 0
    
    def test_streaming_latency(self, setup_streaming_engine):
        """Test that streaming inference meets latency requirements."""
        engine_data = setup_streaming_engine
        engine = engine_data['engine']
        
        # Process samples and measure latency
        n_samples = 100
        latencies = []
        
        for i in range(n_samples):
            X_test = np.random.normal(0.5, 0.1, size=(engine_data['n_features'],))
            X_test = np.clip(X_test, 0, 1)
            
            result = engine.process_sample(X_test)
            latencies.append(result['latency_ms'])
        
        # Check average latency
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        # Verify latency meets healthcare requirements (<100ms)
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms"
        assert max_latency < 200, f"Maximum latency {max_latency:.2f}ms is too high"
    
    def test_streaming_mini_batch(self, setup_streaming_engine):
        """Test streaming inference with mini-batches."""
        engine_data = setup_streaming_engine
        
        # Create engine with mini-batch size
        from stream_inference import StreamingInferenceEngine
        
        engine = StreamingInferenceEngine(
            engine_data['engine'].autoencoder,
            engine_data['engine'].isolation_forest,
            engine_data['engine'].fusion,
            mini_batch_size=10
        )
        
        # Create mini-batch
        X_batch = np.random.normal(0.5, 0.1, size=(10, engine_data['n_features']))
        X_batch = np.clip(X_batch, 0, 1)
        
        # Process mini-batch
        result = engine.process_sample(X_batch)
        
        # Verify output
        assert len(result['prediction']) == 10
        assert len(result['anomaly_score']) == 10
        assert all(p in [0, 1] for p in result['prediction'])
        assert all(0 <= s <= 1 for s in result['anomaly_score'])
    
    def test_streaming_statistics(self, setup_streaming_engine):
        """Test streaming engine statistics tracking."""
        engine_data = setup_streaming_engine
        engine = engine_data['engine']
        
        # Initial statistics
        stats = engine.get_statistics()
        assert stats['total_samples'] == 0
        assert stats['total_anomalies'] == 0
        
        # Process some samples
        n_samples = 20
        for i in range(n_samples):
            X_test = np.random.normal(0.5, 0.1, size=(engine_data['n_features'],))
            X_test = np.clip(X_test, 0, 1)
            engine.process_sample(X_test)
        
        # Check updated statistics
        stats = engine.get_statistics()
        assert stats['total_samples'] == n_samples
        assert stats['avg_latency_ms'] > 0
        assert stats['max_latency_ms'] >= stats['min_latency_ms']
        assert 0 <= stats['anomaly_rate'] <= 1


class TestInferenceErrorHandling:
    """Test error handling in inference pipelines."""
    
    def test_invalid_input_shape(self):
        """Test error handling for invalid input shape."""
        # Create minimal model
        ae_config = {
            'encoding_dim': 5,
            'learning_rate': 0.001,
            'epochs': 1,
            'batch_size': 32,
            'use_gpu': False,
            'random_state': 42
        }
        
        autoencoder = AutoencoderDetector(input_dim=10, config=ae_config)
        autoencoder.build_model(use_dropout=False)
        
        # Try to process input with wrong number of features
        X_wrong = np.random.normal(0.5, 0.1, size=(5, 15))  # Wrong: 15 features instead of 10
        
        # Should raise an error or handle gracefully
        with pytest.raises(Exception):
            autoencoder.compute_reconstruction_error(X_wrong)
    
    def test_missing_model_files(self, tmp_path):
        """Test error handling when model files are missing."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        
        # Try to load from empty directory
        from inference import load_models
        
        config = {
            'autoencoder': {'encoding_dim': 5, 'learning_rate': 0.001},
            'isolation_forest': {'n_estimators': 10},
            'fusion': {'weight_autoencoder': 0.5, 'weight_isolation': 0.5}
        }
        
        with pytest.raises(FileNotFoundError):
            load_models(str(model_dir), 10, config)
    
    def test_empty_input(self):
        """Test error handling for empty input."""
        # Create minimal model
        ae_config = {
            'encoding_dim': 5,
            'learning_rate': 0.001,
            'epochs': 1,
            'batch_size': 32,
            'use_gpu': False,
            'random_state': 42
        }
        
        autoencoder = AutoencoderDetector(input_dim=10, config=ae_config)
        autoencoder.build_model(use_dropout=False)
        
        # Try to process empty input
        X_empty = np.array([]).reshape(0, 10)
        
        # Should handle gracefully or raise appropriate error
        result = autoencoder.compute_reconstruction_error(X_empty)
        assert len(result) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
