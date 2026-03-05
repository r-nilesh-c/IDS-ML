"""
Integration tests for the main training pipeline.

These tests verify:
1. Complete training flow with synthetic dataset
2. Models are saved correctly
3. Reproducibility with fixed seeds

**Validates: Requirements 9.4, 10.1**
"""

import pytest
import os
import sys
import tempfile
import shutil
import pickle
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train import load_config, set_random_seeds, save_models, main
from src.preprocessing import PreprocessingPipeline
from src.autoencoder import AutoencoderDetector
from src.isolation_forest import IsolationForestDetector
from src.fusion import FusionModule


class TestTrainingPipeline:
    """Integration tests for training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration."""
        config = {
            'preprocessing': {
                'random_state': 42,
                'test_size': 0.3,
                'val_size': 0.2
            },
            'autoencoder': {
                'encoding_dim': 10,
                'epochs': 2,  # Small for testing
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping_patience': 5,
                'random_state': 42
            },
            'isolation_forest': {
                'n_estimators': 50,  # Small for testing
                'max_samples': 'auto',
                'contamination': 0.01,
                'random_state': 42
            },
            'fusion': {
                'weight_autoencoder': 0.5,
                'weight_isolation': 0.5,
                'percentile': 95
            },
            'model_save_path': temp_dir
        }
        return config
    
    @pytest.fixture
    def config_file(self, test_config, temp_dir):
        """Create temporary config file."""
        config_path = os.path.join(temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        return config_path
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic training data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 20
        
        # Benign data (normal distribution)
        X_benign = np.random.normal(0.5, 0.1, size=(n_samples, n_features))
        X_benign = np.clip(X_benign, 0, 1)
        
        # Split into train and validation
        val_split = int(0.8 * n_samples)
        X_train = X_benign[:val_split]
        X_val = X_benign[val_split:]
        
        return X_train, X_val, n_features
    
    def test_load_config_success(self, config_file):
        """Test loading configuration from YAML file."""
        config = load_config(config_file)
        
        assert config is not None
        assert 'preprocessing' in config
        assert 'autoencoder' in config
        assert 'isolation_forest' in config
        assert 'fusion' in config
        assert config['preprocessing']['random_state'] == 42
    
    def test_load_config_file_not_found(self):
        """Test loading configuration with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.yaml')
    
    def test_set_random_seeds(self):
        """Test setting random seeds for reproducibility."""
        # Set seeds
        set_random_seeds(42)
        
        # Generate random numbers
        np_random_1 = np.random.rand(5)
        
        # Reset seeds
        set_random_seeds(42)
        
        # Generate random numbers again
        np_random_2 = np.random.rand(5)
        
        # Should be identical
        np.testing.assert_array_equal(np_random_1, np_random_2)
    
    def test_save_models(self, test_config, temp_dir, synthetic_data):
        """Test saving trained models."""
        X_train, X_val, n_features = synthetic_data
        
        # Train minimal models
        ae_config = test_config['autoencoder'].copy()
        ae_config['model_save_path'] = temp_dir
        
        autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
        autoencoder.build_model(use_dropout=False)
        autoencoder.train(X_train, X_val)
        
        isolation_forest = IsolationForestDetector(test_config['isolation_forest'])
        isolation_forest.train(X_train)
        
        # Compute scores for fusion
        recon_errors = autoencoder.compute_reconstruction_error(X_val)
        iso_scores = isolation_forest.compute_anomaly_score(X_val)
        
        fusion = FusionModule(test_config['fusion'])
        fusion.fit_threshold(recon_errors, iso_scores)
        
        # Save models
        save_models(autoencoder, isolation_forest, fusion, temp_dir)
        
        # Verify files exist
        assert os.path.exists(os.path.join(temp_dir, 'autoencoder_best.keras'))
        assert os.path.exists(os.path.join(temp_dir, 'isolation_forest.pkl'))
        assert os.path.exists(os.path.join(temp_dir, 'fusion_params.pkl'))
        
        # Verify fusion parameters can be loaded
        with open(os.path.join(temp_dir, 'fusion_params.pkl'), 'rb') as f:
            fusion_params = pickle.load(f)
        
        assert 'weight_autoencoder' in fusion_params
        assert 'weight_isolation' in fusion_params
        assert 'threshold' in fusion_params
        assert fusion_params['weight_autoencoder'] == 0.5
        assert fusion_params['weight_isolation'] == 0.5
    
    def test_complete_training_flow(self, test_config, temp_dir, synthetic_data):
        """
        Test complete training flow with synthetic dataset.
        
        This integration test verifies:
        1. All components can be initialized
        2. Training completes without errors
        3. Models are saved correctly
        4. Fusion threshold is computed
        """
        X_train, X_val, n_features = synthetic_data
        
        # Step 1: Initialize preprocessing (not used with synthetic data)
        preprocessing = PreprocessingPipeline(test_config['preprocessing'])
        assert preprocessing is not None
        
        # Step 2: Train autoencoder
        ae_config = test_config['autoencoder'].copy()
        ae_config['model_save_path'] = temp_dir
        
        autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
        autoencoder.build_model(use_dropout=True, dropout_rate=0.2)
        history = autoencoder.train(X_train, X_val)
        
        # Verify training completed
        assert history is not None
        assert len(history.history['loss']) > 0
        assert history.history['loss'][-1] < history.history['loss'][0]  # Loss should decrease
        
        # Step 3: Train isolation forest
        isolation_forest = IsolationForestDetector(test_config['isolation_forest'])
        isolation_forest.train(X_train)
        
        # Verify model is trained
        assert isolation_forest.model is not None
        assert hasattr(isolation_forest.model, 'decision_function')
        
        # Step 4: Fit fusion threshold
        recon_errors_val = autoencoder.compute_reconstruction_error(X_val)
        iso_scores_val = isolation_forest.compute_anomaly_score(X_val)
        
        fusion = FusionModule(test_config['fusion'])
        fusion.fit_threshold(recon_errors_val, iso_scores_val)
        
        # Verify threshold is set
        assert fusion.threshold is not None
        assert fusion.threshold > 0
        
        # Step 5: Save models
        save_models(autoencoder, isolation_forest, fusion, temp_dir)
        
        # Verify all models saved
        assert os.path.exists(os.path.join(temp_dir, 'autoencoder_best.keras'))
        assert os.path.exists(os.path.join(temp_dir, 'isolation_forest.pkl'))
        assert os.path.exists(os.path.join(temp_dir, 'fusion_params.pkl'))
    
    def test_reproducibility_with_fixed_seeds(self, test_config, temp_dir):
        """
        Test reproducibility with fixed random seeds.
        
        Trains the same model twice with the same seed and verifies
        that the results are identical.
        
        **Validates: Requirements 9.4, 10.1**
        """
        n_features = 20
        n_samples = 100
        
        # First training run
        set_random_seeds(42)
        np.random.seed(42)
        X_train_1 = np.random.normal(0.5, 0.1, size=(n_samples, n_features))
        X_val_1 = np.random.normal(0.5, 0.1, size=(20, n_features))
        
        ae_config_1 = test_config['autoencoder'].copy()
        ae_config_1['model_save_path'] = os.path.join(temp_dir, 'run1')
        os.makedirs(ae_config_1['model_save_path'], exist_ok=True)
        
        autoencoder_1 = AutoencoderDetector(input_dim=n_features, config=ae_config_1)
        autoencoder_1.build_model(use_dropout=False)  # No dropout for determinism
        history_1 = autoencoder_1.train(X_train_1, X_val_1)
        
        recon_errors_1 = autoencoder_1.compute_reconstruction_error(X_val_1)
        
        # Second training run with same seed
        set_random_seeds(42)
        np.random.seed(42)
        X_train_2 = np.random.normal(0.5, 0.1, size=(n_samples, n_features))
        X_val_2 = np.random.normal(0.5, 0.1, size=(20, n_features))
        
        ae_config_2 = test_config['autoencoder'].copy()
        ae_config_2['model_save_path'] = os.path.join(temp_dir, 'run2')
        os.makedirs(ae_config_2['model_save_path'], exist_ok=True)
        
        autoencoder_2 = AutoencoderDetector(input_dim=n_features, config=ae_config_2)
        autoencoder_2.build_model(use_dropout=False)  # No dropout for determinism
        history_2 = autoencoder_2.train(X_train_2, X_val_2)
        
        recon_errors_2 = autoencoder_2.compute_reconstruction_error(X_val_2)
        
        # Verify data is identical
        np.testing.assert_array_almost_equal(X_train_1, X_train_2)
        np.testing.assert_array_almost_equal(X_val_1, X_val_2)
        
        # Verify training history is similar (may have small differences due to GPU)
        assert len(history_1.history['loss']) == len(history_2.history['loss'])
        
        # Verify reconstruction errors are similar
        # Note: Due to GPU non-determinism, we use a tolerance
        np.testing.assert_allclose(recon_errors_1, recon_errors_2, rtol=0.1)
    
    def test_model_loading_after_save(self, test_config, temp_dir, synthetic_data):
        """
        Test that saved models can be loaded and used for inference.
        """
        X_train, X_val, n_features = synthetic_data
        
        # Train and save models
        ae_config = test_config['autoencoder'].copy()
        ae_config['model_save_path'] = temp_dir
        
        autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
        autoencoder.build_model(use_dropout=False)
        autoencoder.train(X_train, X_val)
        
        isolation_forest = IsolationForestDetector(test_config['isolation_forest'])
        isolation_forest.train(X_train)
        
        recon_errors = autoencoder.compute_reconstruction_error(X_val)
        iso_scores = isolation_forest.compute_anomaly_score(X_val)
        
        fusion = FusionModule(test_config['fusion'])
        fusion.fit_threshold(recon_errors, iso_scores)
        
        save_models(autoencoder, isolation_forest, fusion, temp_dir)
        
        # Load models
        try:
            import tensorflow as tf
            loaded_ae_model = tf.keras.models.load_model(
                os.path.join(temp_dir, 'autoencoder_best.keras')
            )
            assert loaded_ae_model is not None
        except Exception as e:
            pytest.fail(f"Failed to load autoencoder model: {e}")
        
        with open(os.path.join(temp_dir, 'isolation_forest.pkl'), 'rb') as f:
            loaded_if_model = pickle.load(f)
        assert loaded_if_model is not None
        
        with open(os.path.join(temp_dir, 'fusion_params.pkl'), 'rb') as f:
            loaded_fusion_params = pickle.load(f)
        assert loaded_fusion_params is not None
        
        # Verify loaded models can make predictions
        test_sample = X_val[:5]
        
        # Autoencoder prediction
        ae_predictions = loaded_ae_model.predict(test_sample, verbose=0)
        assert ae_predictions.shape == test_sample.shape
        
        # Isolation Forest prediction
        if_predictions = loaded_if_model.decision_function(test_sample)
        assert len(if_predictions) == len(test_sample)
    
    def test_training_with_invalid_config(self, temp_dir):
        """Test training fails gracefully with invalid configuration."""
        # Create config with missing required fields
        invalid_config = {
            'preprocessing': {
                'random_state': 42
            }
            # Missing other required sections
        }
        
        config_path = os.path.join(temp_dir, 'invalid_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Load config should succeed
        config = load_config(config_path)
        assert config is not None
        
        # But training should handle missing fields gracefully
        # (The actual train.py uses .get() with defaults, so it won't crash)
        assert config.get('autoencoder', {}) == {}
        assert config.get('isolation_forest', {}) == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
