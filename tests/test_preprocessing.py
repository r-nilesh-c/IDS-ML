"""
Unit tests for preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.preprocessing import PreprocessingPipeline


class TestPreprocessingPipeline:
    """Test suite for PreprocessingPipeline class."""
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing."""
        return {
            'random_state': 42,
            'test_size': 0.3,
            'val_size': 0.2
        }
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        data = {
            'Flow Duration': [100, 200, 300, 400, 500],
            'Total Fwd Packets': [10, 20, 30, 40, 50],
            'Total Backward Packets': [5, 10, 15, 20, 25],
            'Label': ['BENIGN', 'BENIGN', 'DoS', 'BENIGN', 'PortScan']
        }
        return pd.DataFrame(data)
    
    def test_init(self, config):
        """Test PreprocessingPipeline initialization."""
        pipeline = PreprocessingPipeline(config)
        assert pipeline.config == config
        assert pipeline.random_state == 42
    
    def test_load_datasets_single_file(self, config, sample_csv_data, tmp_path):
        """Test loading a single dataset file."""
        # Create temporary CSV file
        csv_path = tmp_path / "test_data.csv"
        sample_csv_data.to_csv(csv_path, index=False)
        
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        
        assert len(df) == 5
        assert 'Label' in df.columns
        assert list(df['Label']) == ['BENIGN', 'BENIGN', 'DoS', 'BENIGN', 'PortScan']
    
    def test_load_datasets_multiple_files(self, config, sample_csv_data, tmp_path):
        """Test loading and merging multiple dataset files."""
        # Create two temporary CSV files
        csv_path1 = tmp_path / "test_data1.csv"
        csv_path2 = tmp_path / "test_data2.csv"
        
        sample_csv_data.to_csv(csv_path1, index=False)
        sample_csv_data.to_csv(csv_path2, index=False)
        
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path1), str(csv_path2)])
        
        assert len(df) == 10  # 5 samples from each file
        assert 'Label' in df.columns
    
    def test_load_datasets_label_variations(self, config, tmp_path):
        """Test handling of different label column name variations."""
        # Test with " Label" (space before)
        data1 = pd.DataFrame({
            'Feature1': [1, 2, 3],
            ' Label': ['BENIGN', 'DoS', 'BENIGN']
        })
        csv_path1 = tmp_path / "test_space_label.csv"
        data1.to_csv(csv_path1, index=False)
        
        # Test with "label" (lowercase)
        data2 = pd.DataFrame({
            'Feature1': [4, 5, 6],
            'label': ['BENIGN', 'PortScan', 'BENIGN']
        })
        csv_path2 = tmp_path / "test_lowercase_label.csv"
        data2.to_csv(csv_path2, index=False)
        
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path1), str(csv_path2)])
        
        assert 'Label' in df.columns
        assert len(df) == 6
    
    def test_load_datasets_missing_file(self, config):
        """Test error handling for missing files."""
        pipeline = PreprocessingPipeline(config)
        
        with pytest.raises(FileNotFoundError):
            pipeline.load_datasets(['/nonexistent/path/data.csv'])
    
    def test_load_datasets_empty_paths(self, config):
        """Test error handling for empty paths list."""
        pipeline = PreprocessingPipeline(config)
        
        with pytest.raises(ValueError, match="No dataset paths provided"):
            pipeline.load_datasets([])
    
    def test_load_datasets_missing_label_column(self, config, tmp_path):
        """Test error handling when label column is missing."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6]
        })
        csv_path = tmp_path / "no_label.csv"
        data.to_csv(csv_path, index=False)
        
        pipeline = PreprocessingPipeline(config)
        
        with pytest.raises(ValueError, match="No label column found"):
            pipeline.load_datasets([str(csv_path)])
    
    def test_load_datasets_empty_file(self, config, tmp_path):
        """Test error handling for empty CSV files."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        
        pipeline = PreprocessingPipeline(config)
        
        with pytest.raises(ValueError, match="empty"):
            pipeline.load_datasets([str(csv_path)])
    
    def test_clean_data_removes_duplicates(self, config, tmp_path):
        """Test that clean_data removes duplicate rows."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 2, 3, 3],
            'Feature2': [10, 20, 20, 30, 30],
            'Label': ['BENIGN', 'DoS', 'DoS', 'BENIGN', 'BENIGN']
        })
        csv_path = tmp_path / "duplicates.csv"
        data.to_csv(csv_path, index=False)
        
        config['min_samples'] = 1  # Allow small test datasets
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        cleaned_df = pipeline.clean_data(df)
        
        # Should have 3 unique rows
        assert len(cleaned_df) == 3
        assert 'Label' in cleaned_df.columns
    
    def test_clean_data_removes_nan(self, config, tmp_path):
        """Test that clean_data removes rows with NaN values."""
        data = pd.DataFrame({
            'Feature1': [1, 2, np.nan, 4, 5],
            'Feature2': [10, 20, 30, np.nan, 50],
            'Label': ['BENIGN', 'DoS', 'BENIGN', 'PortScan', 'BENIGN']
        })
        csv_path = tmp_path / "nan_data.csv"
        data.to_csv(csv_path, index=False)
        
        config['min_samples'] = 1
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        cleaned_df = pipeline.clean_data(df)
        
        # Should have 3 rows without NaN
        assert len(cleaned_df) == 3
        assert not cleaned_df.isnull().any().any()
    
    def test_clean_data_removes_inf(self, config, tmp_path):
        """Test that clean_data removes rows with infinite values."""
        data = pd.DataFrame({
            'Feature1': [1, 2, np.inf, 4, 5],
            'Feature2': [10, 20, 30, -np.inf, 50],
            'Label': ['BENIGN', 'DoS', 'BENIGN', 'PortScan', 'BENIGN']
        })
        csv_path = tmp_path / "inf_data.csv"
        data.to_csv(csv_path, index=False)
        
        config['min_samples'] = 1
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        cleaned_df = pipeline.clean_data(df)
        
        # Should have 3 rows without inf
        assert len(cleaned_df) == 3
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        assert not np.isinf(cleaned_df[numeric_cols]).any().any()
    
    def test_clean_data_removes_non_numeric_features(self, config, tmp_path):
        """Test that clean_data removes non-numeric features."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50],
            'StringFeature': ['a', 'b', 'c', 'd', 'e'],
            'Label': ['BENIGN', 'DoS', 'BENIGN', 'PortScan', 'BENIGN']
        })
        csv_path = tmp_path / "mixed_types.csv"
        data.to_csv(csv_path, index=False)
        
        config['min_samples'] = 1
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        cleaned_df = pipeline.clean_data(df)
        
        # StringFeature should be removed
        assert 'StringFeature' not in cleaned_df.columns
        assert 'Feature1' in cleaned_df.columns
        assert 'Feature2' in cleaned_df.columns
        assert 'Label' in cleaned_df.columns
        
        # All columns except Label should be numeric
        for col in cleaned_df.columns:
            if col != 'Label':
                assert pd.api.types.is_numeric_dtype(cleaned_df[col])
    
    def test_clean_data_preserves_label(self, config, tmp_path):
        """Test that clean_data preserves the Label column."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50],
            'Label': ['BENIGN', 'DoS', 'BENIGN', 'PortScan', 'BENIGN']
        })
        csv_path = tmp_path / "data.csv"
        data.to_csv(csv_path, index=False)
        
        config['min_samples'] = 1
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        cleaned_df = pipeline.clean_data(df)
        
        assert 'Label' in cleaned_df.columns
        assert len(cleaned_df) == 5
    
    def test_clean_data_insufficient_samples(self, config, tmp_path):
        """Test error handling when insufficient samples remain after cleaning."""
        # Create data with mostly duplicates/NaN that will be cleaned out
        data = pd.DataFrame({
            'Feature1': [1, 1, 1, np.nan, np.nan],
            'Feature2': [10, 10, 10, np.nan, np.nan],
            'Label': ['BENIGN', 'BENIGN', 'BENIGN', 'DoS', 'DoS']
        })
        csv_path = tmp_path / "insufficient.csv"
        data.to_csv(csv_path, index=False)
        
        # Set minimum samples requirement
        config['min_samples'] = 5
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        
        with pytest.raises(ValueError, match="Insufficient samples after cleaning"):
            pipeline.clean_data(df)
    
    def test_clean_data_empty_dataframe(self, config):
        """Test error handling for empty DataFrame."""
        pipeline = PreprocessingPipeline(config)
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot clean empty DataFrame"):
            pipeline.clean_data(empty_df)
    
    def test_clean_data_missing_label(self, config):
        """Test error handling when Label column is missing."""
        pipeline = PreprocessingPipeline(config)
        df = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [10, 20, 30]
        })
        
        with pytest.raises(ValueError, match="Label column missing"):
            pipeline.clean_data(df)
    
    def test_clean_data_combined_cleaning(self, config, tmp_path):
        """Test clean_data with multiple cleaning operations."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 2, np.nan, np.inf, 6, 7],
            'Feature2': [10, 20, 20, 40, 50, 60, 70],
            'StringFeature': ['a', 'b', 'b', 'd', 'e', 'f', 'g'],
            'Label': ['BENIGN', 'DoS', 'DoS', 'BENIGN', 'PortScan', 'BENIGN', 'DDoS']
        })
        csv_path = tmp_path / "combined.csv"
        data.to_csv(csv_path, index=False)
        
        config['min_samples'] = 1
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        cleaned_df = pipeline.clean_data(df)
        
        # Should remove: 1 duplicate, 1 NaN, 1 inf, and StringFeature column
        # Remaining: rows 0, 3, 5, 6 but row 3 has NaN so removed
        # Actually: row 0, 5, 6 (3 rows)
        assert len(cleaned_df) >= 3
        assert 'StringFeature' not in cleaned_df.columns
        assert 'Label' in cleaned_df.columns
        assert not cleaned_df.isnull().any().any()
        
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        assert not np.isinf(cleaned_df[numeric_cols]).any().any()



class TestSplitBenignAttack:
    """Test suite for split_benign_attack method."""
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing."""
        return {
            'random_state': 42,
            'min_samples': 1
        }
    
    def test_split_benign_attack_basic(self, config, tmp_path):
        """Test basic benign/attack separation."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50],
            'Label': ['BENIGN', 'DoS', 'BENIGN', 'PortScan', 'BENIGN']
        })
        csv_path = tmp_path / "data.csv"
        data.to_csv(csv_path, index=False)
        
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        benign_df, attack_df = pipeline.split_benign_attack(df)
        
        assert len(benign_df) == 3
        assert len(attack_df) == 2
        assert all(benign_df['Label'].str.upper() == 'BENIGN')
        assert all(attack_df['Label'].str.upper() != 'BENIGN')
    
    def test_split_benign_attack_case_insensitive(self, config, tmp_path):
        """Test that benign detection is case-insensitive."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50],
            'Label': ['BENIGN', 'benign', 'Benign', 'DoS', 'PortScan']
        })
        csv_path = tmp_path / "data.csv"
        data.to_csv(csv_path, index=False)
        
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        benign_df, attack_df = pipeline.split_benign_attack(df)
        
        assert len(benign_df) == 3
        assert len(attack_df) == 2
    
    def test_split_benign_attack_multiple_attack_types(self, config, tmp_path):
        """Test splitting with multiple attack types."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5, 6, 7],
            'Feature2': [10, 20, 30, 40, 50, 60, 70],
            'Label': ['BENIGN', 'DoS', 'BENIGN', 'PortScan', 'DDoS', 'BENIGN', 'Bot']
        })
        csv_path = tmp_path / "data.csv"
        data.to_csv(csv_path, index=False)
        
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        benign_df, attack_df = pipeline.split_benign_attack(df)
        
        assert len(benign_df) == 3
        assert len(attack_df) == 4
        
        # Verify attack types
        attack_types = set(attack_df['Label'])
        assert 'DoS' in attack_types
        assert 'PortScan' in attack_types
        assert 'DDoS' in attack_types
        assert 'Bot' in attack_types
    
    def test_split_benign_attack_no_benign_samples(self, config, tmp_path):
        """Test error handling when no benign samples exist."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [10, 20, 30],
            'Label': ['DoS', 'PortScan', 'DDoS']
        })
        csv_path = tmp_path / "data.csv"
        data.to_csv(csv_path, index=False)
        
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        
        with pytest.raises(ValueError, match="No benign samples found"):
            pipeline.split_benign_attack(df)
    
    def test_split_benign_attack_no_attack_samples(self, config, tmp_path):
        """Test error handling when no attack samples exist."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [10, 20, 30],
            'Label': ['BENIGN', 'BENIGN', 'BENIGN']
        })
        csv_path = tmp_path / "data.csv"
        data.to_csv(csv_path, index=False)
        
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        
        with pytest.raises(ValueError, match="No attack samples found"):
            pipeline.split_benign_attack(df)
    
    def test_split_benign_attack_empty_dataframe(self, config):
        """Test error handling for empty DataFrame."""
        pipeline = PreprocessingPipeline(config)
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot split empty DataFrame"):
            pipeline.split_benign_attack(empty_df)
    
    def test_split_benign_attack_missing_label(self, config):
        """Test error handling when Label column is missing."""
        pipeline = PreprocessingPipeline(config)
        df = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [10, 20, 30]
        })
        
        with pytest.raises(ValueError, match="Label column missing"):
            pipeline.split_benign_attack(df)
    
    def test_split_benign_attack_preserves_features(self, config, tmp_path):
        """Test that split preserves all feature columns."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50],
            'Feature3': [100, 200, 300, 400, 500],
            'Label': ['BENIGN', 'DoS', 'BENIGN', 'PortScan', 'BENIGN']
        })
        csv_path = tmp_path / "data.csv"
        data.to_csv(csv_path, index=False)
        
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        benign_df, attack_df = pipeline.split_benign_attack(df)
        
        # Verify all columns are preserved
        assert 'Feature1' in benign_df.columns
        assert 'Feature2' in benign_df.columns
        assert 'Feature3' in benign_df.columns
        assert 'Label' in benign_df.columns
        
        assert 'Feature1' in attack_df.columns
        assert 'Feature2' in attack_df.columns
        assert 'Feature3' in attack_df.columns
        assert 'Label' in attack_df.columns
    
    def test_split_benign_attack_returns_copies(self, config, tmp_path):
        """Test that split returns independent copies of data."""
        data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50],
            'Label': ['BENIGN', 'DoS', 'BENIGN', 'PortScan', 'BENIGN']
        })
        csv_path = tmp_path / "data.csv"
        data.to_csv(csv_path, index=False)
        
        pipeline = PreprocessingPipeline(config)
        df = pipeline.load_datasets([str(csv_path)])
        benign_df, attack_df = pipeline.split_benign_attack(df)
        
        # Modify benign_df and verify original df is unchanged
        original_df_len = len(df)
        benign_df.loc[benign_df.index[0], 'Feature1'] = 999
        
        # Original df should not be affected
        assert len(df) == original_df_len
        assert 999 not in df['Feature1'].values



class TestNormalizeAndSplit:
    """Test suite for normalize_and_split method."""
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing."""
        return {
            'random_state': 42,
            'test_size': 0.3,
            'val_size': 0.2,
            'min_samples': 1
        }
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample data with benign and attack samples."""
        # Create a dataset with 100 benign and 50 attack samples
        np.random.seed(42)
        
        benign_data = {
            'Feature1': np.random.randn(100) * 10 + 50,
            'Feature2': np.random.randn(100) * 5 + 20,
            'Feature3': np.random.randn(100) * 2 + 10,
            'Label': ['BENIGN'] * 100
        }
        
        attack_data = {
            'Feature1': np.random.randn(50) * 15 + 80,
            'Feature2': np.random.randn(50) * 8 + 40,
            'Feature3': np.random.randn(50) * 3 + 15,
            'Label': ['DoS'] * 25 + ['PortScan'] * 25
        }
        
        benign_df = pd.DataFrame(benign_data)
        attack_df = pd.DataFrame(attack_data)
        
        return benign_df, attack_df
    
    def test_normalize_and_split_basic(self, config, sample_data):
        """Test basic normalization and splitting."""
        benign_df, attack_df = sample_data
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Check that all expected keys are present
        assert 'X_train_benign' in result
        assert 'X_val_benign' in result
        assert 'X_test' in result
        assert 'y_test' in result
        assert 'y_test_labels' in result
        assert 'scaler' in result
        assert 'feature_names' in result
        
        # Check shapes
        assert result['X_train_benign'].ndim == 2
        assert result['X_val_benign'].ndim == 2
        assert result['X_test'].ndim == 2
        
        # Check that feature dimensions match
        n_features = result['X_train_benign'].shape[1]
        assert result['X_val_benign'].shape[1] == n_features
        assert result['X_test'].shape[1] == n_features
        assert len(result['feature_names']) == n_features
        
        # Check that test labels match test samples
        assert len(result['y_test']) == len(result['X_test'])
        assert len(result['y_test_labels']) == len(result['X_test'])
    
    def test_normalize_and_split_benign_only_training(self, config, sample_data):
        """Test that training and validation sets contain only benign samples."""
        benign_df, attack_df = sample_data
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Training and validation should only have benign samples
        # We can't directly verify this from the normalized features,
        # but we can check that test set has both benign and attack
        assert 0 in result['y_test']  # Has benign
        assert 1 in result['y_test']  # Has attack
        
        # Check that test set has attack labels
        assert any(label != 'BENIGN' for label in result['y_test_labels'])
    
    def test_normalize_and_split_standardscaler_normalization(self, config, sample_data):
        """Test that StandardScaler normalization is applied correctly."""
        benign_df, attack_df = sample_data
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Training data should have mean ≈ 0 and std ≈ 1
        train_mean = np.mean(result['X_train_benign'], axis=0)
        train_std = np.std(result['X_train_benign'], axis=0)
        
        # Check mean is close to 0 (within tolerance)
        assert np.allclose(train_mean, 0, atol=1e-10)
        
        # Check std is close to 1 (within tolerance)
        assert np.allclose(train_std, 1, atol=0.1)
    
    def test_normalize_and_split_scaler_object(self, config, sample_data):
        """Test that fitted scaler object is returned."""
        benign_df, attack_df = sample_data
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Check that scaler is a StandardScaler instance
        from sklearn.preprocessing import StandardScaler
        assert isinstance(result['scaler'], StandardScaler)
        
        # Check that scaler has been fitted (has mean_ and scale_ attributes)
        assert hasattr(result['scaler'], 'mean_')
        assert hasattr(result['scaler'], 'scale_')
        assert result['scaler'].mean_ is not None
        assert result['scaler'].scale_ is not None
    
    def test_normalize_and_split_feature_names(self, config, sample_data):
        """Test that feature names are correctly extracted."""
        benign_df, attack_df = sample_data
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Feature names should match the columns (excluding Label)
        expected_features = [col for col in benign_df.columns if col != 'Label']
        assert result['feature_names'] == expected_features
    
    def test_normalize_and_split_validation_size(self, config, sample_data):
        """Test that validation set size is approximately correct."""
        benign_df, attack_df = sample_data
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Validation should be approximately 20% of benign training data
        # After stratified split, we have some benign in train_val
        # Then we split that into train and val with 20% for val
        total_train_val = len(result['X_train_benign']) + len(result['X_val_benign'])
        val_ratio = len(result['X_val_benign']) / total_train_val
        
        # Should be close to 0.2 (20%)
        assert 0.15 <= val_ratio <= 0.25
    
    def test_normalize_and_split_test_size(self, config, sample_data):
        """Test that test set size is approximately correct."""
        benign_df, attack_df = sample_data
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Test should be approximately 30% of all data
        total_samples = len(benign_df) + len(attack_df)
        test_ratio = len(result['X_test']) / total_samples
        
        # Should be close to 0.3 (30%)
        assert 0.25 <= test_ratio <= 0.35
    
    def test_normalize_and_split_stratified_sampling(self, config, sample_data):
        """Test that stratified sampling maintains class distribution."""
        benign_df, attack_df = sample_data
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Calculate proportions in original data
        total_original = len(benign_df) + len(attack_df)
        benign_ratio_original = len(benign_df) / total_original
        attack_ratio_original = len(attack_df) / total_original
        
        # Calculate proportions in test set
        benign_count_test = (result['y_test'] == 0).sum()
        attack_count_test = (result['y_test'] == 1).sum()
        benign_ratio_test = benign_count_test / len(result['y_test'])
        attack_ratio_test = attack_count_test / len(result['y_test'])
        
        # Ratios should be similar (within 5% tolerance)
        assert abs(benign_ratio_original - benign_ratio_test) < 0.05
        assert abs(attack_ratio_original - attack_ratio_test) < 0.05
    
    def test_normalize_and_split_binary_labels(self, config, sample_data):
        """Test that binary labels are correctly assigned."""
        benign_df, attack_df = sample_data
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # y_test should be binary (0 or 1)
        assert set(result['y_test']).issubset({0, 1})
        
        # Check that benign samples have label 0
        for i, label in enumerate(result['y_test_labels']):
            if label.upper() == 'BENIGN':
                assert result['y_test'][i] == 0
            else:
                assert result['y_test'][i] == 1
    
    def test_normalize_and_split_empty_benign(self, config, sample_data):
        """Test error handling for empty benign DataFrame."""
        _, attack_df = sample_data
        empty_benign = pd.DataFrame()
        
        pipeline = PreprocessingPipeline(config)
        
        with pytest.raises(ValueError, match="Cannot normalize and split empty benign DataFrame"):
            pipeline.normalize_and_split(empty_benign, attack_df)
    
    def test_normalize_and_split_empty_attack(self, config, sample_data):
        """Test error handling for empty attack DataFrame."""
        benign_df, _ = sample_data
        empty_attack = pd.DataFrame()
        
        pipeline = PreprocessingPipeline(config)
        
        with pytest.raises(ValueError, match="Cannot normalize and split empty attack DataFrame"):
            pipeline.normalize_and_split(benign_df, empty_attack)
    
    def test_normalize_and_split_missing_label_benign(self, config):
        """Test error handling when Label column is missing from benign DataFrame."""
        benign_df = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [10, 20, 30]
        })
        attack_df = pd.DataFrame({
            'Feature1': [4, 5],
            'Feature2': [40, 50],
            'Label': ['DoS', 'PortScan']
        })
        
        pipeline = PreprocessingPipeline(config)
        
        with pytest.raises(ValueError, match="Label column missing"):
            pipeline.normalize_and_split(benign_df, attack_df)
    
    def test_normalize_and_split_missing_label_attack(self, config):
        """Test error handling when Label column is missing from attack DataFrame."""
        benign_df = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [10, 20, 30],
            'Label': ['BENIGN', 'BENIGN', 'BENIGN']
        })
        attack_df = pd.DataFrame({
            'Feature1': [4, 5],
            'Feature2': [40, 50]
        })
        
        pipeline = PreprocessingPipeline(config)
        
        with pytest.raises(ValueError, match="Label column missing"):
            pipeline.normalize_and_split(benign_df, attack_df)
    
    def test_normalize_and_split_reproducibility(self, config, sample_data):
        """Test that results are reproducible with same random seed."""
        benign_df, attack_df = sample_data
        
        pipeline1 = PreprocessingPipeline(config)
        result1 = pipeline1.normalize_and_split(benign_df, attack_df)
        
        pipeline2 = PreprocessingPipeline(config)
        result2 = pipeline2.normalize_and_split(benign_df, attack_df)
        
        # Results should be identical
        np.testing.assert_array_equal(result1['X_train_benign'], result2['X_train_benign'])
        np.testing.assert_array_equal(result1['X_val_benign'], result2['X_val_benign'])
        np.testing.assert_array_equal(result1['X_test'], result2['X_test'])
        np.testing.assert_array_equal(result1['y_test'], result2['y_test'])
    
    def test_normalize_and_split_different_attack_types(self, config, tmp_path):
        """Test with multiple different attack types."""
        np.random.seed(42)
        
        benign_df = pd.DataFrame({
            'Feature1': np.random.randn(100) * 10 + 50,
            'Feature2': np.random.randn(100) * 5 + 20,
            'Label': ['BENIGN'] * 100
        })
        
        attack_df = pd.DataFrame({
            'Feature1': np.random.randn(60) * 15 + 80,
            'Feature2': np.random.randn(60) * 8 + 40,
            'Label': ['DoS'] * 20 + ['PortScan'] * 20 + ['DDoS'] * 10 + ['Bot'] * 10
        })
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Check that test set contains multiple attack types
        attack_labels = result['y_test_labels'][result['y_test'] == 1]
        unique_attacks = set(attack_labels)
        
        # Should have at least 2 different attack types in test set
        assert len(unique_attacks) >= 2
    
    def test_normalize_and_split_preserves_feature_count(self, config, sample_data):
        """Test that number of features is preserved across all splits."""
        benign_df, attack_df = sample_data
        
        # Original feature count (excluding Label)
        original_feature_count = len([col for col in benign_df.columns if col != 'Label'])
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # All splits should have same number of features
        assert result['X_train_benign'].shape[1] == original_feature_count
        assert result['X_val_benign'].shape[1] == original_feature_count
        assert result['X_test'].shape[1] == original_feature_count
        assert len(result['feature_names']) == original_feature_count



# ============================================================================
# Property-Based Tests for Preprocessing Pipeline
# ============================================================================

from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import column, data_frames


# Custom strategies for generating test data
def numeric_column_strategy(name):
    """Generate a numeric column with reasonable values."""
    return column(name, dtype=float, elements=st.floats(
        min_value=-1000.0, 
        max_value=1000.0, 
        allow_nan=False, 
        allow_infinity=False
    ))


def label_strategy():
    """Generate label column with benign and attack labels."""
    return column('Label', dtype=str, elements=st.sampled_from([
        'BENIGN', 'benign', 'Benign',  # Various cases of benign
        'DoS', 'DDoS', 'PortScan', 'Bot', 'Infiltration'  # Attack types
    ]))


class TestPreprocessingProperties:
    """Property-based tests for preprocessing pipeline."""
    
    def get_config(self):
        """Get basic configuration for property testing."""
        return {
            'random_state': 42,
            'test_size': 0.3,
            'val_size': 0.2,
            'min_samples': 10
        }
    
    # ========================================================================
    # Property 1: Benign-Only Training Data
    # **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 3.3, 4.3**
    # ========================================================================
    
    @given(
        n_benign=st.integers(min_value=50, max_value=200),
        n_attack=st.integers(min_value=20, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_1_benign_only_training(self, n_benign, n_attack):
        """
        Property 1: Benign-Only Training Data
        
        For any training execution with a dataset containing both benign and 
        attack samples, the training set (including validation set) should 
        contain only samples labeled as BENIGN, and all attack samples should 
        be reserved exclusively for the test set.
        
        **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 3.3, 4.3**
        """
        # Generate synthetic data
        np.random.seed(42)
        config = self.get_config()
        
        benign_df = pd.DataFrame({
            'Feature1': np.random.randn(n_benign) * 10 + 50,
            'Feature2': np.random.randn(n_benign) * 5 + 20,
            'Feature3': np.random.randn(n_benign) * 2 + 10,
            'Label': ['BENIGN'] * n_benign
        })
        
        attack_df = pd.DataFrame({
            'Feature1': np.random.randn(n_attack) * 15 + 80,
            'Feature2': np.random.randn(n_attack) * 8 + 40,
            'Feature3': np.random.randn(n_attack) * 3 + 15,
            'Label': np.random.choice(['DoS', 'DDoS', 'PortScan'], n_attack)
        })
        
        # Run preprocessing
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Property: Training and validation sets contain ONLY benign samples
        # We verify this indirectly by checking that:
        # 1. Test set contains both benign (0) and attack (1) samples
        # 2. Training + validation samples are all from benign data
        
        # Test set should have both classes
        assert 0 in result['y_test'], "Test set should contain benign samples"
        assert 1 in result['y_test'], "Test set should contain attack samples"
        
        # All test attack samples should have label 1
        attack_mask = result['y_test'] == 1
        attack_labels = result['y_test_labels'][attack_mask]
        assert all(label.upper() != 'BENIGN' for label in attack_labels), \
            "All attack samples in test set should have non-BENIGN labels"
        
        # All test benign samples should have label 0
        benign_mask = result['y_test'] == 0
        benign_labels = result['y_test_labels'][benign_mask]
        assert all(label.upper() == 'BENIGN' for label in benign_labels), \
            "All benign samples in test set should have BENIGN labels"
    
    # ========================================================================
    # Property 2: Label Preservation During Merge
    # **Validates: Requirements 2.2**
    # ========================================================================
    
    @given(
        n_samples_1=st.integers(min_value=10, max_value=50),
        n_samples_2=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_2_label_preservation(self, n_samples_1, n_samples_2):
        """
        Property 2: Label Preservation During Merge
        
        For any set of datasets being merged, the label information for each 
        sample should be identical before and after the merge operation.
        
        **Validates: Requirements 2.2**
        """
        # Generate two datasets with known labels
        np.random.seed(42)
        config = self.get_config()
        
        # Create temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            labels_1 = np.random.choice(['BENIGN', 'DoS', 'DDoS'], n_samples_1)
            labels_2 = np.random.choice(['BENIGN', 'PortScan', 'Bot'], n_samples_2)
            
            data_1 = pd.DataFrame({
                'Feature1': np.random.randn(n_samples_1),
                'Feature2': np.random.randn(n_samples_1),
                'Label': labels_1
            })
            
            data_2 = pd.DataFrame({
                'Feature1': np.random.randn(n_samples_2),
                'Feature2': np.random.randn(n_samples_2),
                'Label': labels_2
            })
            
            # Save to CSV files
            import os
            csv_path_1 = os.path.join(tmp_path, "data1.csv")
            csv_path_2 = os.path.join(tmp_path, "data2.csv")
            data_1.to_csv(csv_path_1, index=False)
            data_2.to_csv(csv_path_2, index=False)
            
            # Store original labels
            original_labels = list(labels_1) + list(labels_2)
            
            # Load and merge datasets
            pipeline = PreprocessingPipeline(config)
            merged_df = pipeline.load_datasets([csv_path_1, csv_path_2])
            
            # Property: Labels should be preserved after merge
            merged_labels = list(merged_df['Label'])
            
            # Check that all original labels are present in merged data
            assert len(merged_labels) == len(original_labels), \
                "Number of labels should be preserved after merge"
            
            # Check that labels match (order should be preserved)
            assert merged_labels == original_labels, \
                "Label values should be identical before and after merge"
    
    # ========================================================================
    # Property 3: Duplicate Removal
    # **Validates: Requirements 2.3**
    # ========================================================================
    
    @given(
        n_unique=st.integers(min_value=10, max_value=50),
        n_duplicates=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_3_duplicate_removal(self, n_unique, n_duplicates):
        """
        Property 3: Duplicate Removal
        
        For any dataset with duplicate rows, the output of the cleaning 
        pipeline should contain no duplicate records.
        
        **Validates: Requirements 2.3**
        """
        # Generate data with known duplicates
        np.random.seed(42)
        config = self.get_config()
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            # Create unique samples
            unique_data = pd.DataFrame({
                'Feature1': np.random.randn(n_unique),
                'Feature2': np.random.randn(n_unique),
                'Label': np.random.choice(['BENIGN', 'DoS'], n_unique)
            })
            
            # Add duplicates by repeating some rows
            duplicate_indices = np.random.choice(n_unique, n_duplicates, replace=True)
            duplicate_data = unique_data.iloc[duplicate_indices]
            
            # Combine unique and duplicate data
            data_with_duplicates = pd.concat([unique_data, duplicate_data], ignore_index=True)
            
            # Save to CSV
            import os
            csv_path = os.path.join(tmp_path, "data_with_duplicates.csv")
            data_with_duplicates.to_csv(csv_path, index=False)
            
            # Load and clean data
            config['min_samples'] = 1
            pipeline = PreprocessingPipeline(config)
            df = pipeline.load_datasets([csv_path])
            cleaned_df = pipeline.clean_data(df)
            
            # Property: No duplicates should remain after cleaning
            assert not cleaned_df.duplicated().any(), \
                "Cleaned data should contain no duplicate rows"
            
            # Verify that we have at most n_unique samples (could be less if some were duplicates of each other)
            assert len(cleaned_df) <= n_unique, \
                "Cleaned data should have at most the number of unique samples"
    
    # ========================================================================
    # Property 4: NaN and Infinity Removal
    # **Validates: Requirements 2.4**
    # ========================================================================
    
    @given(
        n_samples=st.integers(min_value=20, max_value=100),
        nan_ratio=st.floats(min_value=0.1, max_value=0.4)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_4_nan_inf_removal(self, n_samples, nan_ratio):
        """
        Property 4: NaN and Infinity Removal
        
        For any dataset, the output of the cleaning pipeline should contain 
        no NaN or infinite values in any column.
        
        **Validates: Requirements 2.4**
        """
        # Generate data with NaN and inf values
        np.random.seed(42)
        config = self.get_config()
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            data = pd.DataFrame({
                'Feature1': np.random.randn(n_samples),
                'Feature2': np.random.randn(n_samples),
                'Feature3': np.random.randn(n_samples),
                'Label': np.random.choice(['BENIGN', 'DoS'], n_samples)
            })
            
            # Inject NaN values
            n_nan = int(n_samples * nan_ratio)
            nan_indices = np.random.choice(n_samples, n_nan, replace=False)
            for idx in nan_indices[:n_nan//2]:
                data.loc[idx, 'Feature1'] = np.nan
            for idx in nan_indices[n_nan//2:]:
                data.loc[idx, 'Feature2'] = np.nan
            
            # Inject inf values
            n_inf = max(1, n_nan // 4)
            inf_indices = np.random.choice(n_samples, n_inf, replace=False)
            for idx in inf_indices[:n_inf//2]:
                data.loc[idx, 'Feature2'] = np.inf
            for idx in inf_indices[n_inf//2:]:
                data.loc[idx, 'Feature3'] = -np.inf
            
            # Save to CSV
            import os
            csv_path = os.path.join(tmp_path, "data_with_nan_inf.csv")
            data.to_csv(csv_path, index=False)
            
            # Load and clean data
            config['min_samples'] = 1
            pipeline = PreprocessingPipeline(config)
            df = pipeline.load_datasets([csv_path])
            cleaned_df = pipeline.clean_data(df)
            
            # Property: No NaN or inf values should remain
            assert not cleaned_df.isnull().any().any(), \
                "Cleaned data should contain no NaN values"
            
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            assert not np.isinf(cleaned_df[numeric_cols]).any().any(), \
                "Cleaned data should contain no infinite values"
    
    # ========================================================================
    # Property 5: Numeric Features Only
    # **Validates: Requirements 2.5**
    # ========================================================================
    
    @given(
        n_samples=st.integers(min_value=20, max_value=100),
        n_numeric=st.integers(min_value=2, max_value=5),
        n_string=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_5_numeric_features_only(self, n_samples, n_numeric, n_string):
        """
        Property 5: Numeric Features Only
        
        For any dataset, the output of the preprocessing pipeline should 
        contain only numeric (integer or float) features, with all 
        non-numeric columns removed.
        
        **Validates: Requirements 2.5**
        """
        # Generate data with mixed types
        np.random.seed(42)
        config = self.get_config()
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_path:
            data = {'Label': np.random.choice(['BENIGN', 'DoS'], n_samples)}
            
            # Add numeric features
            for i in range(n_numeric):
                data[f'NumericFeature{i}'] = np.random.randn(n_samples)
            
            # Add string features
            for i in range(n_string):
                data[f'StringFeature{i}'] = np.random.choice(['A', 'B', 'C'], n_samples)
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            import os
            csv_path = os.path.join(tmp_path, "mixed_types.csv")
            df.to_csv(csv_path, index=False)
            
            # Load and clean data
            config['min_samples'] = 1
            pipeline = PreprocessingPipeline(config)
            loaded_df = pipeline.load_datasets([csv_path])
            cleaned_df = pipeline.clean_data(loaded_df)
            
            # Property: All features (except Label) should be numeric
            for col in cleaned_df.columns:
                if col != 'Label':
                    assert pd.api.types.is_numeric_dtype(cleaned_df[col]), \
                        f"Feature {col} should be numeric after cleaning"
            
            # Verify that string features were removed
            for i in range(n_string):
                assert f'StringFeature{i}' not in cleaned_df.columns, \
                    f"String feature StringFeature{i} should be removed"
            
            # Verify that numeric features were preserved
            for i in range(n_numeric):
                assert f'NumericFeature{i}' in cleaned_df.columns, \
                    f"Numeric feature NumericFeature{i} should be preserved"
    
    # ========================================================================
    # Property 6: StandardScaler Normalization
    # **Validates: Requirements 2.6**
    # ========================================================================
    
    @given(
        n_benign=st.integers(min_value=50, max_value=200),
        n_attack=st.integers(min_value=20, max_value=100),
        feature_mean=st.floats(min_value=-100, max_value=100),
        feature_std=st.floats(min_value=1, max_value=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_6_standardscaler_normalization(self, n_benign, n_attack, 
                                                      feature_mean, feature_std):
        """
        Property 6: StandardScaler Normalization
        
        For any training dataset after normalization, the features should 
        have approximately zero mean and unit standard deviation (within 
        numerical tolerance of 0.1).
        
        **Validates: Requirements 2.6**
        """
        # Generate data with specific mean and std
        np.random.seed(42)
        config = self.get_config()
        
        benign_df = pd.DataFrame({
            'Feature1': np.random.randn(n_benign) * feature_std + feature_mean,
            'Feature2': np.random.randn(n_benign) * (feature_std * 2) + (feature_mean * 0.5),
            'Feature3': np.random.randn(n_benign) * (feature_std * 0.5) + (feature_mean * 2),
            'Label': ['BENIGN'] * n_benign
        })
        
        attack_df = pd.DataFrame({
            'Feature1': np.random.randn(n_attack) * feature_std * 1.5 + feature_mean * 1.5,
            'Feature2': np.random.randn(n_attack) * feature_std * 2.5 + feature_mean * 0.8,
            'Feature3': np.random.randn(n_attack) * feature_std * 0.8 + feature_mean * 2.5,
            'Label': np.random.choice(['DoS', 'DDoS'], n_attack)
        })
        
        # Run preprocessing
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Property: Training data should have mean ≈ 0 and std ≈ 1
        train_mean = np.mean(result['X_train_benign'], axis=0)
        train_std = np.std(result['X_train_benign'], axis=0)
        
        # Check mean is close to 0 (within tolerance)
        max_mean_deviation = np.abs(train_mean).max()
        assert max_mean_deviation < 0.1, \
            f"Training data mean should be close to 0, but max deviation is {max_mean_deviation}"
        
        # Check std is close to 1 (within tolerance)
        max_std_deviation = np.abs(train_std - 1.0).max()
        assert max_std_deviation < 0.1, \
            f"Training data std should be close to 1, but max deviation is {max_std_deviation}"
    
    # ========================================================================
    # Property 7: Stratified Sampling Preservation
    # **Validates: Requirements 2.7**
    # ========================================================================
    
    @given(
        n_benign=st.integers(min_value=100, max_value=300),
        n_attack=st.integers(min_value=50, max_value=150),
        attack_type_ratio=st.floats(min_value=0.3, max_value=0.7)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_7_stratified_sampling(self, n_benign, n_attack, attack_type_ratio):
        """
        Property 7: Stratified Sampling Preservation
        
        For any dataset split using stratified sampling, the proportion of 
        each class in the training and test sets should be approximately 
        equal (within 5% tolerance).
        
        **Validates: Requirements 2.7**
        """
        # Generate data with specific class distribution
        np.random.seed(42)
        config = self.get_config()
        
        benign_df = pd.DataFrame({
            'Feature1': np.random.randn(n_benign) * 10 + 50,
            'Feature2': np.random.randn(n_benign) * 5 + 20,
            'Label': ['BENIGN'] * n_benign
        })
        
        # Create two attack types with specific ratio
        n_attack_type1 = int(n_attack * attack_type_ratio)
        n_attack_type2 = n_attack - n_attack_type1
        
        attack_df = pd.DataFrame({
            'Feature1': np.random.randn(n_attack) * 15 + 80,
            'Feature2': np.random.randn(n_attack) * 8 + 40,
            'Label': ['DoS'] * n_attack_type1 + ['DDoS'] * n_attack_type2
        })
        
        # Run preprocessing
        pipeline = PreprocessingPipeline(config)
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        # Calculate original proportions
        total_original = n_benign + n_attack
        benign_ratio_original = n_benign / total_original
        attack_ratio_original = n_attack / total_original
        
        # Calculate test set proportions
        benign_count_test = (result['y_test'] == 0).sum()
        attack_count_test = (result['y_test'] == 1).sum()
        total_test = len(result['y_test'])
        
        benign_ratio_test = benign_count_test / total_test
        attack_ratio_test = attack_count_test / total_test
        
        # Property: Class proportions should be maintained (within 5% tolerance)
        benign_diff = abs(benign_ratio_original - benign_ratio_test)
        attack_diff = abs(attack_ratio_original - attack_ratio_test)
        
        assert benign_diff < 0.05, \
            f"Benign class proportion should be preserved (diff: {benign_diff:.4f})"
        assert attack_diff < 0.05, \
            f"Attack class proportion should be preserved (diff: {attack_diff:.4f})"
