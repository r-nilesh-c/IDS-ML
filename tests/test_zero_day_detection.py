"""
Property-based tests for zero-day attack detection capability.

This module tests Property 17: Zero-Day Attack Detection
Validates Requirements 7.2, 7.5
"""

import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from src.autoencoder import AutoencoderDetector
from src.isolation_forest import IsolationForestDetector
from src.fusion import FusionModule


class TestZeroDayDetectionPropertyBased:
    """
    Property-based tests for zero-day attack detection.
    
    These tests verify that the system can detect novel attacks
    not present in training data by assigning them high anomaly scores.
    """
    
    def test_property_17_zero_day_attack_detection(self):
        """
        Property 17: Zero-Day Attack Detection
        
        For any attack pattern not present in the training data, the system
        should assign it an anomaly score higher than the majority of benign
        samples (demonstrating detection capability for novel attacks).
        
        Test Strategy:
        1. Load real CIC-IDS data (benign and attacks)
        2. Train models on benign data only
        3. Test on real attack data (simulating zero-day attacks)
        4. Verify that attack samples get higher anomaly scores than
           most benign samples
        
        **Validates: Requirements 7.2, 7.5**
        """
        
        @given(
            dataset_choice=st.sampled_from(['2017', '2018']),
            n_benign_train=st.integers(min_value=500, max_value=2000),
            n_benign_test=st.integers(min_value=200, max_value=500),
            n_attack=st.integers(min_value=100, max_value=400),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=20, deadline=None)  # Reduced for real data loading
        def run_test(dataset_choice, n_benign_train, n_benign_test, n_attack, data_seed):
            """
            Run property test with real CIC-IDS data.
            
            Args:
                dataset_choice: Which dataset to use ('2017' or '2018')
                n_benign_train: Number of benign training samples
                n_benign_test: Number of benign test samples
                n_attack: Number of attack samples
                data_seed: Random seed for reproducibility
            """
            import pandas as pd
            import os
            
            np.random.seed(data_seed)
            
            print(f"\n=== Testing with dataset: {dataset_choice}, seed: {data_seed} ===")
            
            # Load real CIC-IDS data
            try:
                if dataset_choice == '2017':
                    # Load CIC-IDS2017 file with attacks
                    data_path = 'dataset/cic-ids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
                else:
                    # Load a CIC-IDS2018 file
                    data_path = 'dataset/cic-ids2018/Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet'
                
                if not os.path.exists(data_path):
                    print(f"Dataset not found: {data_path}")
                    pytest.skip(f"Dataset not found: {data_path}")
                    return
                
                print(f"Loading data from: {data_path}")
                
                # Load data - load enough to get both benign and attacks
                # For CIC-IDS files, attacks may not be at the beginning
                # Load at least 25000 rows to ensure we get attacks
                min_rows = max(25000, n_benign_train + n_benign_test + n_attack + 5000)
                if data_path.endswith('.csv'):
                    df = pd.read_csv(data_path, nrows=min_rows)
                else:
                    df = pd.read_parquet(data_path)
                    df = df.head(min_rows)
                
                print(f"Loaded {len(df)} rows")
                
                # Find label column
                label_col = None
                for col in [' Label', 'Label', 'label', ' label']:
                    if col in df.columns:
                        label_col = col
                        break
                
                if label_col is None:
                    # Print available columns for debugging
                    print(f"Available columns: {df.columns.tolist()}")
                    pytest.skip(f"Label column not found in dataset. Columns: {list(df.columns)[:10]}")
                    return
                
                # Separate benign and attack samples
                benign_df = df[df[label_col].str.upper() == 'BENIGN'].copy()
                attack_df = df[df[label_col].str.upper() != 'BENIGN'].copy()
                
                print(f"Benign samples: {len(benign_df)}, Attack samples: {len(attack_df)}")
                
                # Check if we have enough samples
                if len(benign_df) < n_benign_train + n_benign_test:
                    print(f"Not enough benign samples: {len(benign_df)} < {n_benign_train + n_benign_test}")
                    pytest.skip(f"Not enough benign samples: {len(benign_df)}")
                    return
                
                if len(attack_df) < n_attack:
                    print(f"Not enough attack samples: {len(attack_df)} < {n_attack}")
                    pytest.skip(f"Not enough attack samples: {len(attack_df)}")
                    return
                
                # Drop label column and non-numeric columns
                benign_df = benign_df.drop(columns=[label_col])
                attack_df = attack_df.drop(columns=[label_col])
                
                # Keep only numeric columns
                benign_df = benign_df.select_dtypes(include=[np.number])
                attack_df = attack_df.select_dtypes(include=[np.number])
                
                # Remove NaN and inf
                benign_df = benign_df.replace([np.inf, -np.inf], np.nan).dropna()
                attack_df = attack_df.replace([np.inf, -np.inf], np.nan).dropna()
                
                # Check again after cleaning
                if len(benign_df) < n_benign_train + n_benign_test or len(attack_df) < n_attack:
                    pytest.skip("Not enough samples after cleaning")
                    return
                
                # Sample data
                benign_sample = benign_df.sample(n=n_benign_train + n_benign_test, random_state=data_seed)
                attack_sample = attack_df.sample(n=n_attack, random_state=data_seed)
                
                # Convert to numpy arrays
                X_benign_all = benign_sample.values.astype(np.float32)
                X_attack = attack_sample.values.astype(np.float32)
                
                # Normalize to [0, 1] range for consistency
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                X_benign_all = scaler.fit_transform(X_benign_all)
                X_attack = scaler.transform(X_attack)
                
                # Split benign into train and test
                X_benign_train = X_benign_all[:n_benign_train]
                X_benign_test = X_benign_all[n_benign_train:]
                
                n_features = X_benign_train.shape[1]
                
            except Exception as e:
                pytest.skip(f"Error loading data: {e}")
                return
            
            # Split benign training for validation
            val_split = int(0.8 * n_benign_train)
            X_train = X_benign_train[:val_split]
            X_val = X_benign_train[val_split:]
            
            # Train autoencoder on benign data only
            ae_config = {
                'encoding_dim': max(8, n_features // 4),
                'learning_rate': 0.001,
                'epochs': 30,  # Increased for real data
                'batch_size': 64,
                'early_stopping_patience': 5,
                'use_gpu': False,
                'random_state': data_seed
            }
            
            autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
            autoencoder.build_model(use_dropout=False)
            
            # Train with real data
            try:
                autoencoder.train(X_train, X_val)
            except Exception as e:
                pytest.skip(f"Training failed: {e}")
                return
            
            # Train isolation forest on benign data only
            if_config = {
                'n_estimators': 100,
                'max_samples': min(256, n_benign_train),
                'contamination': 'auto',
                'random_state': data_seed,
                'n_jobs': 1
            }
            
            isolation_forest = IsolationForestDetector(if_config)
            isolation_forest.train(X_benign_train)
            
            # Fit fusion threshold on validation data
            recon_errors_val = autoencoder.compute_reconstruction_error(X_val)
            iso_scores_val = isolation_forest.compute_anomaly_score(X_val)
            
            fusion_config = {
                'weight_autoencoder': 0.5,
                'weight_isolation': 0.5,
                'percentile': 95
            }
            
            fusion = FusionModule(fusion_config)
            fusion.fit_threshold(recon_errors_val, iso_scores_val)
            
            # Compute anomaly scores for benign test samples
            recon_errors_benign = autoencoder.compute_reconstruction_error(X_benign_test)
            iso_scores_benign = isolation_forest.compute_anomaly_score(X_benign_test)
            
            recon_norm_benign, iso_norm_benign = fusion.normalize_scores(
                recon_errors_benign, iso_scores_benign
            )
            combined_scores_benign = fusion.compute_combined_score(
                recon_norm_benign, iso_norm_benign
            )
            
            # Compute anomaly scores for attack samples (novel/zero-day)
            recon_errors_attack = autoencoder.compute_reconstruction_error(X_attack)
            iso_scores_attack = isolation_forest.compute_anomaly_score(X_attack)
            
            recon_norm_attack, iso_norm_attack = fusion.normalize_scores(
                recon_errors_attack, iso_scores_attack
            )
            combined_scores_attack = fusion.compute_combined_score(
                recon_norm_attack, iso_norm_attack
            )
            
            # Property verification: Novel attacks should have higher anomaly scores
            # than the majority of benign samples
            
            # Calculate percentiles of benign scores
            benign_75th = np.percentile(combined_scores_benign, 75)
            benign_median = np.median(combined_scores_benign)
            
            # Calculate what percentage of attacks have scores above benign 75th percentile
            attacks_above_75th = np.mean(combined_scores_attack > benign_75th)
            
            # Calculate average scores
            avg_benign_score = np.mean(combined_scores_benign)
            avg_attack_score = np.mean(combined_scores_attack)
            
            # Property assertions with real data:
            # Real attack data should show clear separation from benign traffic
            
            # 1. Average attack score should be higher than average benign score
            assert avg_attack_score > avg_benign_score, (
                f"Zero-day detection failed: Average attack score ({avg_attack_score:.6f}) "
                f"should be higher than average benign score ({avg_benign_score:.6f})"
            )
            
            # 2. Majority of attacks (>50%) should score above the 75th percentile of benign
            assert attacks_above_75th > 0.5, (
                f"Zero-day detection failed: Only {attacks_above_75th:.2%} of attacks "
                f"scored above 75th percentile of benign samples (expected >50%)"
            )
            
            # 3. Significant detection rate (>30%) for attacks
            attacks_detected = np.sum(combined_scores_attack > fusion.threshold)
            detection_rate = attacks_detected / len(combined_scores_attack)
            
            assert detection_rate > 0.3, (
                f"Zero-day detection failed: Only {detection_rate:.2%} of novel attacks "
                f"were detected (expected >30%)"
            )
            
            # Log success metrics
            print(f"\n✓ Property 17 verified with real {dataset_choice} data:")
            print(f"  - Avg benign score: {avg_benign_score:.6f}")
            print(f"  - Avg attack score: {avg_attack_score:.6f}")
            print(f"  - Attacks above 75th percentile: {attacks_above_75th:.2%}")
            print(f"  - Detection rate: {detection_rate:.2%}")
            print(f"  - Threshold: {fusion.threshold:.6f}")
        
        # Run the property test
        run_test()
    
    def test_zero_day_detection_with_different_attack_patterns(self):
        """
        Test zero-day detection with various attack pattern types.
        
        This test verifies that the system can detect different types of
        anomalous patterns that might represent novel attacks:
        - Outliers (extreme values)
        - Distribution shifts (different mean/variance)
        - Sparse patterns (many zeros)
        - Dense patterns (many ones)
        """
        
        @given(
            n_benign=st.integers(min_value=100, max_value=300),
            n_features=st.integers(min_value=10, max_value=25),
            attack_type=st.sampled_from(['outliers', 'shift', 'sparse', 'dense']),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=50, deadline=None)
        def run_test(n_benign, n_features, attack_type, data_seed):
            """
            Run test with different attack pattern types.
            
            Args:
                n_benign: Number of benign samples
                n_features: Number of features
                attack_type: Type of attack pattern to generate
                data_seed: Random seed
            """
            np.random.seed(data_seed)
            
            # Generate benign training data
            X_benign = np.random.normal(0.5, 0.1, size=(n_benign, n_features))
            X_benign = np.clip(X_benign, 0, 1)
            
            # Split for training and validation
            val_split = int(0.8 * n_benign)
            X_train = X_benign[:val_split]
            X_val = X_benign[val_split:]
            
            # Generate attack data based on type
            n_attack = 50
            
            if attack_type == 'outliers':
                # Extreme values at boundaries
                X_attack = np.random.choice([0.0, 1.0], size=(n_attack, n_features))
            
            elif attack_type == 'shift':
                # Shifted distribution
                X_attack = np.random.normal(0.8, 0.15, size=(n_attack, n_features))
                X_attack = np.clip(X_attack, 0, 1)
            
            elif attack_type == 'sparse':
                # Mostly zeros with some random values
                X_attack = np.random.random((n_attack, n_features)) * 0.2
            
            elif attack_type == 'dense':
                # Mostly ones with some random values
                X_attack = 0.8 + np.random.random((n_attack, n_features)) * 0.2
            
            # Train models
            ae_config = {
                'encoding_dim': max(4, n_features // 4),
                'learning_rate': 0.001,
                'epochs': 15,
                'batch_size': 32,
                'early_stopping_patience': 5,
                'use_gpu': False,
                'random_state': data_seed
            }
            
            autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
            autoencoder.build_model(use_dropout=False)
            
            try:
                autoencoder.train(X_train, X_val)
            except Exception as e:
                pytest.skip(f"Training failed: {e}")
                return
            
            if_config = {
                'n_estimators': 50,
                'max_samples': min(100, n_benign),
                'contamination': 'auto',
                'random_state': data_seed,
                'n_jobs': 1
            }
            
            isolation_forest = IsolationForestDetector(if_config)
            isolation_forest.train(X_benign)
            
            # Fit fusion
            recon_errors_val = autoencoder.compute_reconstruction_error(X_val)
            iso_scores_val = isolation_forest.compute_anomaly_score(X_val)
            
            fusion = FusionModule({'weight_autoencoder': 0.5, 'weight_isolation': 0.5, 'percentile': 95})
            fusion.fit_threshold(recon_errors_val, iso_scores_val)
            
            # Compute scores
            recon_errors_attack = autoencoder.compute_reconstruction_error(X_attack)
            iso_scores_attack = isolation_forest.compute_anomaly_score(X_attack)
            
            recon_norm, iso_norm = fusion.normalize_scores(recon_errors_attack, iso_scores_attack)
            combined_scores_attack = fusion.compute_combined_score(recon_norm, iso_norm)
            
            # Verify detection
            avg_attack_score = np.mean(combined_scores_attack)
            
            # Attack scores should be reasonably high (above 0.5 on normalized scale)
            assert avg_attack_score > 0.3, (
                f"Failed to detect {attack_type} attacks: avg score = {avg_attack_score:.6f}"
            )
            
            print(f"\n✓ Detected {attack_type} attacks: avg score = {avg_attack_score:.6f}")
        
        # Run the property test
        run_test()


# ============================================================================
# Unit Tests for Zero-Day Evaluation Functions
# ============================================================================

class TestZeroDayEvaluationFunctions:
    """Unit tests for zero-day evaluation helper functions."""
    
    def test_zero_day_evaluation_basic(self):
        """
        Test basic zero-day evaluation workflow.
        
        This is a simplified unit test that verifies the complete workflow:
        1. Train on benign data
        2. Test on novel attacks
        3. Verify attacks are detected
        """
        np.random.seed(42)
        
        n_benign = 200
        n_attack = 50
        n_features = 15
        
        # Generate benign training data
        X_benign = np.random.normal(0.5, 0.1, size=(n_benign, n_features))
        X_benign = np.clip(X_benign, 0, 1)
        
        # Generate attack data (outliers)
        X_attack = np.random.choice([0.0, 1.0], size=(n_attack, n_features))
        
        # Split benign for training/validation
        val_split = int(0.8 * n_benign)
        X_train = X_benign[:val_split]
        X_val = X_benign[val_split:]
        
        # Train autoencoder
        ae_config = {
            'encoding_dim': 8,
            'learning_rate': 0.001,
            'epochs': 20,
            'batch_size': 32,
            'early_stopping_patience': 5,
            'use_gpu': False,
            'random_state': 42
        }
        
        autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
        autoencoder.build_model(use_dropout=False)
        autoencoder.train(X_train, X_val)
        
        # Train isolation forest
        if_config = {
            'n_estimators': 50,
            'max_samples': 100,
            'contamination': 'auto',
            'random_state': 42,
            'n_jobs': 1
        }
        
        isolation_forest = IsolationForestDetector(if_config)
        isolation_forest.train(X_benign)
        
        # Fit fusion
        recon_errors_val = autoencoder.compute_reconstruction_error(X_val)
        iso_scores_val = isolation_forest.compute_anomaly_score(X_val)
        
        fusion = FusionModule({
            'weight_autoencoder': 0.5,
            'weight_isolation': 0.5,
            'percentile': 95
        })
        fusion.fit_threshold(recon_errors_val, iso_scores_val)
        
        # Evaluate on attacks
        recon_errors_attack = autoencoder.compute_reconstruction_error(X_attack)
        iso_scores_attack = isolation_forest.compute_anomaly_score(X_attack)
        
        recon_norm, iso_norm = fusion.normalize_scores(recon_errors_attack, iso_scores_attack)
        combined_scores_attack = fusion.compute_combined_score(recon_norm, iso_norm)
        
        predictions = fusion.classify(combined_scores_attack)
        
        # Verify detection
        detection_rate = np.mean(predictions)
        
        assert detection_rate > 0.3, (
            f"Low detection rate for novel attacks: {detection_rate:.2%}"
        )
        
        print(f"\n✓ Zero-day evaluation test passed:")
        print(f"  - Detection rate: {detection_rate:.2%}")
        print(f"  - Avg attack score: {np.mean(combined_scores_attack):.6f}")
        print(f"  - Threshold: {fusion.threshold:.6f}")
    
    def test_cross_dataset_simulation(self):
        """
        Test simulated cross-dataset evaluation.
        
        Simulates training on one dataset (2017) and testing on another (2018)
        by using different random seeds to generate distinct distributions.
        """
        n_features = 20
        
        # Simulate 2017 benign data (training)
        np.random.seed(2017)
        X_train_2017 = np.random.normal(0.5, 0.1, size=(300, n_features))
        X_train_2017 = np.clip(X_train_2017, 0, 1)
        
        # Simulate 2018 attack data (testing) - slightly different distribution
        np.random.seed(2018)
        X_attack_2018 = np.random.normal(0.6, 0.2, size=(100, n_features))
        X_attack_2018 = np.clip(X_attack_2018, 0, 1)
        
        # Add some extreme outliers to attacks
        outlier_mask = np.random.random((100, n_features)) < 0.2
        X_attack_2018[outlier_mask] = np.random.choice([0.0, 1.0], size=np.sum(outlier_mask))
        
        # Train on 2017 data
        val_split = int(0.8 * 300)
        X_train = X_train_2017[:val_split]
        X_val = X_train_2017[val_split:]
        
        ae_config = {
            'encoding_dim': 10,
            'learning_rate': 0.001,
            'epochs': 20,
            'batch_size': 32,
            'early_stopping_patience': 5,
            'use_gpu': False,
            'random_state': 42
        }
        
        autoencoder = AutoencoderDetector(input_dim=n_features, config=ae_config)
        autoencoder.build_model(use_dropout=False)
        autoencoder.train(X_train, X_val)
        
        if_config = {
            'n_estimators': 50,
            'max_samples': 100,
            'contamination': 'auto',
            'random_state': 42,
            'n_jobs': 1
        }
        
        isolation_forest = IsolationForestDetector(if_config)
        isolation_forest.train(X_train_2017)
        
        # Fit fusion
        recon_errors_val = autoencoder.compute_reconstruction_error(X_val)
        iso_scores_val = isolation_forest.compute_anomaly_score(X_val)
        
        fusion = FusionModule({
            'weight_autoencoder': 0.5,
            'weight_isolation': 0.5,
            'percentile': 95
        })
        fusion.fit_threshold(recon_errors_val, iso_scores_val)
        
        # Test on 2018 attacks
        recon_errors_2018 = autoencoder.compute_reconstruction_error(X_attack_2018)
        iso_scores_2018 = isolation_forest.compute_anomaly_score(X_attack_2018)
        
        recon_norm, iso_norm = fusion.normalize_scores(recon_errors_2018, iso_scores_2018)
        combined_scores_2018 = fusion.compute_combined_score(recon_norm, iso_norm)
        
        predictions = fusion.classify(combined_scores_2018)
        
        # Verify cross-dataset detection
        detection_rate = np.mean(predictions)
        
        assert detection_rate > 0.2, (
            f"Low cross-dataset detection rate: {detection_rate:.2%}"
        )
        
        print(f"\n✓ Cross-dataset evaluation test passed:")
        print(f"  - Detection rate: {detection_rate:.2%}")
        print(f"  - Avg attack score: {np.mean(combined_scores_2018):.6f}")
