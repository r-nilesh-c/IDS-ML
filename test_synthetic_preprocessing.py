"""
Test script to verify preprocessing pipeline with synthetic dataset.

This script creates a small synthetic dataset and runs it through the
complete preprocessing pipeline to verify all data quality checks pass.
"""

import numpy as np
import pandas as pd
import tempfile
import os
from src.preprocessing import PreprocessingPipeline


def create_synthetic_dataset(n_benign=100, n_attack=50):
    """
    Create a small synthetic network flow dataset.
    
    Args:
        n_benign: Number of benign samples
        n_attack: Number of attack samples
        
    Returns:
        DataFrame with synthetic network flow data
    """
    np.random.seed(42)
    
    # Create benign traffic (normal patterns)
    benign_data = {
        'Flow_Duration': np.random.exponential(scale=1000, size=n_benign),
        'Total_Fwd_Packets': np.random.poisson(lam=10, size=n_benign),
        'Total_Backward_Packets': np.random.poisson(lam=8, size=n_benign),
        'Flow_Bytes_per_s': np.random.normal(loc=5000, scale=1000, size=n_benign),
        'Flow_Packets_per_s': np.random.normal(loc=50, scale=10, size=n_benign),
        'Flow_IAT_Mean': np.random.normal(loc=100, scale=20, size=n_benign),
        'Fwd_IAT_Mean': np.random.normal(loc=120, scale=25, size=n_benign),
        'Bwd_IAT_Mean': np.random.normal(loc=80, scale=15, size=n_benign),
        'Packet_Length_Mean': np.random.normal(loc=500, scale=100, size=n_benign),
        'Label': ['BENIGN'] * n_benign
    }
    
    # Create attack traffic (anomalous patterns)
    # DoS attacks: high packet rate, short duration
    n_dos = n_attack // 2
    dos_data = {
        'Flow_Duration': np.random.exponential(scale=100, size=n_dos),  # Shorter
        'Total_Fwd_Packets': np.random.poisson(lam=100, size=n_dos),  # Much higher
        'Total_Backward_Packets': np.random.poisson(lam=2, size=n_dos),  # Lower
        'Flow_Bytes_per_s': np.random.normal(loc=50000, scale=10000, size=n_dos),  # Higher
        'Flow_Packets_per_s': np.random.normal(loc=500, scale=100, size=n_dos),  # Much higher
        'Flow_IAT_Mean': np.random.normal(loc=10, scale=5, size=n_dos),  # Lower
        'Fwd_IAT_Mean': np.random.normal(loc=5, scale=2, size=n_dos),  # Much lower
        'Bwd_IAT_Mean': np.random.normal(loc=200, scale=50, size=n_dos),  # Higher
        'Packet_Length_Mean': np.random.normal(loc=100, scale=20, size=n_dos),  # Smaller
        'Label': ['DoS'] * n_dos
    }
    
    # PortScan attacks: many connections, small packets
    n_portscan = n_attack - n_dos
    portscan_data = {
        'Flow_Duration': np.random.exponential(scale=50, size=n_portscan),  # Very short
        'Total_Fwd_Packets': np.random.poisson(lam=3, size=n_portscan),  # Few packets
        'Total_Backward_Packets': np.random.poisson(lam=1, size=n_portscan),  # Very few
        'Flow_Bytes_per_s': np.random.normal(loc=1000, scale=200, size=n_portscan),  # Low
        'Flow_Packets_per_s': np.random.normal(loc=20, scale=5, size=n_portscan),  # Low
        'Flow_IAT_Mean': np.random.normal(loc=50, scale=10, size=n_portscan),
        'Fwd_IAT_Mean': np.random.normal(loc=60, scale=15, size=n_portscan),
        'Bwd_IAT_Mean': np.random.normal(loc=40, scale=10, size=n_portscan),
        'Packet_Length_Mean': np.random.normal(loc=50, scale=10, size=n_portscan),  # Very small
        'Label': ['PortScan'] * n_portscan
    }
    
    # Combine all data
    benign_df = pd.DataFrame(benign_data)
    dos_df = pd.DataFrame(dos_data)
    portscan_df = pd.DataFrame(portscan_data)
    
    combined_df = pd.concat([benign_df, dos_df, portscan_df], ignore_index=True)
    
    # Shuffle the data
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_df


def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline with synthetic data."""
    
    print("=" * 70)
    print("Testing Preprocessing Pipeline with Synthetic Dataset")
    print("=" * 70)
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    df = create_synthetic_dataset(n_benign=100, n_attack=50)
    print(f"   Created dataset with {len(df)} samples")
    print(f"   - Benign: {(df['Label'] == 'BENIGN').sum()}")
    print(f"   - DoS: {(df['Label'] == 'DoS').sum()}")
    print(f"   - PortScan: {(df['Label'] == 'PortScan').sum()}")
    print(f"   - Features: {len(df.columns) - 1}")  # Exclude Label
    
    # Save to temporary CSV file
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "synthetic_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"   Saved to: {csv_path}")
        
        # Initialize preprocessing pipeline
        print("\n2. Initializing preprocessing pipeline...")
        config = {
            'random_state': 42,
            'test_size': 0.3,
            'val_size': 0.2,
            'min_samples': 10
        }
        pipeline = PreprocessingPipeline(config)
        print("   Pipeline initialized")
        
        # Load dataset
        print("\n3. Loading dataset...")
        loaded_df = pipeline.load_datasets([csv_path])
        print(f"   Loaded {len(loaded_df)} samples")
        print(f"   Columns: {list(loaded_df.columns)}")
        
        # Clean data
        print("\n4. Cleaning data...")
        print("   - Removing duplicates")
        print("   - Removing NaN values")
        print("   - Removing infinite values")
        print("   - Removing non-numeric features")
        cleaned_df = pipeline.clean_data(loaded_df)
        print(f"   Cleaned data: {len(cleaned_df)} samples remaining")
        
        # Verify data quality
        print("\n5. Verifying data quality...")
        
        # Check for duplicates
        has_duplicates = cleaned_df.duplicated().any()
        print(f"   ✓ No duplicates: {not has_duplicates}")
        assert not has_duplicates, "Data should not contain duplicates"
        
        # Check for NaN
        has_nan = cleaned_df.isnull().any().any()
        print(f"   ✓ No NaN values: {not has_nan}")
        assert not has_nan, "Data should not contain NaN values"
        
        # Check for infinite values
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        has_inf = np.isinf(cleaned_df[numeric_cols]).any().any()
        print(f"   ✓ No infinite values: {not has_inf}")
        assert not has_inf, "Data should not contain infinite values"
        
        # Check that all features (except Label) are numeric
        all_numeric = all(
            pd.api.types.is_numeric_dtype(cleaned_df[col]) 
            for col in cleaned_df.columns if col != 'Label'
        )
        print(f"   ✓ All features numeric: {all_numeric}")
        assert all_numeric, "All features should be numeric"
        
        # Split benign and attack
        print("\n6. Splitting benign and attack samples...")
        benign_df, attack_df = pipeline.split_benign_attack(cleaned_df)
        print(f"   Benign samples: {len(benign_df)}")
        print(f"   Attack samples: {len(attack_df)}")
        
        # Verify split
        assert len(benign_df) > 0, "Should have benign samples"
        assert len(attack_df) > 0, "Should have attack samples"
        assert all(benign_df['Label'].str.upper() == 'BENIGN'), "Benign df should only contain BENIGN"
        assert all(attack_df['Label'].str.upper() != 'BENIGN'), "Attack df should not contain BENIGN"
        print("   ✓ Split successful")
        
        # Normalize and split
        print("\n7. Normalizing and splitting data...")
        result = pipeline.normalize_and_split(benign_df, attack_df)
        
        print(f"   Training (benign only): {len(result['X_train_benign'])} samples")
        print(f"   Validation (benign only): {len(result['X_val_benign'])} samples")
        print(f"   Test (benign + attack): {len(result['X_test'])} samples")
        print(f"     - Benign: {(result['y_test'] == 0).sum()}")
        print(f"     - Attack: {(result['y_test'] == 1).sum()}")
        print(f"   Features: {len(result['feature_names'])}")
        
        # Verify normalization
        print("\n8. Verifying normalization...")
        train_mean = np.mean(result['X_train_benign'], axis=0)
        train_std = np.std(result['X_train_benign'], axis=0)
        
        max_mean_deviation = np.abs(train_mean).max()
        max_std_deviation = np.abs(train_std - 1.0).max()
        
        print(f"   Max mean deviation from 0: {max_mean_deviation:.6f}")
        print(f"   Max std deviation from 1: {max_std_deviation:.6f}")
        
        assert max_mean_deviation < 1e-10, "Training data mean should be close to 0"
        assert max_std_deviation < 0.1, "Training data std should be close to 1"
        print("   ✓ Normalization successful (mean ≈ 0, std ≈ 1)")
        
        # Verify stratified sampling
        print("\n9. Verifying stratified sampling...")
        total_original = len(benign_df) + len(attack_df)
        benign_ratio_original = len(benign_df) / total_original
        
        benign_count_test = (result['y_test'] == 0).sum()
        benign_ratio_test = benign_count_test / len(result['y_test'])
        
        ratio_diff = abs(benign_ratio_original - benign_ratio_test)
        print(f"   Original benign ratio: {benign_ratio_original:.3f}")
        print(f"   Test benign ratio: {benign_ratio_test:.3f}")
        print(f"   Difference: {ratio_diff:.3f}")
        
        assert ratio_diff < 0.05, "Class proportions should be maintained"
        print("   ✓ Stratified sampling successful (within 5% tolerance)")
        
        # Verify test set contains both classes
        print("\n10. Verifying test set composition...")
        has_benign = 0 in result['y_test']
        has_attack = 1 in result['y_test']
        print(f"   ✓ Test set contains benign samples: {has_benign}")
        print(f"   ✓ Test set contains attack samples: {has_attack}")
        assert has_benign and has_attack, "Test set should contain both classes"
        
        # Summary
        print("\n" + "=" * 70)
        print("PREPROCESSING PIPELINE VERIFICATION COMPLETE")
        print("=" * 70)
        print("\n✓ All data quality checks passed!")
        print("✓ All preprocessing steps completed successfully!")
        print("\nPipeline is ready for model training.")
        print("=" * 70)


if __name__ == "__main__":
    test_preprocessing_pipeline()
