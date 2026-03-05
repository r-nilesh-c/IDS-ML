"""
Preprocessing pipeline for network flow data.

This module handles loading, cleaning, and normalizing CIC-IDS datasets.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Preprocessing pipeline for CIC-IDS network flow datasets.
    
    Handles loading, merging, cleaning, and normalizing network flow data
    for anomaly detection training and evaluation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessing pipeline with configuration.
        
        Args:
            config: Dictionary containing paths, random seed, test split ratio
        """
        self.config = config
        self.random_state = config.get('random_state', 42)
        logger.info("PreprocessingPipeline initialized")

    def _normalize_label_value(self, label: str) -> str:
        """Normalize raw dataset label variants into canonical class names."""
        if pd.isna(label):
            return 'UNKNOWN'

        text = str(label).strip()
        text_clean = text.replace('∩┐╜', '-').replace('�', '-').replace('–', '-').strip()
        lower = text_clean.lower()

        if lower == 'benign':
            return 'BENIGN'

        if 'web attack' in lower:
            if 'brute' in lower:
                return 'Web Attack - Brute Force'
            if 'xss' in lower:
                return 'Web Attack - XSS'
            if 'sql' in lower:
                return 'Web Attack - SQL Injection'
            return 'Web Attack'

        if 'ddos' in lower or 'hoic' in lower or 'loic' in lower:
            return 'DDoS'

        if 'hulk' in lower:
            return 'DoS Hulk'
        if 'goldeneye' in lower:
            return 'DoS GoldenEye'
        if 'slowhttptest' in lower:
            return 'DoS Slowhttptest'
        if 'slowloris' in lower:
            return 'DoS slowloris'

        if 'portscan' in lower:
            return 'PortScan'
        if 'infil' in lower:
            return 'Infiltration'
        if 'bot' in lower:
            return 'Bot'
        if 'ftp' in lower:
            return 'FTP-Patator'
        if 'ssh' in lower:
            return 'SSH-Patator'
        if 'heartbleed' in lower:
            return 'Heartbleed'

        return text_clean
    
    def load_datasets(self, paths: List[str]) -> pd.DataFrame:
        """
        Load and merge multiple CSV datasets.
        
        Handles common label column name variations and provides error handling
        for missing files and corrupted data.
        
        Args:
            paths: List of file paths to CIC-IDS datasets
            
        Returns:
            Merged DataFrame with all samples
            
        Raises:
            FileNotFoundError: If any dataset file is missing
            ValueError: If datasets cannot be loaded or merged
        """
        if not paths:
            raise ValueError("No dataset paths provided")
        
        dataframes = []
        
        for path in paths:
            # Validate file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset file not found: {path}")
            
            try:
                logger.info(f"Loading dataset from {path}")
                
                # Determine file format and load accordingly
                if path.endswith('.parquet'):
                    df = pd.read_parquet(path)
                    logger.info(f"Successfully loaded {path} as parquet")
                else:
                    # Try loading CSV with different encodings to handle corrupted data
                    df = None
                    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                        try:
                            df = pd.read_csv(path, encoding=encoding, low_memory=False)
                            logger.info(f"Successfully loaded {path} with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if df is None:
                        raise ValueError(f"Could not load {path} with any supported encoding")
                
                # Handle label column name variations
                # Common variations: " Label", "Label", "label", " label"
                label_variations = [' Label', 'Label', 'label', ' label']
                label_col = None
                
                for col_name in label_variations:
                    if col_name in df.columns:
                        label_col = col_name
                        break
                
                if label_col is None:
                    raise ValueError(f"No label column found in {path}. Expected one of: {label_variations}")
                
                # Standardize label column name to 'Label'
                if label_col != 'Label':
                    df = df.rename(columns={label_col: 'Label'})
                    logger.info(f"Renamed label column '{label_col}' to 'Label'")
                
                logger.info(f"Loaded {len(df)} samples from {path}")
                dataframes.append(df)
                
            except pd.errors.EmptyDataError:
                raise ValueError(f"Dataset file is empty: {path}")
            except pd.errors.ParserError as e:
                raise ValueError(f"Failed to parse CSV file {path}: {str(e)}")
            except Exception as e:
                raise ValueError(f"Error loading dataset {path}: {str(e)}")
        
        # Merge all dataframes
        if len(dataframes) == 1:
            merged_df = dataframes[0]
        else:
            try:
                merged_df = pd.concat(dataframes, ignore_index=True)
                logger.info(f"Merged {len(dataframes)} datasets into {len(merged_df)} total samples")
            except Exception as e:
                raise ValueError(f"Failed to merge datasets: {str(e)}")
        
        # Validate merged dataframe
        if merged_df.empty:
            raise ValueError("Merged dataset is empty")
        
        if 'Label' not in merged_df.columns:
            raise ValueError("Label column missing after merge")
        
        logger.info(f"Successfully loaded and merged datasets: {len(merged_df)} total samples")
        return merged_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates, NaN, inf values, and non-numeric features.
        
        Args:
            df: Raw merged DataFrame
            
        Returns:
            Cleaned DataFrame with only numeric features
            
        Raises:
            ValueError: If insufficient samples remain after cleaning
        """
        if df.empty:
            raise ValueError("Cannot clean empty DataFrame")
        
        initial_count = len(df)
        logger.info(f"Starting data cleaning with {initial_count} samples")
        
        # Store label column before cleaning
        if 'Label' not in df.columns:
            raise ValueError("Label column missing from DataFrame")
        
        labels = df['Label'].copy()

        # Normalize labels to reduce class fragmentation across datasets
        df = df.copy()
        df['Label'] = df['Label'].apply(self._normalize_label_value)
        
        # Remove duplicate rows
        df_no_duplicates = df.drop_duplicates()
        duplicates_removed = initial_count - len(df_no_duplicates)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Remove rows with NaN values
        df_no_nan = df_no_duplicates.dropna()
        nan_removed = len(df_no_duplicates) - len(df_no_nan)
        if nan_removed > 0:
            logger.info(f"Removed {nan_removed} rows with NaN values")
        
        # Remove rows with infinite values
        # Check for inf in numeric columns only
        numeric_cols = df_no_nan.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(df_no_nan[numeric_cols]).any(axis=1)
        df_no_inf = df_no_nan[~inf_mask]
        inf_removed = len(df_no_nan) - len(df_no_inf)
        if inf_removed > 0:
            logger.info(f"Removed {inf_removed} rows with infinite values")
        
        # Identify and remove non-numeric features (except Label)
        non_numeric_cols = []
        for col in df_no_inf.columns:
            if col == 'Label':
                continue
            if not pd.api.types.is_numeric_dtype(df_no_inf[col]):
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            logger.info(f"Removing {len(non_numeric_cols)} non-numeric features: {non_numeric_cols}")
            df_cleaned = df_no_inf.drop(columns=non_numeric_cols)
        else:
            df_cleaned = df_no_inf
        
        # Validate sufficient samples remain
        final_count = len(df_cleaned)
        min_samples = self.config.get('min_samples', 100)
        
        if final_count < min_samples:
            raise ValueError(
                f"Insufficient samples after cleaning: {final_count} remaining, "
                f"minimum required: {min_samples}"
            )
        
        samples_removed = initial_count - final_count
        removal_percentage = (samples_removed / initial_count) * 100
        
        logger.info(
            f"Data cleaning complete: {final_count} samples remaining "
            f"({samples_removed} removed, {removal_percentage:.2f}%)"
        )
        
        # Verify Label column is still present
        if 'Label' not in df_cleaned.columns:
            raise ValueError("Label column was removed during cleaning")
        
        return df_cleaned
    
    def remove_outliers(self, df: pd.DataFrame, method='iqr', threshold=3.0) -> pd.DataFrame:
        """
        Remove outliers from numeric features using IQR or Z-score method.
        
        Args:
            df: DataFrame with numeric features
            method: 'iqr' for Interquartile Range or 'zscore' for Z-score method
            threshold: Multiplier for IQR (default 3.0) or Z-score threshold
            
        Returns:
            DataFrame with outliers removed
            
        Raises:
            ValueError: If invalid method specified
        """
        if df.empty:
            raise ValueError("Cannot remove outliers from empty DataFrame")
        
        if method not in ['iqr', 'zscore']:
            raise ValueError(f"Invalid method: {method}. Must be 'iqr' or 'zscore'")
        
        initial_count = len(df)
        logger.info(f"Removing outliers using {method} method (threshold={threshold})")
        
        # Get numeric columns (exclude Label)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' in numeric_cols:
            numeric_cols.remove('Label')
        
        # Create mask for rows to keep
        keep_mask = pd.Series([True] * len(df), index=df.index)
        
        outliers_per_feature = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                # IQR method: Q1 - threshold*IQR to Q3 + threshold*IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR == 0:
                    # Skip features with zero IQR (constant values)
                    continue
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                
            elif method == 'zscore':
                # Z-score method: |z| < threshold
                mean = df[col].mean()
                std = df[col].std()
                
                if std == 0:
                    # Skip features with zero std (constant values)
                    continue
                
                z_scores = np.abs((df[col] - mean) / std)
                col_mask = z_scores < threshold
            
            # Count outliers for this feature
            outliers_count = (~col_mask).sum()
            if outliers_count > 0:
                outliers_per_feature[col] = outliers_count
            
            # Update overall mask
            keep_mask = keep_mask & col_mask
        
        # Apply mask
        df_no_outliers = df[keep_mask].copy()
        
        outliers_removed = initial_count - len(df_no_outliers)
        removal_percentage = (outliers_removed / initial_count) * 100
        
        logger.info(
            f"Outlier removal complete: {len(df_no_outliers)} samples remaining "
            f"({outliers_removed} removed, {removal_percentage:.2f}%)"
        )
        
        # Log top features with most outliers
        if outliers_per_feature:
            top_outlier_features = sorted(
                outliers_per_feature.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            logger.info("Top features with outliers removed:")
            for feat, count in top_outlier_features:
                pct = (count / initial_count) * 100
                logger.info(f"  {feat}: {count} ({pct:.1f}%)")
        
        return df_no_outliers
    
    def split_benign_attack(self, df: pd.DataFrame) -> tuple:
        """
        Separate benign and attack samples based on label column.
        
        Args:
            df: Cleaned DataFrame with label column
            
        Returns:
            Tuple of (benign_df, attack_df)
            
        Raises:
            ValueError: If Label column is missing or if no benign/attack samples exist
        """
        if df.empty:
            raise ValueError("Cannot split empty DataFrame")
        
        if 'Label' not in df.columns:
            raise ValueError("Label column missing from DataFrame")
        
        # Identify benign samples (case-insensitive)
        benign_mask = df['Label'].str.upper() == 'BENIGN'
        benign_df = df[benign_mask].copy()
        attack_df = df[~benign_mask].copy()
        
        # Validate that both benign and attack samples exist
        benign_count = len(benign_df)
        attack_count = len(attack_df)
        
        if benign_count == 0:
            raise ValueError(
                "No benign samples found in dataset. "
                "At least one benign sample is required for training."
            )
        
        if attack_count == 0:
            raise ValueError(
                "No attack samples found in dataset. "
                "At least one attack sample is required for evaluation."
            )
        
        # Log sample counts for each category
        logger.info(f"Split dataset into {benign_count} benign and {attack_count} attack samples")
        
        # Log attack type distribution
        if attack_count > 0:
            attack_types = attack_df['Label'].value_counts()
            logger.info(f"Attack type distribution:")
            for attack_type, count in attack_types.items():
                logger.info(f"  {attack_type}: {count} samples")
        
        return benign_df, attack_df
    
    def select_features(self, benign_df: pd.DataFrame, attack_df: pd.DataFrame, 
                       n_features: int = 30, method: str = 'variance') -> tuple:
        """
        Select most important features using variance or statistical methods.
        
        Args:
            benign_df: DataFrame containing benign samples
            attack_df: DataFrame containing attack samples
            n_features: Number of features to select (default: 30)
            method: 'variance' for variance threshold, 'statistical' for F-test
            
        Returns:
            Tuple of (benign_df_selected, attack_df_selected, selected_features)
            
        Raises:
            ValueError: If invalid method or insufficient features
        """
        if benign_df.empty or attack_df.empty:
            raise ValueError("Cannot select features from empty DataFrame")
        
        if method not in ['variance', 'statistical']:
            raise ValueError(f"Invalid method: {method}. Must be 'variance' or 'statistical'")
        
        logger.info(f"Selecting top {n_features} features using {method} method")
        
        # Get features (exclude Label)
        benign_features = benign_df.drop(columns=['Label'])
        attack_features = attack_df.drop(columns=['Label'])
        
        initial_feature_count = len(benign_features.columns)
        
        if n_features >= initial_feature_count:
            logger.warning(
                f"Requested {n_features} features but only {initial_feature_count} available. "
                f"Keeping all features."
            )
            return benign_df, attack_df, list(benign_features.columns)
        
        if method == 'variance':
            # Remove low-variance features first
            selector = VarianceThreshold(threshold=0.01)
            
            # Fit on benign data only
            selector.fit(benign_features)
            
            # Get feature mask
            feature_mask = selector.get_support()
            selected_feature_names = benign_features.columns[feature_mask].tolist()
            
            # If still too many features, select top N by variance
            if len(selected_feature_names) > n_features:
                variances = benign_features[selected_feature_names].var()
                top_features = variances.nlargest(n_features).index.tolist()
                selected_feature_names = top_features
        
        elif method == 'statistical':
            # Use F-test to select features that best discriminate benign vs attack
            # Combine data for feature selection
            X_combined = pd.concat([benign_features, attack_features], ignore_index=True)
            y_combined = np.concatenate([
                np.zeros(len(benign_features)),
                np.ones(len(attack_features))
            ])
            
            # Select K best features
            selector = SelectKBest(f_classif, k=n_features)
            selector.fit(X_combined, y_combined)
            
            # Get selected feature names
            feature_mask = selector.get_support()
            selected_feature_names = benign_features.columns[feature_mask].tolist()
        
        logger.info(f"Selected {len(selected_feature_names)} features from {initial_feature_count}")
        logger.info(f"Top 10 selected features: {selected_feature_names[:10]}")
        
        # Apply selection to both dataframes
        benign_df_selected = benign_df[selected_feature_names + ['Label']].copy()
        attack_df_selected = attack_df[selected_feature_names + ['Label']].copy()
        
        return benign_df_selected, attack_df_selected, selected_feature_names

    def normalize_and_split(self, benign_df: pd.DataFrame, attack_df: pd.DataFrame) -> Dict:
        """
        Normalize features and create train/validation/test splits.
        
        This method:
        1. Splits benign data into train and validation sets (80/20)
        2. Combines remaining benign with all attacks for test set
        3. Uses stratified sampling to maintain class distribution in test set
        4. Fits StandardScaler on training benign data only
        5. Applies normalization to all splits
        
        Args:
            benign_df: DataFrame containing only benign samples
            attack_df: DataFrame containing only attack samples
            
        Returns:
            Dictionary containing:
                - X_train_benign: Training features (benign only), normalized
                - X_val_benign: Validation features (benign only), normalized
                - X_test: Test features (benign + attacks), normalized
                - y_test: Test labels (0=benign, 1=attack)
                - y_test_labels: Test labels (original string labels)
                - scaler: Fitted StandardScaler object
                - feature_names: List of feature column names
                
        Raises:
            ValueError: If DataFrames are empty or invalid
        """
        if benign_df.empty:
            raise ValueError("Cannot normalize and split empty benign DataFrame")
        
        if attack_df.empty:
            raise ValueError("Cannot normalize and split empty attack DataFrame")
        
        if 'Label' not in benign_df.columns or 'Label' not in attack_df.columns:
            raise ValueError("Label column missing from DataFrame")
        
        logger.info("Starting normalization and splitting")
        
        # Get configuration parameters
        val_size = self.config.get('val_size', 0.2)
        test_size = self.config.get('test_size', 0.3)
        
        # Separate features and labels
        benign_features = benign_df.drop(columns=['Label'])
        benign_labels = benign_df['Label'].copy()
        
        attack_features = attack_df.drop(columns=['Label'])
        attack_labels = attack_df['Label'].copy()
        
        # Store feature names
        feature_names = list(benign_features.columns)
        logger.info(f"Number of features: {len(feature_names)}")
        
        # Split benign data into train and validation (80/20 of benign data)
        X_benign_train, X_benign_val, y_benign_train, y_benign_val = train_test_split(
            benign_features,
            benign_labels,
            test_size=val_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        logger.info(f"Benign train samples: {len(X_benign_train)}")
        logger.info(f"Benign validation samples: {len(X_benign_val)}")
        
        # For test set, we need to split all data (benign + attack) using stratified sampling
        # First, combine all data
        all_features = pd.concat([benign_features, attack_features], ignore_index=True)
        all_labels = pd.concat([benign_labels, attack_labels], ignore_index=True)
        
        # Create binary labels for stratification (0=benign, 1=attack)
        binary_labels = (all_labels.str.upper() != 'BENIGN').astype(int)
        
        # Stratified split to maintain class distribution
        # We'll use the remaining data after taking out training benign
        # Actually, we need to split all data into train+val and test
        # Then use only benign from train+val for training
        
        # Let's reconsider: we want test_size proportion of ALL data for testing
        # And we want stratified sampling to maintain attack type distribution
        
        # Split all data into train_val and test with stratification
        X_train_val, X_test, y_train_val, y_test_labels = train_test_split(
            all_features,
            all_labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=binary_labels,
            shuffle=True
        )
        
        # Create binary test labels
        y_test = (y_test_labels.str.upper() != 'BENIGN').astype(int)
        
        logger.info(f"Test samples: {len(X_test)} (benign: {(y_test == 0).sum()}, attack: {(y_test == 1).sum()})")
        
        # Now from train_val, extract only benign samples for training
        benign_mask = y_train_val.str.upper() == 'BENIGN'
        X_benign_all = X_train_val[benign_mask]
        
        # Split benign train_val into train and validation
        X_train_benign, X_val_benign = train_test_split(
            X_benign_all,
            test_size=val_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        logger.info(f"Final benign train samples: {len(X_train_benign)}")
        logger.info(f"Final benign validation samples: {len(X_val_benign)}")
        
        # Use RobustScaler instead of StandardScaler for better outlier handling
        # RobustScaler uses median and IQR, which are more robust to outliers
        use_robust_scaler = self.config.get('use_robust_scaler', True)
        
        if use_robust_scaler:
            scaler = RobustScaler()
            logger.info("Using RobustScaler for normalization (robust to outliers)")
        else:
            scaler = StandardScaler()
            logger.info("Using StandardScaler for normalization")
        
        X_train_benign_normalized = scaler.fit_transform(X_train_benign)
        
        logger.info(f"Scaler fitted on benign training data")
        
        # Apply normalization to validation and test sets
        X_val_benign_normalized = scaler.transform(X_val_benign)
        X_test_normalized = scaler.transform(X_test)
        
        # Verify normalization (training data should have mean≈0, std≈1)
        train_mean = np.mean(X_train_benign_normalized, axis=0)
        train_std = np.std(X_train_benign_normalized, axis=0)
        
        mean_check = np.abs(train_mean).max()
        std_check = np.abs(train_std - 1.0).max()
        
        logger.info(f"Normalization check - Max absolute mean: {mean_check:.6f}, Max std deviation from 1: {std_check:.6f}")
        
        if mean_check > 1e-10:
            logger.warning(f"Training data mean not close to zero: {mean_check}")
        
        if std_check > 0.1:
            logger.warning(f"Training data std not close to one: {std_check}")
        
        # Convert to numpy arrays if they aren't already
        X_train_benign_normalized = np.array(X_train_benign_normalized)
        X_val_benign_normalized = np.array(X_val_benign_normalized)
        X_test_normalized = np.array(X_test_normalized)
        y_test = np.array(y_test)
        y_test_labels = np.array(y_test_labels)
        
        # For Stage 2 supervised classifier, we need full labeled training data
        # Normalize the train_val data for Stage 2
        X_train_val_normalized = scaler.transform(X_train_val)
        y_train_val_binary = (y_train_val.str.upper() != 'BENIGN').astype(int)
        
        logger.info(f"Stage 2 training data: {len(X_train_val_normalized)} samples (benign: {(y_train_val_binary == 0).sum()}, attack: {(y_train_val_binary == 1).sum()})")
        
        logger.info("Normalization and splitting complete")
        
        return {
            'X_train_benign': X_train_benign_normalized,
            'X_val_benign': X_val_benign_normalized,
            'X_train': X_train_val_normalized,  # Full training data for Stage 2
            'y_train': np.array(y_train_val),  # String labels for Stage 2
            'y_train_binary': np.array(y_train_val_binary),  # Binary labels
            'X_test': X_test_normalized,
            'y_test': y_test,
            'y_test_labels': y_test_labels,
            'scaler': scaler,
            'feature_names': feature_names
        }
