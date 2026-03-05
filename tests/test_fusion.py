"""
Unit and property-based tests for FusionModule class.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fusion import FusionModule


class TestFusionModuleInit:
    """Test FusionModule initialization."""
    
    def test_init_basic(self):
        """Test basic initialization with valid parameters."""
        config = {
            'weight_autoencoder': 0.5,
            'weight_isolation': 0.5,
            'percentile': 95
        }
        
        fusion = FusionModule(config)
        
        assert fusion.weight_autoencoder == 0.5
        assert fusion.weight_isolation == 0.5
        assert fusion.percentile == 95
        assert fusion.threshold is None  # Not fitted yet
    
    def test_init_with_defaults(self):
        """Test initialization with minimal config (using defaults)."""
        config = {}
        
        fusion = FusionModule(config)
        
        assert fusion.weight_autoencoder == 0.5  # default
        assert fusion.weight_isolation == 0.5  # default
        assert fusion.percentile == 95  # default
    
    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        config = {
            'weight_autoencoder': 0.7,
            'weight_isolation': 0.3,
            'percentile': 99
        }
        
        fusion = FusionModule(config)
        
        assert fusion.weight_autoencoder == 0.7
        assert fusion.weight_isolation == 0.3
        assert fusion.percentile == 99
    
    def test_init_weights_must_sum_to_one(self):
        """Test that weights must sum to 1.0."""
        config = {
            'weight_autoencoder': 0.6,
            'weight_isolation': 0.5  # Sum = 1.1
        }
        
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            FusionModule(config)
    
    def test_init_invalid_percentile_too_low(self):
        """Test that percentile must be > 0."""
        config = {'percentile': 0}
        
        with pytest.raises(ValueError, match="Percentile must be in range"):
            FusionModule(config)
    
    def test_init_invalid_percentile_too_high(self):
        """Test that percentile must be < 100."""
        config = {'percentile': 100}
        
        with pytest.raises(ValueError, match="Percentile must be in range"):
            FusionModule(config)
    
    def test_init_invalid_percentile_negative(self):
        """Test that percentile cannot be negative."""
        config = {'percentile': -10}
        
        with pytest.raises(ValueError, match="Percentile must be in range"):
            FusionModule(config)


class TestFusionModuleFitThreshold:
    """Test FusionModule fit_threshold method."""
    
    def test_fit_threshold_basic(self):
        """Test basic threshold fitting."""
        config = {'percentile': 95}
        fusion = FusionModule(config)
        
        # Generate benign validation data
        np.random.seed(42)
        recon_errors = np.random.rand(100)
        iso_scores = np.random.rand(100)
        
        # Fit threshold
        fusion.fit_threshold(recon_errors, iso_scores)
        
        # Verify statistics are computed
        assert fusion.recon_min is not None
        assert fusion.recon_max is not None
        assert fusion.iso_min is not None
        assert fusion.iso_max is not None
        assert fusion.threshold is not None
    
    def test_fit_threshold_empty_recon_errors_raises_error(self):
        """Test that empty recon_errors raises ValueError."""
        config = {}
        fusion = FusionModule(config)
        
        with pytest.raises(ValueError, match="recon_errors_benign cannot be empty"):
            fusion.fit_threshold(np.array([]), np.random.rand(100))
    
    def test_fit_threshold_empty_iso_scores_raises_error(self):
        """Test that empty iso_scores raises ValueError."""
        config = {}
        fusion = FusionModule(config)
        
        with pytest.raises(ValueError, match="iso_scores_benign cannot be empty"):
            fusion.fit_threshold(np.random.rand(100), np.array([]))
    
    def test_fit_threshold_mismatched_lengths_raises_error(self):
        """Test that mismatched input lengths raise ValueError."""
        config = {}
        fusion = FusionModule(config)
        
        with pytest.raises(ValueError, match="Input lengths must match"):
            fusion.fit_threshold(np.random.rand(100), np.random.rand(50))
    
    def test_fit_threshold_nan_in_recon_errors_raises_error(self):
        """Test that NaN in recon_errors raises ValueError."""
        config = {}
        fusion = FusionModule(config)
        
        recon_errors = np.random.rand(100)
        recon_errors[10] = np.nan
        iso_scores = np.random.rand(100)
        
        with pytest.raises(ValueError, match="recon_errors_benign contains NaN"):
            fusion.fit_threshold(recon_errors, iso_scores)
    
    def test_fit_threshold_inf_in_iso_scores_raises_error(self):
        """Test that inf in iso_scores raises ValueError."""
        config = {}
        fusion = FusionModule(config)
        
        recon_errors = np.random.rand(100)
        iso_scores = np.random.rand(100)
        iso_scores[10] = np.inf
        
        with pytest.raises(ValueError, match="iso_scores_benign contains NaN or infinite"):
            fusion.fit_threshold(recon_errors, iso_scores)


class TestFusionModuleNormalizeScores:
    """Test FusionModule normalize_scores method."""
    
    def test_normalize_scores_basic(self):
        """Test basic score normalization."""
        config = {}
        fusion = FusionModule(config)
        
        # Fit threshold first
        np.random.seed(42)
        recon_benign = np.random.rand(100)
        iso_benign = np.random.rand(100)
        fusion.fit_threshold(recon_benign, iso_benign)
        
        # Normalize test scores
        recon_test = np.random.rand(50)
        iso_test = np.random.rand(50)
        
        recon_norm, iso_norm = fusion.normalize_scores(recon_test, iso_test)
        
        # Verify output shape
        assert recon_norm.shape == (50,)
        assert iso_norm.shape == (50,)
        
        # Verify values are in [0, 1] range
        assert np.all(recon_norm >= 0) and np.all(recon_norm <= 1)
        assert np.all(iso_norm >= 0) and np.all(iso_norm <= 1)
    
    def test_normalize_scores_without_fit_raises_error(self):
        """Test that normalizing without fitting raises ValueError."""
        config = {}
        fusion = FusionModule(config)
        
        recon_test = np.random.rand(50)
        iso_test = np.random.rand(50)
        
        with pytest.raises(ValueError, match="Normalization statistics not computed"):
            fusion.normalize_scores(recon_test, iso_test)
    
    def test_normalize_scores_clips_to_unit_range(self):
        """Test that scores outside benign range are clipped to [0, 1]."""
        config = {}
        fusion = FusionModule(config)
        
        # Fit with benign data in range [0.4, 0.6]
        np.random.seed(42)
        recon_benign = np.random.uniform(0.4, 0.6, size=100)
        iso_benign = np.random.uniform(0.4, 0.6, size=100)
        fusion.fit_threshold(recon_benign, iso_benign)
        
        # Test with data outside benign range
        recon_test = np.array([0.1, 0.5, 0.9])  # Below, within, above
        iso_test = np.array([0.1, 0.5, 0.9])
        
        recon_norm, iso_norm = fusion.normalize_scores(recon_test, iso_test)
        
        # Verify clipping
        assert np.all(recon_norm >= 0) and np.all(recon_norm <= 1)
        assert np.all(iso_norm >= 0) and np.all(iso_norm <= 1)


class TestFusionModuleComputeCombinedScore:
    """Test FusionModule compute_combined_score method."""
    
    def test_compute_combined_score_basic(self):
        """Test basic combined score computation."""
        config = {
            'weight_autoencoder': 0.5,
            'weight_isolation': 0.5
        }
        fusion = FusionModule(config)
        
        # Fit threshold first
        np.random.seed(42)
        recon_benign = np.random.rand(100)
        iso_benign = np.random.rand(100)
        fusion.fit_threshold(recon_benign, iso_benign)
        
        # Compute combined scores
        recon_test = np.random.rand(50)
        iso_test = np.random.rand(50)
        
        combined = fusion.compute_combined_score(recon_test, iso_test)
        
        # Verify output shape
        assert combined.shape == (50,)
        
        # Verify values are in [0, 1] range
        assert np.all(combined >= 0) and np.all(combined <= 1)
    
    def test_compute_combined_score_weighted_average(self):
        """Test that combined score is weighted average."""
        config = {
            'weight_autoencoder': 0.7,
            'weight_isolation': 0.3
        }
        fusion = FusionModule(config)
        
        # Fit with simple benign data
        recon_benign = np.array([0.0, 1.0])
        iso_benign = np.array([0.0, 1.0])
        fusion.fit_threshold(recon_benign, iso_benign)
        
        # Test with known values
        recon_test = np.array([0.0, 1.0])
        iso_test = np.array([0.0, 1.0])
        
        combined = fusion.compute_combined_score(recon_test, iso_test)
        
        # For normalized scores [0, 1], combined should be weighted average
        # combined[0] = 0.7 * 0 + 0.3 * 0 = 0
        # combined[1] = 0.7 * 1 + 0.3 * 1 = 1
        assert np.isclose(combined[0], 0.0)
        assert np.isclose(combined[1], 1.0)


class TestFusionModuleClassify:
    """Test FusionModule classify method."""
    
    def test_classify_basic(self):
        """Test basic classification."""
        config = {'percentile': 95}
        fusion = FusionModule(config)
        
        # Fit threshold
        np.random.seed(42)
        recon_benign = np.random.rand(100)
        iso_benign = np.random.rand(100)
        fusion.fit_threshold(recon_benign, iso_benign)
        
        # Classify test scores
        combined_scores = np.array([0.3, 0.5, 0.7, 0.9])
        predictions = fusion.classify(combined_scores)
        
        # Verify output shape and type
        assert predictions.shape == (4,)
        assert predictions.dtype == int
        
        # Verify binary predictions
        assert np.all((predictions == 0) | (predictions == 1))
    
    def test_classify_without_fit_raises_error(self):
        """Test that classifying without fitting raises ValueError."""
        config = {}
        fusion = FusionModule(config)
        
        combined_scores = np.array([0.5, 0.7])
        
        with pytest.raises(ValueError, match="Threshold not computed"):
            fusion.classify(combined_scores)
    
    def test_classify_threshold_rule(self):
        """Test that classification follows threshold rule."""
        config = {'percentile': 50}  # Median
        fusion = FusionModule(config)
        
        # Fit with known benign data
        recon_benign = np.linspace(0, 1, 100)
        iso_benign = np.linspace(0, 1, 100)
        fusion.fit_threshold(recon_benign, iso_benign)
        
        # Test with scores below and above threshold
        threshold = fusion.threshold
        combined_scores = np.array([threshold - 0.1, threshold + 0.1])
        predictions = fusion.classify(combined_scores)
        
        # Below threshold should be 0, above should be 1
        assert predictions[0] == 0
        assert predictions[1] == 1


class TestFusionModulePropertyBased:
    """Property-based tests for FusionModule using Hypothesis."""
    
    def test_property_10_score_normalization_to_unit_range(self):
        """
        Property 10: Score Normalization to Unit Range
        
        Validates Requirements 5.1, 5.2: Score normalization
        
        Property: For any set of reconstruction errors and isolation scores,
        the fusion module's normalization should transform them to the range [0, 1]
        using min-max scaling based on benign validation statistics.
        
        **Validates: Requirements 5.1, 5.2**
        """
        from hypothesis import given, settings, strategies as st
        
        @given(
            n_benign=st.integers(min_value=10, max_value=200),
            n_test=st.integers(min_value=5, max_value=100),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=100, deadline=None)
        def property_test(n_benign, n_test, data_seed):
            # Configure fusion module
            config = {
                'weight_autoencoder': 0.5,
                'weight_isolation': 0.5,
                'percentile': 95
            }
            
            fusion = FusionModule(config)
            
            # Generate benign validation data
            np.random.seed(data_seed)
            recon_benign = np.random.rand(n_benign).astype(np.float64)
            iso_benign = np.random.rand(n_benign).astype(np.float64)
            
            # Fit threshold
            fusion.fit_threshold(recon_benign, iso_benign)
            
            # Generate test data
            recon_test = np.random.rand(n_test).astype(np.float64)
            iso_test = np.random.rand(n_test).astype(np.float64)
            
            # Normalize scores
            recon_norm, iso_norm = fusion.normalize_scores(recon_test, iso_test)
            
            # Property assertions
            assert recon_norm.shape == (n_test,), \
                f"Expected shape ({n_test},), got {recon_norm.shape}"
            
            assert iso_norm.shape == (n_test,), \
                f"Expected shape ({n_test},), got {iso_norm.shape}"
            
            assert np.all(recon_norm >= 0) and np.all(recon_norm <= 1), \
                f"Normalized reconstruction errors must be in [0, 1], got range [{np.min(recon_norm)}, {np.max(recon_norm)}]"
            
            assert np.all(iso_norm >= 0) and np.all(iso_norm <= 1), \
                f"Normalized isolation scores must be in [0, 1], got range [{np.min(iso_norm)}, {np.max(iso_norm)}]"
            
            assert np.all(np.isfinite(recon_norm)), \
                "Normalized reconstruction errors must be finite"
            
            assert np.all(np.isfinite(iso_norm)), \
                "Normalized isolation scores must be finite"
        
        # Run the property test
        property_test()
    
    def test_property_11_weighted_average_combination(self):
        """
        Property 11: Weighted Average Combination
        
        Validates Requirement 5.3: Weighted combination
        
        Property: For any normalized scores and weights that sum to 1.0,
        the combined score should equal the weighted average:
        combined = w_ae * recon_norm + w_if * iso_norm
        
        **Validates: Requirements 5.3**
        """
        from hypothesis import given, settings, strategies as st, assume
        
        @given(
            n_samples=st.integers(min_value=10, max_value=100),
            weight_ae=st.floats(min_value=0.1, max_value=0.9),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=100, deadline=None)
        def property_test(n_samples, weight_ae, data_seed):
            # Compute complementary weight
            weight_if = 1.0 - weight_ae
            
            # Configure fusion module
            config = {
                'weight_autoencoder': weight_ae,
                'weight_isolation': weight_if,
                'percentile': 95
            }
            
            fusion = FusionModule(config)
            
            # Generate benign validation data
            np.random.seed(data_seed)
            recon_benign = np.random.rand(n_samples).astype(np.float64)
            iso_benign = np.random.rand(n_samples).astype(np.float64)
            
            # Fit threshold
            fusion.fit_threshold(recon_benign, iso_benign)
            
            # Generate test data
            recon_test = np.random.rand(n_samples).astype(np.float64)
            iso_test = np.random.rand(n_samples).astype(np.float64)
            
            # Compute combined scores
            combined = fusion.compute_combined_score(recon_test, iso_test)
            
            # Manually compute normalized scores and weighted average
            recon_norm, iso_norm = fusion.normalize_scores(recon_test, iso_test)
            expected_combined = weight_ae * recon_norm + weight_if * iso_norm
            
            # Property assertion: combined score equals weighted average
            assert np.allclose(combined, expected_combined, rtol=1e-10, atol=1e-12), \
                f"Combined score must equal weighted average. Max diff: {np.max(np.abs(combined - expected_combined))}"
        
        # Run the property test
        property_test()
    
    def test_property_12_percentile_based_threshold(self):
        """
        Property 12: Percentile-Based Threshold
        
        Validates Requirements 5.4, 5.5: Threshold computation
        
        Property: For any benign validation set and percentile value,
        the computed threshold should equal the specified percentile of
        the combined scores on the benign validation set.
        
        **Validates: Requirements 5.4, 5.5**
        """
        from hypothesis import given, settings, strategies as st
        
        @given(
            n_benign=st.integers(min_value=20, max_value=200),
            percentile=st.integers(min_value=90, max_value=99),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=100, deadline=None)
        def property_test(n_benign, percentile, data_seed):
            # Configure fusion module
            config = {
                'weight_autoencoder': 0.5,
                'weight_isolation': 0.5,
                'percentile': percentile
            }
            
            fusion = FusionModule(config)
            
            # Generate benign validation data
            np.random.seed(data_seed)
            recon_benign = np.random.rand(n_benign).astype(np.float64)
            iso_benign = np.random.rand(n_benign).astype(np.float64)
            
            # Fit threshold
            fusion.fit_threshold(recon_benign, iso_benign)
            
            # Manually compute combined scores on benign validation set
            combined_benign = fusion.compute_combined_score(recon_benign, iso_benign)
            
            # Manually compute percentile
            expected_threshold = np.percentile(combined_benign, percentile)
            
            # Property assertion: threshold equals percentile
            assert np.isclose(fusion.threshold, expected_threshold, rtol=1e-10, atol=1e-12), \
                f"Threshold must equal {percentile}th percentile. Got {fusion.threshold}, expected {expected_threshold}"
        
        # Run the property test
        property_test()
    
    def test_property_13_threshold_based_classification(self):
        """
        Property 13: Threshold-Based Classification
        
        Validates Requirement 5.6: Classification rule
        
        Property: For any combined score and threshold,
        the classification should follow the rule:
        - prediction = 1 if combined_score > threshold
        - prediction = 0 if combined_score <= threshold
        
        **Validates: Requirements 5.6**
        """
        from hypothesis import given, settings, strategies as st
        
        @given(
            n_benign=st.integers(min_value=20, max_value=100),
            n_test=st.integers(min_value=10, max_value=50),
            data_seed=st.integers(min_value=0, max_value=10000)
        )
        @settings(max_examples=100, deadline=None)
        def property_test(n_benign, n_test, data_seed):
            # Configure fusion module
            config = {
                'weight_autoencoder': 0.5,
                'weight_isolation': 0.5,
                'percentile': 95
            }
            
            fusion = FusionModule(config)
            
            # Generate benign validation data
            np.random.seed(data_seed)
            recon_benign = np.random.rand(n_benign).astype(np.float64)
            iso_benign = np.random.rand(n_benign).astype(np.float64)
            
            # Fit threshold
            fusion.fit_threshold(recon_benign, iso_benign)
            
            # Generate test data
            recon_test = np.random.rand(n_test).astype(np.float64)
            iso_test = np.random.rand(n_test).astype(np.float64)
            
            # Compute combined scores
            combined = fusion.compute_combined_score(recon_test, iso_test)
            
            # Classify
            predictions = fusion.classify(combined)
            
            # Property assertion: classification follows threshold rule
            for i in range(n_test):
                if combined[i] > fusion.threshold:
                    assert predictions[i] == 1, \
                        f"Sample {i}: combined={combined[i]} > threshold={fusion.threshold}, but prediction={predictions[i]}"
                else:
                    assert predictions[i] == 0, \
                        f"Sample {i}: combined={combined[i]} <= threshold={fusion.threshold}, but prediction={predictions[i]}"
        
        # Run the property test
        property_test()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
