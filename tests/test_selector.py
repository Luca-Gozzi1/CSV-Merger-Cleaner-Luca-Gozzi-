"""
Additional tests for the FeatureSelector class.

These tests cover methods that were not previously tested:
- get_feature_importance_ftest()
- get_feature_importance_mutual_info()
- get_combined_importance()
- select_top_k() with various methods
- get_correlation_with_target()

Run with: pytest tests/test_selector.py -v

Author: Luca Gozzi
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np

from src.features.selector import FeatureSelector


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample data for testing feature selection."""
    np.random.seed(42)
    n = 200
    
    # Create features with varying importance
    X = pd.DataFrame({
        # Highly predictive feature (correlated with target)
        "important_feature": np.random.randn(n) * 10,
        # Moderately predictive
        "moderate_feature": np.random.randn(n) * 5,
        # Low predictive power
        "weak_feature": np.random.randn(n),
        # Nearly constant (low variance)
        "low_variance_feature": np.ones(n) + np.random.randn(n) * 0.001,
        # Random noise
        "noise_feature": np.random.randn(n),
        # Categorical encoded
        "category_encoded": np.random.randint(0, 5, n).astype(float),
    })
    
    # Create target correlated with important_feature
    y = pd.Series(
        (X["important_feature"] + X["moderate_feature"] * 0.5 + np.random.randn(n) > 0).astype(int),
        name="target"
    )
    
    return X, y


@pytest.fixture
def selector(sample_data):
    """Create a FeatureSelector instance."""
    X, y = sample_data
    return FeatureSelector(X, y)


@pytest.fixture
def correlated_data():
    """Create data with highly correlated features."""
    np.random.seed(42)
    n = 200
    
    base = np.random.randn(n)
    
    X = pd.DataFrame({
        "feature_a": base,
        "feature_b": base + np.random.randn(n) * 0.01,  # Almost identical to a
        "feature_c": base * 2 + np.random.randn(n) * 0.1,  # Highly correlated
        "feature_d": np.random.randn(n),  # Independent
    })
    
    y = pd.Series((base > 0).astype(int), name="target")
    
    return X, y


# =============================================================================
# F-TEST IMPORTANCE TESTS
# =============================================================================

class TestFTestImportance:
    """Tests for get_feature_importance_ftest() method."""
    
    def test_ftest_returns_dataframe(self, selector):
        """Should return a DataFrame."""
        result = selector.get_feature_importance_ftest()
        
        assert isinstance(result, pd.DataFrame)
    
    def test_ftest_has_required_columns(self, selector):
        """Should have feature, f_score, p_value, rank columns."""
        result = selector.get_feature_importance_ftest()
        
        assert "feature" in result.columns
        assert "f_score" in result.columns
        assert "p_value" in result.columns
        assert "rank" in result.columns
    
    def test_ftest_has_significance_column(self, selector):
        """Should mark statistically significant features."""
        result = selector.get_feature_importance_ftest()
        
        assert "is_significant" in result.columns
        assert result["is_significant"].dtype == bool
    
    def test_ftest_scores_are_positive(self, selector):
        """F-scores should be non-negative."""
        result = selector.get_feature_importance_ftest()
        
        assert (result["f_score"] >= 0).all()
    
    def test_ftest_pvalues_valid_range(self, selector):
        """P-values should be between 0 and 1."""
        result = selector.get_feature_importance_ftest()
        
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()
    
    def test_ftest_ranks_are_sequential(self, selector):
        """Ranks should be sequential starting from 1."""
        result = selector.get_feature_importance_ftest()
        
        expected_ranks = list(range(1, len(result) + 1))
        assert result["rank"].tolist() == expected_ranks
    
    def test_ftest_sorted_by_score(self, selector):
        """Results should be sorted by f_score descending."""
        result = selector.get_feature_importance_ftest()
        
        scores = result["f_score"].tolist()
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# MUTUAL INFORMATION TESTS
# =============================================================================

class TestMutualInfoImportance:
    """Tests for get_feature_importance_mutual_info() method."""
    
    def test_mutual_info_returns_dataframe(self, selector):
        """Should return a DataFrame."""
        result = selector.get_feature_importance_mutual_info()
        
        assert isinstance(result, pd.DataFrame)
    
    def test_mutual_info_has_required_columns(self, selector):
        """Should have feature, mutual_info, rank columns."""
        result = selector.get_feature_importance_mutual_info()
        
        assert "feature" in result.columns
        assert "mutual_info" in result.columns
        assert "rank" in result.columns
    
    def test_mutual_info_scores_non_negative(self, selector):
        """Mutual information scores should be non-negative."""
        result = selector.get_feature_importance_mutual_info()
        
        assert (result["mutual_info"] >= 0).all()
    
    def test_mutual_info_sorted_by_score(self, selector):
        """Results should be sorted by mutual_info descending."""
        result = selector.get_feature_importance_mutual_info()
        
        scores = result["mutual_info"].tolist()
        assert scores == sorted(scores, reverse=True)
    
    def test_mutual_info_all_features_included(self, selector, sample_data):
        """All numeric features should be included."""
        X, _ = sample_data
        result = selector.get_feature_importance_mutual_info()
        
        # Should have same number of features as numeric columns
        assert len(result) == len(X.select_dtypes(include=[np.number]).columns)


# =============================================================================
# COMBINED IMPORTANCE TESTS
# =============================================================================

class TestCombinedImportance:
    """Tests for get_combined_importance() method."""
    
    def test_combined_returns_dataframe(self, selector):
        """Should return a DataFrame."""
        result = selector.get_combined_importance()
        
        assert isinstance(result, pd.DataFrame)
    
    def test_combined_has_all_methods(self, selector):
        """Should include importance from all methods."""
        result = selector.get_combined_importance()
        
        assert "rf_importance" in result.columns
        assert "f_score" in result.columns
        assert "mutual_info" in result.columns
    
    def test_combined_has_normalized_scores(self, selector):
        """Should have normalized scores (0-1 range)."""
        result = selector.get_combined_importance()
        
        for col in ["rf_importance_norm", "f_score_norm", "mutual_info_norm"]:
            if col in result.columns:
                assert (result[col] >= 0).all()
                assert (result[col] <= 1).all()
    
    def test_combined_has_average_importance(self, selector):
        """Should calculate average importance."""
        result = selector.get_combined_importance()
        
        assert "avg_importance" in result.columns
        assert (result["avg_importance"] >= 0).all()
        assert (result["avg_importance"] <= 1).all()
    
    def test_combined_sorted_by_average(self, selector):
        """Should be sorted by avg_importance descending."""
        result = selector.get_combined_importance()
        
        scores = result["avg_importance"].tolist()
        assert scores == sorted(scores, reverse=True)
    
    def test_combined_has_rank(self, selector):
        """Should include rank column."""
        result = selector.get_combined_importance()
        
        assert "rank" in result.columns
        expected_ranks = list(range(1, len(result) + 1))
        assert result["rank"].tolist() == expected_ranks


# =============================================================================
# SELECT TOP K TESTS (with different methods)
# =============================================================================

class TestSelectTopKMethods:
    """Tests for select_top_k() with various methods."""
    
    def test_select_top_k_ftest(self, selector):
        """Should work with f_test method."""
        result = selector.select_top_k(k=3, method="f_test")
        
        assert isinstance(result, list)
        assert len(result) == 3
    
    def test_select_top_k_mutual_info(self, selector):
        """Should work with mutual_info method."""
        result = selector.select_top_k(k=3, method="mutual_info")
        
        assert isinstance(result, list)
        assert len(result) == 3
    
    def test_select_top_k_combined(self, selector):
        """Should work with combined method."""
        result = selector.select_top_k(k=3, method="combined")
        
        assert isinstance(result, list)
        assert len(result) == 3
    
    def test_select_top_k_random_forest(self, selector):
        """Should work with random_forest method (default)."""
        result = selector.select_top_k(k=3, method="random_forest")
        
        assert isinstance(result, list)
        assert len(result) == 3
    
    def test_select_top_k_invalid_method(self, selector):
        """Should raise error for unknown method."""
        with pytest.raises(ValueError, match="Unknown method"):
            selector.select_top_k(k=3, method="invalid_method")
    
    def test_select_top_k_returns_strings(self, selector):
        """Should return list of feature names (strings)."""
        result = selector.select_top_k(k=3, method="f_test")
        
        assert all(isinstance(name, str) for name in result)
    
    def test_select_top_k_respects_k(self, selector):
        """Should return exactly k features."""
        for k in [1, 2, 4, 5]:
            result = selector.select_top_k(k=k, method="random_forest")
            assert len(result) == k
    
    def test_select_top_k_different_methods_may_differ(self, selector):
        """Different methods may select different features."""
        rf_result = selector.select_top_k(k=3, method="random_forest")
        ftest_result = selector.select_top_k(k=3, method="f_test")
        
        # They might be different (or same), just ensure they're valid
        assert len(rf_result) == 3
        assert len(ftest_result) == 3


# =============================================================================
# CORRELATION WITH TARGET TESTS
# =============================================================================

class TestCorrelationWithTarget:
    """Tests for get_correlation_with_target() method."""
    
    def test_correlation_returns_dataframe(self, selector):
        """Should return a DataFrame."""
        result = selector.get_correlation_with_target()
        
        assert isinstance(result, pd.DataFrame)
    
    def test_correlation_has_required_columns(self, selector):
        """Should have feature, correlation, abs_correlation columns."""
        result = selector.get_correlation_with_target()
        
        assert "feature" in result.columns
        assert "correlation" in result.columns
        assert "abs_correlation" in result.columns
    
    def test_correlation_valid_range(self, selector):
        """Correlations should be between -1 and 1."""
        result = selector.get_correlation_with_target()
        
        assert (result["correlation"] >= -1).all()
        assert (result["correlation"] <= 1).all()
    
    def test_abs_correlation_non_negative(self, selector):
        """Absolute correlations should be non-negative."""
        result = selector.get_correlation_with_target()
        
        assert (result["abs_correlation"] >= 0).all()
        assert (result["abs_correlation"] <= 1).all()
    
    def test_abs_correlation_matches_correlation(self, selector):
        """abs_correlation should be absolute value of correlation."""
        result = selector.get_correlation_with_target()
        
        expected_abs = result["correlation"].abs()
        pd.testing.assert_series_equal(
            result["abs_correlation"].reset_index(drop=True),
            expected_abs.reset_index(drop=True),
            check_names=False
        )
    
    def test_correlation_sorted_by_abs(self, selector):
        """Should be sorted by abs_correlation descending."""
        result = selector.get_correlation_with_target()
        
        abs_corrs = result["abs_correlation"].tolist()
        assert abs_corrs == sorted(abs_corrs, reverse=True)
    
    def test_important_feature_has_high_correlation(self, sample_data):
        """Important feature should have high correlation with target."""
        X, y = sample_data
        selector = FeatureSelector(X, y)
        
        result = selector.get_correlation_with_target()
        
        # The important_feature should be near the top
        important_rank = result[result["feature"] == "important_feature"].index[0]
        assert important_rank < 3  # Should be in top 3


# =============================================================================
# HIGH CORRELATION DETECTION TESTS
# =============================================================================

class TestHighCorrelationDetection:
    """Additional tests for remove_high_correlation()."""
    
    def test_detects_correlated_pairs(self, correlated_data):
        """Should detect highly correlated feature pairs."""
        X, y = correlated_data
        selector = FeatureSelector(X, y)
        
        # feature_a and feature_b are almost identical
        result = selector.remove_high_correlation(threshold=0.95)
        
        # Should find at least one highly correlated pair
        assert len(result) >= 1
    
    def test_returns_tuples(self, correlated_data):
        """Should return list of tuples."""
        X, y = correlated_data
        selector = FeatureSelector(X, y)
        
        result = selector.remove_high_correlation(threshold=0.9)
        
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 3  # (feature1, feature2, correlation)
    
    def test_correlation_above_threshold(self, correlated_data):
        """Returned correlations should be above threshold."""
        X, y = correlated_data
        selector = FeatureSelector(X, y)
        
        threshold = 0.9
        result = selector.remove_high_correlation(threshold=threshold)
        
        for feat1, feat2, corr in result:
            assert corr > threshold


# =============================================================================
# LOW VARIANCE DETECTION TESTS
# =============================================================================

class TestLowVarianceDetection:
    """Additional tests for remove_low_variance()."""
    
    def test_detects_low_variance(self, sample_data):
        """Should detect low variance features."""
        X, y = sample_data
        selector = FeatureSelector(X, y)
        
        # low_variance_feature has very low variance
        result = selector.remove_low_variance(threshold=0.01)
        
        assert "low_variance_feature" in result
    
    def test_returns_list_of_strings(self, selector):
        """Should return list of feature names."""
        result = selector.remove_low_variance(threshold=0.01)
        
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)
    
    def test_higher_threshold_finds_more(self, selector):
        """Higher threshold should find more low-variance features."""
        low_result = selector.remove_low_variance(threshold=0.001)
        high_result = selector.remove_low_variance(threshold=1.0)
        
        assert len(high_result) >= len(low_result)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_feature(self):
        """Should handle single feature."""
        X = pd.DataFrame({"single": np.random.randn(100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        
        selector = FeatureSelector(X, y)
        
        result = selector.select_top_k(k=1, method="random_forest")
        assert result == ["single"]
    
    def test_select_more_than_available(self, selector, sample_data):
        """Should handle k larger than number of features."""
        X, _ = sample_data
        n_features = len(X.columns)
        
        result = selector.select_top_k(k=n_features + 10, method="random_forest")
        
        # Should return all available features
        assert len(result) == n_features
    
    def test_all_features_same_variance(self):
        """Should handle features with equal variance."""
        np.random.seed(42)
        X = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "c": np.random.randn(100),
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        selector = FeatureSelector(X, y)
        result = selector.remove_low_variance(threshold=0.01)
        
        # None should be low variance
        assert len(result) == 0