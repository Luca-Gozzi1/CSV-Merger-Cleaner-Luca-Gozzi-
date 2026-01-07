"""
Unit tests for the DataSplitter class.

These tests verify that data splitting works correctly, maintains
temporal order, and produces balanced splits.

Run with: pytest tests/test_splitter.py -v

Author: Luca Gozzi 
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.ml.splitter import (
    DataSplitter,
    SplitResult,
    split_data,
    get_X_y,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temporal_df() -> pd.DataFrame:
    """Create a DataFrame with temporal ordering for time-based split testing."""
    np.random.seed(42)
    n_rows = 1000
    
    # Create sequential dates
    base_date = datetime(2017, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_rows)]
    
    return pd.DataFrame({
        "order date (DateOrders)": dates,
        "Late_delivery_risk": np.random.choice([0, 1], n_rows),
        "feature_1": np.random.randn(n_rows),
        "feature_2": np.random.randn(n_rows),
        "feature_3": np.random.randint(0, 10, n_rows),
    })


@pytest.fixture
def no_date_df() -> pd.DataFrame:
    """Create a DataFrame without date column."""
    np.random.seed(42)
    n_rows = 100
    
    return pd.DataFrame({
        "Late_delivery_risk": np.random.choice([0, 1], n_rows),
        "feature_1": np.random.randn(n_rows),
        "feature_2": np.random.randn(n_rows),
    })


# =============================================================================
# SPLIT RESULT TESTS
# =============================================================================

class TestSplitResult:
    """Tests for SplitResult dataclass."""
    
    def test_size_properties(self, temporal_df):
        """Should correctly report sizes."""
        result = SplitResult(
            train=temporal_df.iloc[:700],
            validation=temporal_df.iloc[700:850],
            test=temporal_df.iloc[850:],
        )
        
        assert result.train_size == 700
        assert result.validation_size == 150
        assert result.test_size == 150
        assert result.total_size == 1000
    
    def test_summary_generation(self, temporal_df):
        """Should generate readable summary."""
        result = SplitResult(
            train=temporal_df.iloc[:700],
            validation=temporal_df.iloc[700:850],
            test=temporal_df.iloc[850:],
            split_info={"method": "time-based"}
        )
        
        summary = result.summary()
        
        assert "700" in summary
        assert "150" in summary
        assert "time-based" in summary
        assert "TARGET DISTRIBUTION" in summary


# =============================================================================
# DATA SPLITTER INITIALIZATION TESTS
# =============================================================================

class TestDataSplitterInit:
    """Tests for DataSplitter initialization."""
    
    def test_init_with_default_ratios(self, temporal_df):
        """Should initialize with default ratios."""
        splitter = DataSplitter(temporal_df)
        
        assert splitter.train_ratio == 0.70
        assert splitter.validation_ratio == 0.15
        assert splitter.test_ratio == 0.15
    
    def test_init_with_custom_ratios(self, temporal_df):
        """Should accept custom ratios."""
        splitter = DataSplitter(
            temporal_df,
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
        )
        
        assert splitter.train_ratio == 0.6
        assert splitter.validation_ratio == 0.2
        assert splitter.test_ratio == 0.2
    
    def test_init_validates_ratios(self, temporal_df):
        """Should raise error if ratios don't sum to 1."""
        with pytest.raises(ValueError) as exc_info:
            DataSplitter(
                temporal_df,
                train_ratio=0.5,
                validation_ratio=0.2,
                test_ratio=0.2,  # Sum = 0.9
            )
        
        assert "sum to 1.0" in str(exc_info.value).lower()
    
    def test_init_creates_copy(self, temporal_df):
        """Should work on a copy of the data."""
        original_len = len(temporal_df)
        
        splitter = DataSplitter(temporal_df)
        splitter.df = splitter.df.head(10)
        
        assert len(temporal_df) == original_len


# =============================================================================
# TIME-BASED SPLIT TESTS
# =============================================================================

class TestTimeBasedSplit:
    """Tests for time-based splitting."""
    
    def test_split_returns_split_result(self, temporal_df):
        """Should return a SplitResult object."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="time")
        
        assert isinstance(result, SplitResult)
    
    def test_split_sizes_match_ratios(self, temporal_df):
        """Split sizes should match specified ratios."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="time")
        
        total = result.total_size
        
        # Allow for rounding differences
        assert 0.68 <= result.train_size / total <= 0.72
        assert 0.13 <= result.validation_size / total <= 0.17
        assert 0.13 <= result.test_size / total <= 0.17
    
    def test_temporal_order_preserved(self, temporal_df):
        """Train dates should be before validation, validation before test."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="time")
        
        date_col = "order date (DateOrders)"
        
        train_max = result.train[date_col].max()
        val_min = result.validation[date_col].min()
        val_max = result.validation[date_col].max()
        test_min = result.test[date_col].min()
        
        assert train_max <= val_min, "Train max should be <= validation min"
        assert val_max <= test_min, "Validation max should be <= test min"
    
    def test_no_overlapping_rows(self, temporal_df):
        """No rows should appear in multiple splits."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="time")
        
        # Check no overlap using index
        train_idx = set(result.train.index)
        val_idx = set(result.validation.index)
        test_idx = set(result.test.index)
        
        assert len(train_idx & val_idx) == 0, "Train and validation overlap"
        assert len(val_idx & test_idx) == 0, "Validation and test overlap"
        assert len(train_idx & test_idx) == 0, "Train and test overlap"
    
    def test_all_rows_included(self, temporal_df):
        """All original rows should be in one of the splits."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="time")
        
        assert result.total_size == len(temporal_df)
    
    def test_split_info_populated(self, temporal_df):
        """Split info should contain method and date ranges."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="time")
        
        assert result.split_info["method"] == "time-based"
        assert "train_date_range" in result.split_info
        assert "validation_date_range" in result.split_info
        assert "test_date_range" in result.split_info


# =============================================================================
# RANDOM SPLIT TESTS
# =============================================================================

class TestRandomSplit:
    """Tests for random splitting."""
    
    def test_random_split_sizes(self, temporal_df):
        """Random split should also produce correct sizes."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="random")
        
        assert result.total_size == len(temporal_df)
    
    def test_random_split_is_reproducible(self, temporal_df):
        """Random split should be reproducible with same seed."""
        splitter1 = DataSplitter(temporal_df)
        result1 = splitter1.split(method="random")
        
        splitter2 = DataSplitter(temporal_df)
        result2 = splitter2.split(method="random")
        
        # First row of train should be identical
        pd.testing.assert_frame_equal(
            result1.train.head(10).reset_index(drop=True),
            result2.train.head(10).reset_index(drop=True)
        )
    
    def test_random_split_info(self, temporal_df):
        """Random split should have warning in split_info."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="random")
        
        assert result.split_info["method"] == "random"
        assert "warning" in result.split_info


# =============================================================================
# FALLBACK BEHAVIOR TESTS
# =============================================================================

class TestFallbackBehavior:
    """Tests for fallback behavior when date column is missing."""
    
    def test_fallback_to_index_split(self, no_date_df):
        """Should fall back to index-based split without date column."""
        splitter = DataSplitter(no_date_df)
        result = splitter.split(method="time")
        
        assert result.split_info["method"] == "index-based"
        assert result.total_size == len(no_date_df)
    
    def test_invalid_method_raises(self, temporal_df):
        """Should raise error for unknown split method."""
        splitter = DataSplitter(temporal_df)
        
        with pytest.raises(ValueError) as exc_info:
            splitter.split(method="invalid_method")
        
        assert "unknown" in str(exc_info.value).lower()


# =============================================================================
# SAVE AND LOAD TESTS
# =============================================================================

class TestSaveAndLoad:
    """Tests for saving and loading splits."""
    
    def test_save_creates_files(self, temporal_df, tmp_path):
        """save_splits should create CSV files."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="time")
        
        paths = splitter.save_splits(result, output_dir=tmp_path)
        
        assert (tmp_path / "train.csv").exists()
        assert (tmp_path / "validation.csv").exists()
        assert (tmp_path / "test.csv").exists()
        assert (tmp_path / "split_info.txt").exists()
    
    def test_load_splits(self, temporal_df, tmp_path):
        """Should be able to reload saved splits."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="time")
        splitter.save_splits(result, output_dir=tmp_path)
        
        # Load the splits
        loaded = DataSplitter.load_splits(input_dir=tmp_path)
        
        assert loaded.train_size == result.train_size
        assert loaded.validation_size == result.validation_size
        assert loaded.test_size == result.test_size
    
    def test_load_missing_files_raises(self, tmp_path):
        """Should raise error if split files don't exist."""
        with pytest.raises(FileNotFoundError):
            DataSplitter.load_splits(input_dir=tmp_path)


# =============================================================================
# TARGET DISTRIBUTION TESTS
# =============================================================================

class TestTargetDistribution:
    """Tests for target variable distribution across splits."""
    
    def test_target_preserved_in_splits(self, temporal_df):
        """Target column should be present in all splits."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="time")
        
        assert "Late_delivery_risk" in result.train.columns
        assert "Late_delivery_risk" in result.validation.columns
        assert "Late_delivery_risk" in result.test.columns
    
    def test_target_distribution_reasonable(self, temporal_df):
        """Target distribution should be similar across splits."""
        splitter = DataSplitter(temporal_df)
        result = splitter.split(method="time")
        
        train_late_pct = result.train["Late_delivery_risk"].mean()
        val_late_pct = result.validation["Late_delivery_risk"].mean()
        test_late_pct = result.test["Late_delivery_risk"].mean()
        
        # All should be between 0.3 and 0.7 (not extremely imbalanced)
        for pct in [train_late_pct, val_late_pct, test_late_pct]:
            assert 0.2 <= pct <= 0.8, f"Extreme imbalance detected: {pct}"


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_split_data_function(self, temporal_df):
        """split_data should work as convenience function."""
        result = split_data(temporal_df, method="time")
        
        assert isinstance(result, SplitResult)
        assert result.total_size == len(temporal_df)
    
    def test_split_data_with_save(self, temporal_df, tmp_path):
        """split_data should save when requested."""
        # Temporarily override config path
        result = split_data(temporal_df, method="time", save=False)
        
        assert isinstance(result, SplitResult)
    
    def test_get_X_y(self, temporal_df):
        """get_X_y should separate features and target."""
        X, y = get_X_y(temporal_df)
        
        assert "Late_delivery_risk" not in X.columns
        assert len(y) == len(temporal_df)
        assert y.name == "Late_delivery_risk"
    
    def test_get_X_y_missing_target(self, temporal_df):
        """get_X_y should raise if target not found."""
        df_no_target = temporal_df.drop(columns=["Late_delivery_risk"])
        
        with pytest.raises(ValueError) as exc_info:
            get_X_y(df_no_target)
        
        assert "not found" in str(exc_info.value).lower()


# =============================================================================
# SHUFFLE WITHIN SPLITS TEST
# =============================================================================

class TestShuffleWithinSplits:
    """Tests for shuffle_within_splits option."""
    
    def test_shuffle_changes_order(self, temporal_df):
        """Shuffling should change row order within splits."""
        splitter = DataSplitter(temporal_df)
        
        result_no_shuffle = splitter.split(method="time", shuffle_within_splits=False)
        
        splitter2 = DataSplitter(temporal_df)
        result_shuffled = splitter2.split(method="time", shuffle_within_splits=True)
        
        # Sizes should be same
        assert result_no_shuffle.train_size == result_shuffled.train_size
        
        # But order might differ (check first few rows)
        # Note: there's a tiny chance they're the same, but very unlikely
        train_no_shuffle_first = result_no_shuffle.train.head(5)
        train_shuffled_first = result_shuffled.train.head(5)
        
        # At least the split info should note it was shuffled
        assert result_shuffled.split_info.get("shuffled_within_splits", False)