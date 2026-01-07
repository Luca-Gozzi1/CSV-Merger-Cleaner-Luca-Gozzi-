"""
Unit tests for the feature engineering module.

These tests verify that feature creation and encoding work correctly.

Run with: pytest tests/test_features.py -v

Author: Luca Gozzi 
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.engineer import (
    FeatureEngineer,
    FeatureEngineeringReport,
    engineer_features,
)
from src.features.selector import FeatureSelector


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_supply_chain_df() -> pd.DataFrame:
    """Create a sample DataFrame mimicking DataCo structure for testing."""
    np.random.seed(42)
    n_rows = 100
    
    base_date = datetime(2017, 1, 1)
    order_dates = [base_date + timedelta(days=i) for i in range(n_rows)]
    ship_dates = [d + timedelta(days=np.random.randint(1, 5)) for d in order_dates]
    
    return pd.DataFrame({
        "Order Id": [f"ORD-{i:05d}" for i in range(n_rows)],
        "order date (DateOrders)": order_dates,
        "shipping date (DateOrders)": ship_dates,
        "Days for shipping (real)": np.random.randint(1, 10, n_rows),
        "Days for shipment (scheduled)": np.random.randint(2, 7, n_rows),
        "Late_delivery_risk": np.random.choice([0, 1], n_rows),
        "Shipping Mode": np.random.choice(
            ["Standard Class", "First Class", "Second Class", "Same Day"],
            n_rows
        ),
        "Order Item Discount Rate": np.random.uniform(0, 0.3, n_rows),
        "Order Item Quantity": np.random.randint(1, 10, n_rows),
        "Order Item Product Price": np.random.uniform(10, 500, n_rows),
        "Sales": np.random.uniform(50, 1000, n_rows),
        "Category Name": np.random.choice(
            ["Electronics", "Furniture", "Office Supplies"],
            n_rows
        ),
        "Market": np.random.choice(
            ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"],
            n_rows
        ),
        "Order Region": np.random.choice(
            ["Central America", "Western Europe", "Southeast Asia"],
            n_rows
        ),
        "Customer Segment": np.random.choice(
            ["Consumer", "Corporate", "Home Office"],
            n_rows
        ),
    })


# =============================================================================
# FEATURE ENGINEERING REPORT TESTS
# =============================================================================

class TestFeatureEngineeringReport:
    """Tests for FeatureEngineeringReport dataclass."""
    
    def test_default_values(self):
        """Report should have sensible defaults."""
        report = FeatureEngineeringReport()
        
        assert report.initial_features == 0
        assert report.final_features == 0
        assert len(report.features_created) == 0
    
    def test_add_feature(self):
        """Should track created features."""
        report = FeatureEngineeringReport()
        
        report.add_feature("test_feature", "A test feature")
        
        assert "test_feature" in report.features_created
        assert len(report.transformations) == 1
    
    def test_summary_generation(self):
        """Summary should include key information."""
        report = FeatureEngineeringReport(
            initial_features=10,
            final_features=25,
        )
        report.add_feature("new_feat")
        
        summary = report.summary()
        
        assert "10" in summary
        assert "25" in summary
        assert "new_feat" in summary


# =============================================================================
# FEATURE ENGINEER TESTS
# =============================================================================

class TestFeatureEngineerInit:
    """Tests for FeatureEngineer initialization."""
    
    def test_init_creates_copy(self, sample_supply_chain_df):
        """Should work on a copy, not modify original."""
        original_cols = len(sample_supply_chain_df.columns)
        
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer.df["new_col"] = 1
        
        assert len(sample_supply_chain_df.columns) == original_cols
    
    def test_init_creates_report(self, sample_supply_chain_df):
        """Should create a report on init."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        
        assert isinstance(engineer.report, FeatureEngineeringReport)
        assert engineer.report.initial_features == len(sample_supply_chain_df.columns)


class TestTemporalFeatures:
    """Tests for temporal feature creation."""
    
    def test_creates_day_of_week(self, sample_supply_chain_df):
        """Should create order_day_of_week feature."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_temporal_features()
        
        assert "order_day_of_week" in engineer.df.columns
        assert engineer.df["order_day_of_week"].between(0, 6).all()
    
    def test_creates_month(self, sample_supply_chain_df):
        """Should create order_month feature."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_temporal_features()
        
        assert "order_month" in engineer.df.columns
        assert engineer.df["order_month"].between(1, 12).all()
    
    def test_creates_is_weekend(self, sample_supply_chain_df):
        """Should create is_weekend binary feature."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_temporal_features()
        
        assert "is_weekend" in engineer.df.columns
        assert set(engineer.df["is_weekend"].unique()).issubset({0, 1})
    
    def test_creates_holiday_season(self, sample_supply_chain_df):
        """Should create is_holiday_season feature."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_temporal_features()
        
        assert "is_holiday_season" in engineer.df.columns
    
    def test_reports_temporal_features(self, sample_supply_chain_df):
        """Should report all created temporal features."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_temporal_features()
        
        assert len(engineer.report.features_created) >= 6


class TestShippingFeatures:
    """Tests for shipping feature creation."""
    
    def test_creates_lead_time_variance(self, sample_supply_chain_df):
        """Should create shipping_lead_time_variance feature."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_shipping_features()
        
        assert "shipping_lead_time_variance" in engineer.df.columns
    
    def test_lead_time_variance_calculation(self, sample_supply_chain_df):
        """Lead time variance should be real - scheduled."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_shipping_features()
        
        expected = (
            sample_supply_chain_df["Days for shipping (real)"] -
            sample_supply_chain_df["Days for shipment (scheduled)"]
        )
        
        pd.testing.assert_series_equal(
            engineer.df["shipping_lead_time_variance"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )
    
    def test_creates_shipping_mode_risk(self, sample_supply_chain_df):
        """Should create shipping_mode_risk feature."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_shipping_features()
        
        assert "shipping_mode_risk" in engineer.df.columns
        assert engineer.df["shipping_mode_risk"].between(0, 1).all()
    
    def test_creates_is_expedited(self, sample_supply_chain_df):
        """Should create is_expedited binary feature."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_shipping_features()
        
        assert "is_expedited" in engineer.df.columns
        assert set(engineer.df["is_expedited"].unique()).issubset({0, 1})


class TestGeographicFeatures:
    """Tests for geographic feature creation."""
    
    def test_creates_market_risk_score(self, sample_supply_chain_df):
        """Should create market_risk_score feature."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_geographic_features()
        
        assert "market_risk_score" in engineer.df.columns
        assert engineer.df["market_risk_score"].between(0, 1).all()
    
    def test_creates_is_international(self, sample_supply_chain_df):
        """Should create is_international binary feature."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_geographic_features()
        
        assert "is_international" in engineer.df.columns


class TestProductFeatures:
    """Tests for product/order feature creation."""
    
    def test_creates_order_item_value(self, sample_supply_chain_df):
        """Should create order_item_value feature."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_product_features()
        
        assert "order_item_value" in engineer.df.columns
    
    def test_order_value_calculation(self, sample_supply_chain_df):
        """Order value should be price * quantity."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_product_features()
        
        expected = (
            sample_supply_chain_df["Order Item Product Price"] *
            sample_supply_chain_df["Order Item Quantity"]
        )
        
        pd.testing.assert_series_equal(
            engineer.df["order_item_value"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )
    
    def test_creates_has_discount(self, sample_supply_chain_df):
        """Should create has_discount binary feature."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._create_product_features()
        
        assert "has_discount" in engineer.df.columns


class TestCategoricalEncoding:
    """Tests for categorical encoding."""
    
    def test_one_hot_encoding_shipping_mode(self, sample_supply_chain_df):
        """Should one-hot encode Shipping Mode."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._encode_categoricals(drop_original=False)
        
        # Should have columns like shipping_mode_Standard Class
        shipping_cols = [c for c in engineer.df.columns if c.startswith("shipping_mode_")]
        assert len(shipping_cols) > 0
    
    def test_label_encoding_market(self, sample_supply_chain_df):
        """Should label encode Market."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._encode_categoricals(drop_original=False)
        
        assert "market_encoded" in engineer.df.columns
        assert engineer.df["market_encoded"].dtype in [np.int32, np.int64, int]
    
    def test_drop_original_categoricals(self, sample_supply_chain_df):
        """Should drop original categorical columns when requested."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer._encode_categoricals(drop_original=True)
        
        # Original Shipping Mode should be dropped
        assert "Shipping Mode" not in engineer.df.columns


class TestFullPipeline:
    """Tests for the complete feature engineering pipeline."""
    
    def test_engineer_all_returns_dataframe(self, sample_supply_chain_df):
        """engineer_all should return a DataFrame."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        result = engineer.engineer_all()
        
        assert isinstance(result, pd.DataFrame)
    
    def test_engineer_all_creates_features(self, sample_supply_chain_df):
        """engineer_all should create new features."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        result = engineer.engineer_all()
        
        assert len(engineer.report.features_created) > 10
    
    def test_engineer_all_updates_report(self, sample_supply_chain_df):
        """engineer_all should update final_features in report."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        engineer.engineer_all()
        
        assert engineer.report.final_features > 0
    
    def test_preserves_target(self, sample_supply_chain_df):
        """Should preserve target column."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        result = engineer.engineer_all()
        
        assert "Late_delivery_risk" in result.columns
    
    def test_no_missing_values_in_features(self, sample_supply_chain_df):
        """Engineered features should not have missing values."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        result = engineer.engineer_all()
        
        # Check numeric columns only
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        missing = result[numeric_cols].isna().sum().sum()
        
        # Allow small number of missing (from edge cases)
        assert missing < len(result) * 0.01  # Less than 1%


class TestConvenienceFunction:
    """Tests for engineer_features convenience function."""
    
    def test_returns_tuple(self, sample_supply_chain_df):
        """Should return tuple of (DataFrame, Report)."""
        result = engineer_features(sample_supply_chain_df)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], FeatureEngineeringReport)


# =============================================================================
# FEATURE SELECTOR TESTS
# =============================================================================

class TestFeatureSelector:
    """Tests for FeatureSelector class."""
    
    @pytest.fixture
    def selector_setup(self, sample_supply_chain_df):
        """Create feature-engineered data for selector testing."""
        engineer = FeatureEngineer(sample_supply_chain_df)
        df = engineer.engineer_all()
        
        X = df.drop(columns=["Late_delivery_risk"])
        y = df["Late_delivery_risk"]
        
        return FeatureSelector(X, y)
    
    def test_init(self, selector_setup):
        """Should initialize with features and target."""
        selector = selector_setup
        
        assert len(selector.numeric_cols) > 0
    
    def test_remove_low_variance(self, selector_setup):
        """Should identify low-variance features."""
        selector = selector_setup
        low_var = selector.remove_low_variance(threshold=0.001)
        
        assert isinstance(low_var, list)
    
    def test_remove_high_correlation(self, selector_setup):
        """Should identify correlated feature pairs."""
        selector = selector_setup
        high_corr = selector.remove_high_correlation(threshold=0.9)
        
        assert isinstance(high_corr, list)
    
    def test_get_feature_importance_rf(self, selector_setup):
        """Should return RF importance DataFrame."""
        selector = selector_setup
        importance = selector.get_feature_importance_rf(n_estimators=10)
        
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
    
    def test_select_top_k(self, selector_setup):
        """Should select top k features."""
        selector = selector_setup
        top_features = selector.select_top_k(k=5, method="random_forest")
        
        assert len(top_features) == 5
        assert all(isinstance(f, str) for f in top_features)