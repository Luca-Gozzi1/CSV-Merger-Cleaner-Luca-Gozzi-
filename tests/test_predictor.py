"""
Unit tests for the DelayPredictor class.

Run with: pytest tests/test_predictor.py -v

Author: Luca Gozzi
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from src.ml.predictor import DelayPredictor, load_predictor


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create sample feature data."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randint(0, 10, n_samples),
    })


@pytest.fixture
def trained_model(sample_features) -> tuple:
    """Create a simple trained model."""
    X = sample_features
    y = (X["feature_1"] + X["feature_2"] > 0).astype(int)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model, X.columns.tolist()


@pytest.fixture
def predictor(trained_model) -> DelayPredictor:
    """Create a DelayPredictor instance."""
    model, feature_names = trained_model
    return DelayPredictor(
        model=model,
        feature_names=feature_names,
        model_name="Test Model"
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestDelayPredictorInit:
    """Tests for DelayPredictor initialization."""
    
    def test_init_basic(self, trained_model):
        """Should initialize with model."""
        model, feature_names = trained_model
        
        predictor = DelayPredictor(model, model_name="Test")
        
        assert predictor.model_name == "Test"
        assert predictor.scaler is None
    
    def test_init_with_feature_names(self, trained_model):
        """Should store feature names."""
        model, feature_names = trained_model
        
        predictor = DelayPredictor(
            model,
            feature_names=feature_names,
            model_name="Test"
        )
        
        assert predictor.feature_names == feature_names


# =============================================================================
# PREDICTION TESTS
# =============================================================================

class TestPredictions:
    """Tests for prediction methods."""
    
    def test_predict_returns_array(self, predictor, sample_features):
        """predict() should return numpy array."""
        predictions = predictor.predict(sample_features)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_features)
    
    def test_predict_binary_values(self, predictor, sample_features):
        """Predictions should be binary (0 or 1)."""
        predictions = predictor.predict(sample_features)
        
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba_returns_array(self, predictor, sample_features):
        """predict_proba() should return probabilities."""
        probabilities = predictor.predict_proba(sample_features)
        
        assert isinstance(probabilities, np.ndarray)
        assert len(probabilities) == len(sample_features)
    
    def test_predict_proba_valid_range(self, predictor, sample_features):
        """Probabilities should be between 0 and 1."""
        probabilities = predictor.predict_proba(sample_features)
        
        assert all(0 <= p <= 1 for p in probabilities)


# =============================================================================
# RISK SCORE TESTS
# =============================================================================

class TestRiskScores:
    """Tests for risk score calculation."""
    
    def test_get_risk_scores_returns_dataframe(self, predictor, sample_features):
        """get_risk_scores() should return DataFrame."""
        risk_df = predictor.get_risk_scores(sample_features)
        
        assert isinstance(risk_df, pd.DataFrame)
        assert len(risk_df) == len(sample_features)
    
    def test_risk_scores_columns(self, predictor, sample_features):
        """Risk DataFrame should have expected columns."""
        risk_df = predictor.get_risk_scores(sample_features)
        
        expected_columns = [
            "delay_probability",
            "delay_prediction",
            "risk_score",
            "risk_category",
        ]
        
        for col in expected_columns:
            assert col in risk_df.columns
    
    def test_risk_categories_valid(self, predictor, sample_features):
        """Risk categories should be low/medium/high."""
        risk_df = predictor.get_risk_scores(sample_features)
        
        valid_categories = {"low", "medium", "high"}
        actual_categories = set(risk_df["risk_category"].dropna().unique())
        
        assert actual_categories.issubset(valid_categories)
    
    def test_risk_score_range(self, predictor, sample_features):
        """Risk scores should be 0-100."""
        risk_df = predictor.get_risk_scores(sample_features)
        
        assert risk_df["risk_score"].min() >= 0
        assert risk_df["risk_score"].max() <= 100


# =============================================================================
# DATAFRAME ENRICHMENT TESTS
# =============================================================================

class TestDataFrameEnrichment:
    """Tests for adding predictions to DataFrames."""
    
    def test_add_predictions_to_dataframe(self, predictor, sample_features):
        """Should add prediction columns to DataFrame."""
        enriched = predictor.add_predictions_to_dataframe(sample_features)
        
        assert "delay_risk_probability" in enriched.columns
        assert "delay_risk_prediction" in enriched.columns
        assert "delay_risk_category" in enriched.columns
        assert "delay_risk_score" in enriched.columns
    
    def test_add_predictions_preserves_original(self, predictor, sample_features):
        """Should preserve original columns."""
        enriched = predictor.add_predictions_to_dataframe(sample_features)
        
        for col in sample_features.columns:
            assert col in enriched.columns
    
    def test_get_high_risk_shipments(self, predictor, sample_features):
        """Should filter to high-risk shipments."""
        high_risk = predictor.get_high_risk_shipments(sample_features, threshold=0.5)
        
        assert isinstance(high_risk, pd.DataFrame)
        # All returned should have probability >= threshold
        if len(high_risk) > 0:
            assert all(high_risk["delay_risk_probability"] >= 0.5)


# =============================================================================
# SUMMARY TESTS
# =============================================================================

class TestSummary:
    """Tests for risk summary statistics."""
    
    def test_summarize_risk_distribution(self, predictor, sample_features):
        """Should return summary dictionary."""
        summary = predictor.summarize_risk_distribution(sample_features)
        
        assert isinstance(summary, dict)
        assert "total_shipments" in summary
        assert "predicted_late" in summary
        assert "average_risk_score" in summary
        assert "risk_category_counts" in summary
    
    def test_summary_totals_match(self, predictor, sample_features):
        """Summary counts should match input size."""
        summary = predictor.summarize_risk_distribution(sample_features)
        
        assert summary["total_shipments"] == len(sample_features)
        assert (summary["predicted_late"] + summary["predicted_on_time"]) == len(sample_features)
