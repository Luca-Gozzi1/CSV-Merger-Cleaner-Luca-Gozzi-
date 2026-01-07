"""
Unit tests for the ModelEvaluator class.

Run with: pytest tests/test_evaluator.py -v

Author: Luca Gozzi
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.ml.evaluator import (
    ModelEvaluator,
    EvaluationResult,
    compare_models,
    plot_model_comparison,
    plot_roc_curves_comparison,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_data() -> tuple:
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    y = pd.Series(
        (X["feature_0"] + X["feature_1"] > 0).astype(int)
    )
    
    return X, y


@pytest.fixture
def trained_lr_model(sample_data) -> tuple:
    """Create a trained Logistic Regression model."""
    X, y = sample_data
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    return model, X.columns.tolist()


@pytest.fixture
def trained_rf_model(sample_data) -> tuple:
    """Create a trained Random Forest model."""
    X, y = sample_data
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model, X.columns.tolist()


# =============================================================================
# EVALUATION RESULT TESTS
# =============================================================================

class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_summary_generation(self):
        """Should generate readable summary."""
        result = EvaluationResult(
            model_name="Test Model",
            dataset_name="test",
            metrics={"accuracy": 0.85, "precision": 0.80},
            confusion_matrix=np.array([[50, 10], [5, 35]]),
            y_true=np.array([0] * 60 + [1] * 40),
            y_pred=np.array([0] * 55 + [1] * 45),
        )
        
        summary = result.summary()
        
        assert "Test Model" in summary
        assert "0.85" in summary
        assert "test" in summary
    
    def test_passes_thresholds(self):
        """Should correctly check metric thresholds."""
        result = EvaluationResult(
            model_name="Test",
            dataset_name="test",
            metrics={"accuracy": 0.90, "recall": 0.85, "roc_auc": 0.92},
            confusion_matrix=np.array([[50, 10], [5, 35]]),
            y_true=np.array([0] * 60 + [1] * 40),
            y_pred=np.array([0] * 55 + [1] * 45),
        )
        
        thresholds = result.passes_thresholds()
        
        assert thresholds["accuracy"] == True
        assert thresholds["recall"] == True
        assert thresholds["roc_auc"] == True


# =============================================================================
# MODEL EVALUATOR INITIALIZATION TESTS
# =============================================================================

class TestModelEvaluatorInit:
    """Tests for ModelEvaluator initialization."""
    
    def test_init_basic(self, trained_lr_model):
        """Should initialize with model."""
        model, feature_names = trained_lr_model
        
        evaluator = ModelEvaluator(model, "Logistic Regression")
        
        assert evaluator.model_name == "Logistic Regression"
        assert evaluator.scaler is None
    
    def test_init_with_feature_names(self, trained_lr_model):
        """Should accept feature names."""
        model, feature_names = trained_lr_model
        
        evaluator = ModelEvaluator(
            model, "Logistic Regression",
            feature_names=feature_names
        )
        
        assert len(evaluator.feature_names) == len(feature_names)


# =============================================================================
# EVALUATION TESTS
# =============================================================================

class TestEvaluation:
    """Tests for model evaluation."""
    
    def test_evaluate_returns_result(self, trained_lr_model, sample_data):
        """Evaluation should return EvaluationResult."""
        model, feature_names = trained_lr_model
        X, y = sample_data
        
        evaluator = ModelEvaluator(model, "LR", feature_names=feature_names)
        result = evaluator.evaluate(X, y, "test")
        
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "LR"
        assert result.dataset_name == "test"
    
    def test_evaluate_calculates_metrics(self, trained_lr_model, sample_data):
        """Evaluation should calculate all metrics."""
        model, feature_names = trained_lr_model
        X, y = sample_data
        
        evaluator = ModelEvaluator(model, "LR", feature_names=feature_names)
        result = evaluator.evaluate(X, y)
        
        assert "accuracy" in result.metrics
        assert "precision" in result.metrics
        assert "recall" in result.metrics
        assert "f1_score" in result.metrics
        assert "roc_auc" in result.metrics
    
    def test_evaluate_creates_confusion_matrix(self, trained_lr_model, sample_data):
        """Evaluation should create confusion matrix."""
        model, feature_names = trained_lr_model
        X, y = sample_data
        
        evaluator = ModelEvaluator(model, "LR", feature_names=feature_names)
        result = evaluator.evaluate(X, y)
        
        assert result.confusion_matrix.shape == (2, 2)
        assert result.confusion_matrix.sum() == len(y)
    
    def test_evaluate_stores_predictions(self, trained_lr_model, sample_data):
        """Evaluation should store predictions."""
        model, feature_names = trained_lr_model
        X, y = sample_data
        
        evaluator = ModelEvaluator(model, "LR", feature_names=feature_names)
        result = evaluator.evaluate(X, y)
        
        assert len(result.y_true) == len(y)
        assert len(result.y_pred) == len(y)
        assert result.y_proba is not None
        assert len(result.y_proba) == len(y)


# =============================================================================
# FEATURE IMPORTANCE TESTS
# =============================================================================

class TestFeatureImportance:
    """Tests for feature importance extraction."""
    
    def test_lr_feature_importance(self, trained_lr_model):
        """Should extract importance from Logistic Regression."""
        model, feature_names = trained_lr_model
        
        evaluator = ModelEvaluator(model, "LR", feature_names=feature_names)
        importance_df = evaluator.get_feature_importance()
        
        assert importance_df is not None
        assert len(importance_df) == len(feature_names)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
    
    def test_rf_feature_importance(self, trained_rf_model):
        """Should extract importance from Random Forest."""
        model, feature_names = trained_rf_model
        
        evaluator = ModelEvaluator(model, "RF", feature_names=feature_names)
        importance_df = evaluator.get_feature_importance()
        
        assert importance_df is not None
        assert len(importance_df) == len(feature_names)
    
    def test_importance_is_sorted(self, trained_rf_model):
        """Feature importance should be sorted descending."""
        model, feature_names = trained_rf_model
        
        evaluator = ModelEvaluator(model, "RF", feature_names=feature_names)
        importance_df = evaluator.get_feature_importance()
        
        importances = importance_df["importance"].tolist()
        assert importances == sorted(importances, reverse=True)


# =============================================================================
# PLOTTING TESTS
# =============================================================================

class TestPlotting:
    """Tests for visualization methods."""
    
    def test_plot_confusion_matrix(self, trained_lr_model, sample_data, tmp_path):
        """Should create confusion matrix plot."""
        model, feature_names = trained_lr_model
        X, y = sample_data
        
        evaluator = ModelEvaluator(model, "LR", feature_names=feature_names)
        result = evaluator.evaluate(X, y)
        
        save_path = tmp_path / "cm.png"
        fig = evaluator.plot_confusion_matrix(result, save_path=save_path)
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_roc_curve(self, trained_lr_model, sample_data, tmp_path):
        """Should create ROC curve plot."""
        model, feature_names = trained_lr_model
        X, y = sample_data
        
        evaluator = ModelEvaluator(model, "LR", feature_names=feature_names)
        result = evaluator.evaluate(X, y)
        
        save_path = tmp_path / "roc.png"
        fig = evaluator.plot_roc_curve(result, save_path=save_path)
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_feature_importance(self, trained_rf_model, tmp_path):
        """Should create feature importance plot."""
        model, feature_names = trained_rf_model
        
        evaluator = ModelEvaluator(model, "RF", feature_names=feature_names)
        
        save_path = tmp_path / "importance.png"
        fig = evaluator.plot_feature_importance(top_n=5, save_path=save_path)
        
        assert fig is not None
        assert save_path.exists()


# =============================================================================
# COMPARISON TESTS
# =============================================================================

class TestComparison:
    """Tests for model comparison functions."""
    
    def test_compare_models(self, trained_lr_model, trained_rf_model, sample_data):
        """Should create comparison DataFrame."""
        X, y = sample_data
        
        lr_model, lr_features = trained_lr_model
        rf_model, rf_features = trained_rf_model
        
        lr_evaluator = ModelEvaluator(lr_model, "LR", feature_names=lr_features)
        rf_evaluator = ModelEvaluator(rf_model, "RF", feature_names=rf_features)
        
        lr_result = lr_evaluator.evaluate(X, y)
        rf_result = rf_evaluator.evaluate(X, y)
        
        comparison = compare_models([lr_result, rf_result])
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "Model" in comparison.columns
        assert "accuracy" in comparison.columns
    
    def test_plot_model_comparison(self, trained_lr_model, trained_rf_model, sample_data, tmp_path):
        """Should create comparison bar chart."""
        X, y = sample_data
        
        lr_model, lr_features = trained_lr_model
        rf_model, rf_features = trained_rf_model
        
        lr_evaluator = ModelEvaluator(lr_model, "LR", feature_names=lr_features)
        rf_evaluator = ModelEvaluator(rf_model, "RF", feature_names=rf_features)
        
        lr_result = lr_evaluator.evaluate(X, y)
        rf_result = rf_evaluator.evaluate(X, y)
        
        save_path = tmp_path / "comparison.png"
        fig = plot_model_comparison([lr_result, rf_result], save_path=save_path)
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_roc_comparison(self, trained_lr_model, trained_rf_model, sample_data, tmp_path):
        """Should create ROC comparison plot."""
        X, y = sample_data
        
        lr_model, lr_features = trained_lr_model
        rf_model, rf_features = trained_rf_model
        
        lr_evaluator = ModelEvaluator(lr_model, "LR", feature_names=lr_features)
        rf_evaluator = ModelEvaluator(rf_model, "RF", feature_names=rf_features)
        
        lr_result = lr_evaluator.evaluate(X, y)
        rf_result = rf_evaluator.evaluate(X, y)
        
        save_path = tmp_path / "roc_comparison.png"
        fig = plot_roc_curves_comparison([lr_result, rf_result], save_path=save_path)
        
        assert fig is not None
        assert save_path.exists()