"""
Unit tests for the ModelTrainer class.

These tests verify that model training works correctly for all
supported algorithms.

Run with: pytest tests/test_trainer.py -v

Author: Luca Gozzi
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.ml.trainer import (
    ModelTrainer,
    TrainingResult,
    train_models,
    XGBOOST_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def training_data() -> tuple:
    """Create sample training data for model testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # Generate random features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    # Generate target with some relationship to features
    y = pd.Series(
        (X["feature_0"] + X["feature_1"] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    )
    
    return X, y


@pytest.fixture
def train_val_data(training_data) -> tuple:
    """Split training data into train and validation sets."""
    X, y = training_data
    
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    return X_train, y_train, X_val, y_val


# =============================================================================
# TRAINING RESULT TESTS
# =============================================================================

class TestTrainingResult:
    """Tests for TrainingResult dataclass."""
    
    def test_summary_generation(self, training_data):
        """Should generate readable summary."""
        X, y = training_data
        
        result = TrainingResult(
            model=None,
            model_name="Test Model",
            training_time=1.5,
            feature_names=X.columns.tolist(),
            params={"param1": "value1"},
        )
        
        summary = result.summary()
        
        assert "Test Model" in summary
        assert "1.5" in summary
        assert "param1" in summary


# =============================================================================
# MODEL TRAINER INITIALIZATION TESTS
# =============================================================================

class TestModelTrainerInit:
    """Tests for ModelTrainer initialization."""
    
    def test_init_basic(self, training_data):
        """Should initialize with training data."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y)
        
        assert len(trainer.feature_names) == X.shape[1]
        assert trainer.scaler is not None  # Default is to scale
    
    def test_init_without_scaling(self, training_data):
        """Should work without feature scaling."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y, scale_features=False)
        
        assert trainer.scaler is None
    
    def test_init_with_validation(self, train_val_data):
        """Should accept validation data."""
        X_train, y_train, X_val, y_val = train_val_data
        
        trainer = ModelTrainer(X_train, y_train, X_val, y_val)
        
        assert trainer.X_val is not None
        assert trainer.y_val is not None


# =============================================================================
# LOGISTIC REGRESSION TESTS
# =============================================================================

class TestLogisticRegression:
    """Tests for Logistic Regression training."""
    
    def test_train_returns_result(self, training_data):
        """Should return TrainingResult."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y)
        result = trainer.train_logistic_regression()
        
        assert isinstance(result, TrainingResult)
        assert result.model_name == "Logistic Regression"
    
    def test_model_can_predict(self, training_data):
        """Trained model should be able to make predictions."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y)
        result = trainer.train_logistic_regression()
        
        # Make predictions
        X_scaled = trainer._get_scaled_data(X)
        predictions = result.model.predict(X_scaled)
        
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})
    
    def test_model_can_predict_proba(self, training_data):
        """Trained model should return probability scores."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y)
        result = trainer.train_logistic_regression()
        
        X_scaled = trainer._get_scaled_data(X)
        proba = result.model.predict_proba(X_scaled)
        
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_training_time_recorded(self, training_data):
        """Training time should be recorded."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y)
        result = trainer.train_logistic_regression()
        
        assert result.training_time > 0
    
    def test_scaler_attached(self, training_data):
        """Scaler should be attached to result."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y, scale_features=True)
        result = trainer.train_logistic_regression()
        
        assert result.scaler is not None


# =============================================================================
# RANDOM FOREST TESTS
# =============================================================================

class TestRandomForest:
    """Tests for Random Forest training."""
    
    def test_train_returns_result(self, training_data):
        """Should return TrainingResult."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y, scale_features=False)
        result = trainer.train_random_forest()
        
        assert isinstance(result, TrainingResult)
        assert result.model_name == "Random Forest"
    
    def test_model_can_predict(self, training_data):
        """Trained model should be able to make predictions."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y, scale_features=False)
        result = trainer.train_random_forest()
        
        predictions = result.model.predict(X.values)
        
        assert len(predictions) == len(y)
    
    def test_feature_importances_available(self, training_data):
        """Random Forest should provide feature importances."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y, scale_features=False)
        result = trainer.train_random_forest()
        
        importances = result.model.feature_importances_
        
        assert len(importances) == X.shape[1]
        assert np.isclose(importances.sum(), 1.0)  # Should sum to 1
    
    def test_no_scaler_for_rf(self, training_data):
        """Random Forest should not have scaler attached."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y, scale_features=True)  # Even if scaling is on
        result = trainer.train_random_forest()
        
        assert result.scaler is None  # RF doesn't use scaling


# =============================================================================
# XGBOOST TESTS
# =============================================================================

@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoost:
    """Tests for XGBoost training."""
    
    def test_train_returns_result(self, training_data):
        """Should return TrainingResult."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y, scale_features=False)
        result = trainer.train_xgboost()
        
        assert isinstance(result, TrainingResult)
        assert result.model_name == "XGBoost"
    
    def test_model_can_predict(self, training_data):
        """Trained model should be able to make predictions."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y, scale_features=False)
        result = trainer.train_xgboost()
        
        predictions = result.model.predict(X.values)
        
        assert len(predictions) == len(y)
    
    def test_with_early_stopping(self, train_val_data):
        """Should support early stopping with validation set."""
        X_train, y_train, X_val, y_val = train_val_data
        
        trainer = ModelTrainer(X_train, y_train, X_val, y_val, scale_features=False)
        result = trainer.train_xgboost(use_early_stopping=True)
        
        assert isinstance(result, TrainingResult)
    
    def test_feature_importances_available(self, training_data):
        """XGBoost should provide feature importances."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y, scale_features=False)
        result = trainer.train_xgboost()
        
        importances = result.model.feature_importances_
        
        assert len(importances) == X.shape[1]


# =============================================================================
# TRAIN ALL TESTS
# =============================================================================

class TestTrainAll:
    """Tests for training all models."""
    
    def test_train_all_returns_dict(self, training_data):
        """train_all should return dictionary of results."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y)
        results = trainer.train_all()
        
        assert isinstance(results, dict)
        assert ModelTrainer.LOGISTIC_REGRESSION in results
        assert ModelTrainer.RANDOM_FOREST in results
    
    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_train_all_includes_xgboost(self, training_data):
        """train_all should include XGBoost when available."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y)
        results = trainer.train_all()
        
        assert ModelTrainer.XGBOOST in results


# =============================================================================
# SAVE AND LOAD TESTS
# =============================================================================

class TestSaveAndLoad:
    """Tests for saving and loading models."""
    
    def test_save_model(self, training_data, tmp_path):
        """Should save model to disk."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y)
        result = trainer.train_logistic_regression()
        
        filepath = tmp_path / "test_model.pkl"
        saved_path = ModelTrainer.save_model(result, filepath)
        
        assert saved_path.exists()
    
    def test_load_model(self, training_data, tmp_path):
        """Should load saved model."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y)
        result = trainer.train_logistic_regression()
        
        filepath = tmp_path / "test_model.pkl"
        ModelTrainer.save_model(result, filepath)
        
        # Load the model
        model, scaler, feature_names = ModelTrainer.load_model(filepath)
        
        assert model is not None
        assert len(feature_names) == X.shape[1]
    
    def test_loaded_model_can_predict(self, training_data, tmp_path):
        """Loaded model should be able to make predictions."""
        X, y = training_data
        
        trainer = ModelTrainer(X, y)
        result = trainer.train_logistic_regression()
        
        filepath = tmp_path / "test_model.pkl"
        ModelTrainer.save_model(result, filepath)
        
        # Load and predict
        model, scaler, _ = ModelTrainer.load_model(filepath)
        
        X_scaled = scaler.transform(X) if scaler else X.values
        predictions = model.predict(X_scaled)
        
        assert len(predictions) == len(y)
    
    def test_load_missing_file_raises(self, tmp_path):
        """Should raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            ModelTrainer.load_model(tmp_path / "nonexistent.pkl")


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunction:
    """Tests for train_models convenience function."""
    
    def test_train_models_basic(self, training_data):
        """train_models should work with basic inputs."""
        X, y = training_data
        
        results = train_models(X, y, save=False)
        
        assert isinstance(results, dict)
        assert len(results) >= 2  # At least LR and RF
    
    def test_train_specific_models(self, training_data):
        """Should train only specified models."""
        X, y = training_data
        
        results = train_models(
            X, y,
            models=[ModelTrainer.LOGISTIC_REGRESSION],
            save=False
        )
        
        assert len(results) == 1
        assert ModelTrainer.LOGISTIC_REGRESSION in results