"""
Model training module for Supply Chain Explorer.

This module provides the ModelTrainer class responsible for training
machine learning models for shipment delay prediction. It supports
multiple algorithms and handles the full training workflow including
preprocessing, fitting, and model persistence.

Supported Models:
1. Logistic Regression - Simple, interpretable baseline
2. Random Forest - Robust ensemble method
3. XGBoost - State-of-the-art gradient boosting

Author: Luca Gozzi
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import time
import warnings

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

# XGBoost import with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. XGBoost models will not be available.")

from src.config import (
    TARGET_COLUMN,
    RANDOM_SEED,
    MODELS_DIR,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)


# Configure module logger
logger = logging.getLogger(__name__)

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class TrainingResult:
    """
    Container for model training results.
    
    Holds the trained model along with metadata about the training
    process for documentation and analysis.
    
    Attributes:
        model: The trained model object.
        model_name: Name/type of the model.
        training_time: Time taken to train (seconds).
        feature_names: List of feature names used.
        params: Hyperparameters used for training.
        scaler: Fitted scaler (if used).
    """
    model: Any
    model_name: str
    training_time: float
    feature_names: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None
    
    def summary(self) -> str:
        """Generate a human-readable summary of training."""
        lines = [
            "=" * 60,
            f"TRAINING RESULT: {self.model_name}",
            "=" * 60,
            f"Training time: {self.training_time:.2f} seconds",
            f"Number of features: {len(self.feature_names)}",
            f"Scaler used: {self.scaler is not None}",
            "-" * 60,
            "HYPERPARAMETERS:",
        ]
        
        for key, value in self.params.items():
            lines.append(f"  {key}: {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class ModelTrainer:
    """
    Trains machine learning models for shipment delay prediction.
    
    This class provides a unified interface for training different
    model types. It handles feature scaling, model fitting, and
    persistence (saving/loading models).
    
    Attributes:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features (optional).
        y_val: Validation target (optional).
        scale_features: Whether to standardize features.
        
    Example:
        >>> trainer = ModelTrainer(X_train, y_train, X_val, y_val)
        >>> result = trainer.train_logistic_regression()
        >>> print(result.summary())
        >>> trainer.save_model(result, "models/logistic.pkl")
    """
    
    # Model type constants
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        scale_features: bool = True,
    ) -> None:
        """
        Initialize the ModelTrainer.
        
        Args:
            X_train: Training features DataFrame.
            y_train: Training target Series.
            X_val: Validation features (optional, for early stopping).
            y_val: Validation target (optional).
            scale_features: Whether to apply standard scaling to features.
                          Recommended for Logistic Regression, optional for trees.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.scale_features = scale_features
        
        self.feature_names = X_train.columns.tolist()
        self.scaler: Optional[StandardScaler] = None
        
        # Prepare scaled data if needed
        if scale_features:
            self._fit_scaler()
        
        logger.info(
            f"ModelTrainer initialized with {len(X_train):,} training samples, "
            f"{len(self.feature_names)} features"
        )
    
    def _fit_scaler(self) -> None:
        """
        Fit a StandardScaler on training data.
        
        StandardScaler transforms features to have mean=0 and std=1.
        This is important for Logistic Regression which is sensitive
        to feature scales. Tree-based models don't need scaling.
        
        The scaler is fit ONLY on training data to prevent data leakage.
        """
        logger.info("Fitting StandardScaler on training data...")
        
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        
        logger.info("Scaler fitted successfully")
    
    def _get_scaled_data(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Apply scaling to features.
        
        Args:
            X: Features to scale.
            
        Returns:
            Scaled features as numpy array.
        """
        if self.scaler is None:
            return X.values
        
        return self.scaler.transform(X)
    
    def train_logistic_regression(
        self,
        params: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        """
        Train a Logistic Regression model.
        
        Logistic Regression is a linear model that predicts probabilities
        using the logistic (sigmoid) function:
        
        P(late) = 1 / (1 + exp(-(b0 + b1*x1 + b2*x2 + ...)))
        
        Advantages:
        - Fast training and prediction
        - Highly interpretable (coefficients show feature importance)
        - Works well with many features
        - Good baseline to compare other models against
        
        Disadvantages:
        - Assumes linear relationship between features and log-odds
        - May underperform on complex non-linear patterns
        
        Args:
            params: Optional custom hyperparameters.
            
        Returns:
            TrainingResult containing the trained model.
        """
        logger.info("Training Logistic Regression model...")
        
        # Use custom params or defaults
        model_params = params or LOGISTIC_REGRESSION_PARAMS.copy()
        
        # Create model
        model = LogisticRegression(**model_params)
        
        # Get scaled training data
        X_scaled = self._get_scaled_data(self.X_train)
        
        # Train with timing
        start_time = time.time()
        model.fit(X_scaled, self.y_train)
        training_time = time.time() - start_time
        
        logger.info(f"Logistic Regression trained in {training_time:.2f}s")
        
        return TrainingResult(
            model=model,
            model_name="Logistic Regression",
            training_time=training_time,
            feature_names=self.feature_names,
            params=model_params,
            scaler=self.scaler,
        )
    
    def train_random_forest(
        self,
        params: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        """
        Train a Random Forest model.
        
        Random Forest is an ensemble of decision trees that:
        1. Creates multiple trees on random subsets of data (bagging)
        2. Uses random subsets of features for each split
        3. Averages predictions across all trees
        
        Advantages:
        - Handles non-linear relationships automatically
        - Robust to outliers and missing values
        - Provides feature importance scores
        - Less prone to overfitting than single trees
        
        Disadvantages:
        - Less interpretable than logistic regression
        - Slower training than linear models
        - May struggle with very high-dimensional sparse data
        
        Args:
            params: Optional custom hyperparameters.
            
        Returns:
            TrainingResult containing the trained model.
        """
        logger.info("Training Random Forest model...")
        
        # Use custom params or defaults
        model_params = params or RANDOM_FOREST_PARAMS.copy()
        
        # Create model
        model = RandomForestClassifier(**model_params)
        
        # Random Forest doesn't need scaling, use raw data
        X_data = self.X_train.values
        
        # Train with timing
        start_time = time.time()
        model.fit(X_data, self.y_train)
        training_time = time.time() - start_time
        
        logger.info(f"Random Forest trained in {training_time:.2f}s")
        
        # Log feature importance preview
        importance = model.feature_importances_
        top_indices = np.argsort(importance)[-5:][::-1]
        logger.info("Top 5 features by importance:")
        for idx in top_indices:
            logger.info(f"  {self.feature_names[idx]}: {importance[idx]:.4f}")
        
        return TrainingResult(
            model=model,
            model_name="Random Forest",
            training_time=training_time,
            feature_names=self.feature_names,
            params=model_params,
            scaler=None,  # RF doesn't use scaling
        )
    
    def train_xgboost(
        self,
        params: Optional[Dict[str, Any]] = None,
        use_early_stopping: bool = True,
        early_stopping_rounds: int = 10,
    ) -> TrainingResult:
        """
        Train an XGBoost model.
        
        XGBoost (eXtreme Gradient Boosting) builds trees sequentially,
        where each tree corrects errors made by previous trees.
        
        Advantages:
        - Often achieves best performance on tabular data
        - Built-in regularization prevents overfitting
        - Handles missing values automatically
        - Supports early stopping with validation set
        
        Disadvantages:
        - Many hyperparameters to tune
        - Can overfit on small datasets
        - Slower than Random Forest for training
        
        Args:
            params: Optional custom hyperparameters.
            use_early_stopping: Whether to use early stopping with validation set.
            early_stopping_rounds: Stop if no improvement for this many rounds.
            
        Returns:
            TrainingResult containing the trained model.
            
        Raises:
            ImportError: If XGBoost is not installed.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )
        
        logger.info("Training XGBoost model...")
        
        # Use custom params or defaults
        model_params = params or XGBOOST_PARAMS.copy()
        
        # Calculate scale_pos_weight for class imbalance
        n_negative = (self.y_train == 0).sum()
        n_positive = (self.y_train == 1).sum()
        scale_pos_weight = n_negative / n_positive
        model_params["scale_pos_weight"] = scale_pos_weight
        
        logger.info(f"Class balance - Negative: {n_negative}, Positive: {n_positive}")
        logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")
        
        # Create model
        model = xgb.XGBClassifier(**model_params)
        
        # Prepare training data (XGBoost doesn't need scaling)
        X_data = self.X_train.values
        
        # Setup early stopping if validation set available
        fit_params: Dict[str, Any] = {}
        if use_early_stopping and self.X_val is not None:
            fit_params["eval_set"] = [(self.X_val.values, self.y_val)]
            fit_params["verbose"] = False
        
        # Train with timing
        start_time = time.time()
        model.fit(X_data, self.y_train, **fit_params)
        training_time = time.time() - start_time
        
        logger.info(f"XGBoost trained in {training_time:.2f}s")
        
        # Log feature importance preview
        importance = model.feature_importances_
        top_indices = np.argsort(importance)[-5:][::-1]
        logger.info("Top 5 features by importance:")
        for idx in top_indices:
            logger.info(f"  {self.feature_names[idx]}: {importance[idx]:.4f}")
        
        return TrainingResult(
            model=model,
            model_name="XGBoost",
            training_time=training_time,
            feature_names=self.feature_names,
            params=model_params,
            scaler=None,  # XGBoost doesn't use scaling
        )
    
    def train_all(self) -> Dict[str, TrainingResult]:
        """
        Train all available models.
        
        Returns:
            Dictionary mapping model names to TrainingResults.
        """
        logger.info("Training all models...")
        
        results = {}
        
        # Logistic Regression
        results[self.LOGISTIC_REGRESSION] = self.train_logistic_regression()
        
        # Random Forest
        results[self.RANDOM_FOREST] = self.train_random_forest()
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            results[self.XGBOOST] = self.train_xgboost()
        else:
            logger.warning("Skipping XGBoost (not installed)")
        
        logger.info(f"Trained {len(results)} models successfully")
        
        return results
    
    @staticmethod
    def save_model(
        result: TrainingResult,
        filepath: Optional[Path] = None,
    ) -> Path:
        """
        Save a trained model to disk.
        
        Saves both the model and the scaler (if used) in a single file
        using joblib. This ensures the model can be loaded and used
        for prediction without needing to refit the scaler.
        
        Args:
            result: TrainingResult containing the model to save.
            filepath: Path to save the model. If None, uses default path.
            
        Returns:
            Path where the model was saved.
        """
        # Determine filepath
        if filepath is None:
            model_name = result.model_name.lower().replace(" ", "_")
            filepath = MODELS_DIR / f"{model_name}.pkl"
        
        filepath = Path(filepath)
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        save_data = {
            "model": result.model,
            "scaler": result.scaler,
            "feature_names": result.feature_names,
            "params": result.params,
            "model_name": result.model_name,
        }
        
        joblib.dump(save_data, filepath)
        
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    @staticmethod
    def load_model(filepath: Path) -> Tuple[Any, Optional[StandardScaler], List[str]]:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file.
            
        Returns:
            Tuple of (model, scaler, feature_names).
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load saved data
        save_data = joblib.load(filepath)
        
        model = save_data["model"]
        scaler = save_data.get("scaler")
        feature_names = save_data.get("feature_names", [])
        
        logger.info(f"Model loaded from {filepath}")
        
        return model, scaler, feature_names


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    models: Optional[List[str]] = None,
    save: bool = True,
) -> Dict[str, TrainingResult]:
    """
    Convenience function for training multiple models.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features (optional).
        y_val: Validation target (optional).
        models: List of model types to train. If None, trains all.
        save: Whether to save models to disk.
        
    Returns:
        Dictionary mapping model names to TrainingResults.
        
    Example:
        >>> results = train_models(X_train, y_train, X_val, y_val)
        >>> for name, result in results.items():
        ...     print(f"{name}: trained in {result.training_time:.1f}s")
    """
    trainer = ModelTrainer(X_train, y_train, X_val, y_val)
    
    # Determine which models to train
    if models is None:
        results = trainer.train_all()
    else:
        results = {}
        for model_type in models:
            if model_type == ModelTrainer.LOGISTIC_REGRESSION:
                results[model_type] = trainer.train_logistic_regression()
            elif model_type == ModelTrainer.RANDOM_FOREST:
                results[model_type] = trainer.train_random_forest()
            elif model_type == ModelTrainer.XGBOOST:
                results[model_type] = trainer.train_xgboost()
            else:
                logger.warning(f"Unknown model type: {model_type}")
    
    # Save models if requested
    if save:
        for model_type, result in results.items():
            ModelTrainer.save_model(result)
    
    return results