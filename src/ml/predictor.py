"""
Prediction module for Supply Chain Explorer.

This module provides the DelayPredictor class for making predictions
on new shipment data using trained models. It handles feature preparation,
model loading, and risk score calculation.

The predictor integrates with the analytics modules to provide
delay risk assessments for inventory, shipment, and vendor analysis.

Author: Luca Gozzi
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import pandas as pd
import numpy as np

from src.config import (
    TARGET_COLUMN,
    MODELS_DIR,
    RISK_THRESHOLDS,
)
from src.ml.trainer import ModelTrainer


# Configure module logger
logger = logging.getLogger(__name__)


class DelayPredictor:
    """
    Makes delay predictions on new shipment data.
    
    This class loads a trained model and provides methods to:
    1. Predict delay probability for individual shipments
    2. Assign risk categories (low/medium/high)
    3. Generate risk scores for integration with analytics modules
    
    Attributes:
        model: Loaded trained model.
        scaler: Loaded scaler (if applicable).
        feature_names: List of expected feature names.
        model_name: Name of the loaded model.
        
    Example:
        >>> predictor = DelayPredictor.from_saved_model("random_forest")
        >>> predictions = predictor.predict(new_shipments_df)
        >>> risk_scores = predictor.get_risk_scores(new_shipments_df)
    """
    
    # Risk category thresholds
    LOW_RISK_THRESHOLD = RISK_THRESHOLDS.get("low", 0.3)
    HIGH_RISK_THRESHOLD = RISK_THRESHOLDS.get("high", 0.6)
    
    def __init__(
        self,
        model: Any,
        scaler: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        model_name: str = "Unknown",
    ) -> None:
        """
        Initialize the DelayPredictor.
        
        Args:
            model: Trained model with predict() and predict_proba() methods.
            scaler: Optional scaler for feature preprocessing.
            feature_names: List of expected feature names.
            model_name: Name of the model for logging.
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.model_name = model_name
        
        logger.info(f"DelayPredictor initialized with {model_name}")
    
    @classmethod
    def from_saved_model(
        cls,
        model_name: str = "random_forest",
        models_dir: Optional[Path] = None,
    ) -> "DelayPredictor":
        """
        Create a DelayPredictor from a saved model file.
        
        Args:
            model_name: Name of the model (e.g., 'random_forest', 'logistic_regression', 'xgboost').
            models_dir: Directory containing saved models.
            
        Returns:
            DelayPredictor instance with loaded model.
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        models_dir = models_dir or MODELS_DIR
        
        # Normalize model name to filename
        filename = model_name.lower().replace(" ", "_") + ".pkl"
        filepath = Path(models_dir) / filename
        
        logger.info(f"Loading model from {filepath}")
        
        model, scaler, feature_names = ModelTrainer.load_model(filepath)
        
        return cls(
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            model_name=model_name,
        )
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.
        
        Ensures features are in the correct order and applies scaling
        if necessary.
        
        Args:
            X: Features DataFrame.
            
        Returns:
            Prepared features DataFrame.
        """
        # Check for missing features
        if self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                logger.warning(f"Missing features: {missing}")
            
            # Reorder columns to match training
            available = [f for f in self.feature_names if f in X.columns]
            X = X[available]
        
        # Apply scaling if scaler exists
        if self.scaler is not None:
            scaled_values = self.scaler.transform(X)
            X = pd.DataFrame(scaled_values, columns=X.columns, index=X.index)
        
        return X
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict delay class (0=on-time, 1=late).
        
        Args:
            X: Features DataFrame.
            
        Returns:
            Array of predictions (0 or 1).
        """
        X_prepared = self._prepare_features(X)
        return self.model.predict(X_prepared)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict delay probability.
        
        Args:
            X: Features DataFrame.
            
        Returns:
            Array of probabilities (0.0 to 1.0) for late delivery.
        """
        X_prepared = self._prepare_features(X)
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_prepared)[:, 1]
        else:
            # Fallback for models without predict_proba
            logger.warning("Model doesn't support predict_proba, using predict")
            return self.model.predict(X_prepared).astype(float)
    
    def get_risk_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores and categories for shipments.
        
        Returns a DataFrame with:
        - delay_probability: Float 0-1
        - delay_prediction: Binary 0/1
        - risk_category: 'low', 'medium', or 'high'
        - risk_score: Scaled 0-100 score
        
        Args:
            X: Features DataFrame.
            
        Returns:
            DataFrame with risk assessments.
        """
        probabilities = self.predict_proba(X)
        predictions = self.predict(X)
        
        # Create risk DataFrame
        risk_df = pd.DataFrame({
            "delay_probability": probabilities,
            "delay_prediction": predictions,
            "risk_score": (probabilities * 100).round(1),
        }, index=X.index)
        
        # Assign risk categories
        risk_df["risk_category"] = pd.cut(
            probabilities,
            bins=[-0.01, self.LOW_RISK_THRESHOLD, self.HIGH_RISK_THRESHOLD, 1.01],
            labels=["low", "medium", "high"],
        )
        
        return risk_df
    
    def add_predictions_to_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Add prediction columns to an existing DataFrame.
        
        This is useful for enriching shipment data with risk assessments.
        
        Args:
            df: Original DataFrame with features.
            feature_columns: Columns to use as features. If None, uses all
                           columns except target.
            
        Returns:
            DataFrame with added prediction columns.
        """
        df = df.copy()
        
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != TARGET_COLUMN]
        
        # Get features
        X = df[feature_columns]
        
        # Calculate risk scores
        risk_df = self.get_risk_scores(X)
        
        # Add to original DataFrame
        df["delay_risk_probability"] = risk_df["delay_probability"]
        df["delay_risk_prediction"] = risk_df["delay_prediction"]
        df["delay_risk_category"] = risk_df["risk_category"]
        df["delay_risk_score"] = risk_df["risk_score"]
        
        logger.info(f"Added prediction columns to DataFrame ({len(df)} rows)")
        
        return df
    
    def get_high_risk_shipments(
        self,
        df: pd.DataFrame,
        threshold: float = None,
    ) -> pd.DataFrame:
        """
        Filter to high-risk shipments.
        
        Args:
            df: DataFrame with features (or already has predictions).
            threshold: Probability threshold for high risk.
            
        Returns:
            DataFrame containing only high-risk shipments.
        """
        threshold = threshold or self.HIGH_RISK_THRESHOLD
        
        # Add predictions if not present
        if "delay_risk_probability" not in df.columns:
            df = self.add_predictions_to_dataframe(df)
        
        # Filter to high risk
        high_risk = df[df["delay_risk_probability"] >= threshold].copy()
        
        # Sort by risk (highest first)
        high_risk = high_risk.sort_values("delay_risk_probability", ascending=False)
        
        logger.info(f"Found {len(high_risk)} high-risk shipments (>= {threshold:.0%})")
        
        return high_risk
    
    def summarize_risk_distribution(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics of risk distribution.
        
        Args:
            X: Features DataFrame.
            
        Returns:
            Dictionary with risk distribution statistics.
        """
        risk_df = self.get_risk_scores(X)
        
        summary = {
            "total_shipments": len(X),
            "predicted_late": int(risk_df["delay_prediction"].sum()),
            "predicted_on_time": int((1 - risk_df["delay_prediction"]).sum()),
            "average_risk_score": float(risk_df["risk_score"].mean()),
            "median_risk_score": float(risk_df["risk_score"].median()),
            "risk_category_counts": risk_df["risk_category"].value_counts().to_dict(),
            "high_risk_count": int((risk_df["risk_category"] == "high").sum()),
            "high_risk_percentage": float((risk_df["risk_category"] == "high").mean() * 100),
        }
        
        return summary


def load_predictor(model_name: str = "random_forest") -> DelayPredictor:
    """
    Convenience function to load a predictor.
    
    Args:
        model_name: Name of model to load.
        
    Returns:
        DelayPredictor instance.
    """
    return DelayPredictor.from_saved_model(model_name)