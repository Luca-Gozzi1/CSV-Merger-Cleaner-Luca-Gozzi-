"""
Machine Learning module for Supply Chain Explorer.

This module contains utilities for data splitting, model training,
evaluation, and prediction for shipment delay classification.
"""

from src.ml.splitter import DataSplitter, SplitResult, split_data, get_X_y
from src.ml.trainer import (
    ModelTrainer,
    TrainingResult,
    train_models,
    XGBOOST_AVAILABLE,
)
from src.ml.evaluator import (
    ModelEvaluator,
    EvaluationResult,
    compare_models,
    plot_model_comparison,
    plot_roc_curves_comparison,
)
from src.ml.predictor import (
    DelayPredictor,
    load_predictor,
)

__all__ = [
    # Splitter
    "DataSplitter",
    "SplitResult",
    "split_data",
    "get_X_y",
    # Trainer
    "ModelTrainer",
    "TrainingResult",
    "train_models",
    "XGBOOST_AVAILABLE",
    # Evaluator
    "ModelEvaluator",
    "EvaluationResult",
    "compare_models",
    "plot_model_comparison",
    "plot_roc_curves_comparison",
    # Predictor
    "DelayPredictor",
    "load_predictor",
]