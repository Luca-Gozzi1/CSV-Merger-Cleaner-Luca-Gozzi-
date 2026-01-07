"""
Model evaluation module for Supply Chain Explorer.

This module provides the ModelEvaluator class responsible for evaluating
trained models using various metrics and generating visualizations.

Evaluation includes:
1. Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
2. Confusion matrices
3. ROC curves
4. Feature importance analysis
5. Model comparison

Author: Luca Gozzi
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from src.config import (
    TARGET_COLUMN,
    RESULTS_DIR,
    FIGURES_DIR,
    PRIMARY_METRIC,
    MIN_ACCURACY,
    MIN_RECALL,
    MIN_ROC_AUC,
)


# Configure module logger
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class EvaluationResult:
    """
    Container for model evaluation results.
    
    Holds all metrics and metadata from evaluating a model
    on a specific dataset.
    
    Attributes:
        model_name: Name of the evaluated model.
        dataset_name: Name of dataset (e.g., 'validation', 'test').
        metrics: Dictionary of metric names to values.
        confusion_matrix: Confusion matrix as numpy array.
        y_true: Actual target values.
        y_pred: Predicted target values.
        y_proba: Predicted probabilities (if available).
    """
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None
    
    def summary(self) -> str:
        """Generate a human-readable summary of evaluation."""
        lines = [
            "=" * 60,
            f"EVALUATION: {self.model_name} on {self.dataset_name}",
            "=" * 60,
            "METRICS:",
        ]
        
        for metric_name, value in self.metrics.items():
            lines.append(f"  {metric_name}: {value:.4f}")
        
        lines.append("-" * 60)
        lines.append("CONFUSION MATRIX:")
        lines.append("                 Predicted")
        lines.append("                 On-Time  Late")
        lines.append(f"  Actual On-Time   {self.confusion_matrix[0, 0]:5d}   {self.confusion_matrix[0, 1]:5d}")
        lines.append(f"  Actual Late      {self.confusion_matrix[1, 0]:5d}   {self.confusion_matrix[1, 1]:5d}")
        
        lines.append("-" * 60)
        
        # Interpretation
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        lines.append("INTERPRETATION:")
        lines.append(f"  True Negatives (correctly predicted on-time): {tn:,}")
        lines.append(f"  False Positives (on-time predicted as late): {fp:,}")
        lines.append(f"  False Negatives (late predicted as on-time): {fn:,}")
        lines.append(f"  True Positives (correctly predicted late): {tp:,}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def passes_thresholds(self) -> Dict[str, bool]:
        """Check if metrics pass minimum thresholds."""
        return {
            "accuracy": self.metrics.get("accuracy", 0) >= MIN_ACCURACY,
            "recall": self.metrics.get("recall", 0) >= MIN_RECALL,
            "roc_auc": self.metrics.get("roc_auc", 0) >= MIN_ROC_AUC,
        }


class ModelEvaluator:
    """
    Evaluates machine learning models for shipment delay prediction.
    
    This class provides comprehensive evaluation including metrics
    calculation, visualization generation, and model comparison.
    
    Attributes:
        model: Trained model to evaluate.
        model_name: Name of the model for labeling.
        scaler: Optional scaler to apply to features.
        feature_names: List of feature names.
        
    Example:
        >>> evaluator = ModelEvaluator(model, "Random Forest", scaler, feature_names)
        >>> result = evaluator.evaluate(X_test, y_test, "test")
        >>> print(result.summary())
        >>> evaluator.plot_confusion_matrix(result)
    """
    
    def __init__(
        self,
        model: Any,
        model_name: str,
        scaler: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the ModelEvaluator.
        
        Args:
            model: Trained model with predict() and predict_proba() methods.
            model_name: Name of the model for labeling outputs.
            scaler: Optional scaler to transform features before prediction.
            feature_names: List of feature names for importance analysis.
        """
        self.model = model
        self.model_name = model_name
        self.scaler = scaler
        self.feature_names = feature_names or []
        
        logger.info(f"ModelEvaluator initialized for {model_name}")
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.
        
        Applies scaling if a scaler is provided. Preserves DataFrame format
        to avoid sklearn warnings about feature names.
        
        Args:
            X: Features DataFrame.
            
        Returns:
            Prepared features as DataFrame (preserves feature names).
        """
        if self.scaler is not None:
            # Scaler returns numpy array, wrap it back into DataFrame
            scaled_values = self.scaler.transform(X)
            return pd.DataFrame(scaled_values, columns=X.columns, index=X.index)
        
        # Return DataFrame as-is to preserve feature names
        return X
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "test",
    ) -> EvaluationResult:
        """
        Evaluate the model on a dataset.
        
        Calculates all classification metrics and creates confusion matrix.
        
        Args:
            X: Features DataFrame.
            y: Target Series.
            dataset_name: Name for labeling (e.g., 'validation', 'test').
            
        Returns:
            EvaluationResult containing all metrics and predictions.
        """
        logger.info(f"Evaluating {self.model_name} on {dataset_name} set...")
        
        # Prepare features (preserves DataFrame format to avoid warnings)
        X_prepared = self._prepare_features(X)
        
        # Get predictions
        y_pred = self.model.predict(X_prepared)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_prepared)[:, 1]
        
        # Convert to numpy arrays
        y_true = y.values if isinstance(y, pd.Series) else y
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_proba)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
        
        return EvaluationResult(
            model_name=self.model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            confusion_matrix=cm,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: Actual target values.
            y_pred: Predicted target values.
            y_proba: Predicted probabilities (optional).
            
        Returns:
            Dictionary of metric names to values.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "specificity": self._calculate_specificity(y_true, y_pred),
        }
        
        # ROC-AUC requires probabilities
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            metrics["average_precision"] = average_precision_score(y_true, y_proba)
        
        return metrics
    
    def _calculate_specificity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Calculate specificity (true negative rate).
        
        Specificity = TN / (TN + FP)
        
        This measures how well the model identifies on-time deliveries.
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        if (tn + fp) == 0:
            return 0.0
        
        return tn / (tn + fp)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the model.
        
        Works with models that have feature_importances_ attribute
        (Random Forest, XGBoost, etc.) or coef_ attribute (Logistic Regression).
        
        Returns:
            DataFrame with feature names and importance scores, sorted descending.
            None if model doesn't support feature importance.
        """
        importance = None
        
        # Tree-based models
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        
        # Linear models (use absolute coefficient values)
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_).flatten()
        
        if importance is None:
            logger.warning(f"Model {self.model_name} doesn't support feature importance")
            return None
        
        # Create DataFrame
        if len(self.feature_names) != len(importance):
            logger.warning("Feature names length doesn't match importance length")
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        else:
            feature_names = self.feature_names
        
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        })
        
        # Sort by importance
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        
        # Add rank
        df["rank"] = range(1, len(df) + 1)
        
        return df
    
    def plot_confusion_matrix(
        self,
        result: EvaluationResult,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            result: EvaluationResult containing confusion matrix.
            save_path: Path to save the figure (optional).
            figsize: Figure size in inches.
            
        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            result.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["On-Time", "Late"],
            yticklabels=["On-Time", "Late"],
            ax=ax,
            annot_kws={"size": 14},
        )
        
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("Actual Label", fontsize=12)
        ax.set_title(
            f"Confusion Matrix: {result.model_name}\n({result.dataset_name} set)",
            fontsize=14,
        )
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(
        self,
        result: EvaluationResult,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> Optional[plt.Figure]:
        """
        Plot ROC curve.
        
        Args:
            result: EvaluationResult containing predictions.
            save_path: Path to save the figure (optional).
            figsize: Figure size in inches.
            
        Returns:
            Matplotlib Figure object, or None if probabilities not available.
        """
        if result.y_proba is None:
            logger.warning("Cannot plot ROC curve: no probability predictions available")
            return None
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(result.y_true, result.y_proba)
        roc_auc = result.metrics.get("roc_auc", 0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(
            fpr, tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})",
        )
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(
            f"ROC Curve: {result.model_name}\n({result.dataset_name} set)",
            fontsize=14,
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        result: EvaluationResult,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> Optional[plt.Figure]:
        """
        Plot Precision-Recall curve.
        
        This is especially useful for imbalanced datasets.
        
        Args:
            result: EvaluationResult containing predictions.
            save_path: Path to save the figure (optional).
            figsize: Figure size in inches.
            
        Returns:
            Matplotlib Figure object, or None if probabilities not available.
        """
        if result.y_proba is None:
            logger.warning("Cannot plot PR curve: no probability predictions available")
            return None
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(
            result.y_true, result.y_proba
        )
        avg_precision = result.metrics.get("average_precision", 0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot PR curve
        ax.plot(
            recall, precision,
            color="darkorange",
            lw=2,
            label=f"PR curve (AP = {avg_precision:.4f})",
        )
        
        # Plot baseline (proportion of positive class)
        baseline = result.y_true.mean()
        ax.axhline(y=baseline, color="navy", lw=2, linestyle="--", label=f"Baseline ({baseline:.2f})")
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(
            f"Precision-Recall Curve: {result.model_name}\n({result.dataset_name} set)",
            fontsize=14,
        )
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"PR curve saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Optional[plt.Figure]:
        """
        Plot feature importance as horizontal bar chart.
        
        Args:
            top_n: Number of top features to show.
            save_path: Path to save the figure (optional).
            figsize: Figure size in inches.
            
        Returns:
            Matplotlib Figure object, or None if importance not available.
        """
        importance_df = self.get_feature_importance()
        
        if importance_df is None:
            return None
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart (reversed for top at top)
        y_pos = range(len(top_features) - 1, -1, -1)
        
        bars = ax.barh(
            y_pos,
            top_features["importance"],
            color="steelblue",
            edgecolor="navy",
        )
        
        # Add value labels
        for bar, importance in zip(bars, top_features["importance"]):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{importance:.4f}",
                va="center",
                fontsize=9,
            )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features["feature"])
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(
            f"Top {top_n} Feature Importance: {self.model_name}",
            fontsize=14,
        )
        ax.grid(True, axis="x", alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig


def compare_models(
    results: List[EvaluationResult],
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create a comparison table of multiple model results.
    
    Args:
        results: List of EvaluationResult objects.
        save_path: Path to save the comparison as CSV (optional).
        
    Returns:
        DataFrame with models as rows and metrics as columns.
    """
    comparison_data = []
    
    for result in results:
        row = {
            "Model": result.model_name,
            "Dataset": result.dataset_name,
            **result.metrics,
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Model comparison saved to {save_path}")
    
    return df


def plot_model_comparison(
    results: List[EvaluationResult],
    metrics: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Create bar chart comparing models across metrics.
    
    Args:
        results: List of EvaluationResult objects.
        metrics: List of metric names to compare. If None, uses default set.
        save_path: Path to save the figure (optional).
        figsize: Figure size in inches.
        
    Returns:
        Matplotlib Figure object.
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    
    # Filter to metrics that exist in all results
    available_metrics = []
    for metric in metrics:
        if all(metric in r.metrics for r in results):
            available_metrics.append(metric)
    
    if not available_metrics:
        logger.warning("No common metrics found for comparison")
        return None
    
    # Prepare data
    model_names = [r.model_name for r in results]
    n_models = len(model_names)
    n_metrics = len(available_metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up bar positions
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    # Plot bars for each model
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for i, result in enumerate(results):
        values = [result.metrics[m] for m in available_metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=result.model_name, color=colors[i])
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )
    
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison Across Metrics", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in available_metrics])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Model comparison plot saved to {save_path}")
    
    return fig

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Find the optimal classification threshold for a given metric.
    
    By default, classifiers use 0.5 as the threshold, but this may not
    be optimal for imbalanced datasets or specific business requirements.
    
    Args:
        y_true: Actual target values (0 or 1).
        y_proba: Predicted probabilities for class 1.
        metric: Metric to optimize. Options:
                - 'f1': F1 score (balance precision and recall)
                - 'accuracy': Overall accuracy
                - 'balanced': Balanced accuracy (good for imbalanced data)
                - 'recall': Maximize recall (catch all late deliveries)
                - 'precision': Maximize precision (reduce false alarms)
        thresholds: Array of thresholds to try. Default: 0.1 to 0.9 in 0.05 steps.
    
    Returns:
        Tuple of (optimal_threshold, best_score, results_dataframe)
    
    Example:
        >>> optimal_thresh, score, df = find_optimal_threshold(y_true, y_proba, 'f1')
        >>> print(f"Optimal threshold: {optimal_thresh:.2f} with F1={score:.4f}")
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.05)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate all metrics for this threshold
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Balanced accuracy
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (rec + specificity) / 2
        
        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'specificity': specificity,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold based on selected metric
    metric_map = {
        'f1': 'f1_score',
        'accuracy': 'accuracy',
        'balanced': 'balanced_accuracy',
        'recall': 'recall',
        'precision': 'precision',
    }
    
    metric_col = metric_map.get(metric, 'f1_score')
    best_idx = results_df[metric_col].idxmax()
    best_row = results_df.loc[best_idx]
    
    optimal_threshold = best_row['threshold']
    best_score = best_row[metric_col]
    
    logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.2f} (score: {best_score:.4f})")
    
    return optimal_threshold, best_score, results_df


def evaluate_with_threshold(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    model_name: str = "Model",
    scaler: Optional[Any] = None,
) -> EvaluationResult:
    """
    Evaluate a model using a custom classification threshold.
    
    Args:
        model: Trained model with predict_proba method.
        X: Features DataFrame.
        y: Target Series.
        threshold: Classification threshold (default 0.5).
        model_name: Name for labeling.
        scaler: Optional scaler to apply to features.
    
    Returns:
        EvaluationResult with metrics calculated using the custom threshold.
    """
    # Apply scaler if provided
    if scaler is not None:
        X_prepared = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        X_prepared = X
    
    # Get probabilities
    y_proba = model.predict_proba(X_prepared)[:, 1]
    
    # Apply custom threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Convert to numpy
    y_true = y.values if isinstance(y, pd.Series) else y
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "threshold_used": threshold,
    }
    
    # Specificity
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    logger.info(
        f"{model_name} with threshold={threshold:.2f}: "
        f"Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}"
    )
    
    return EvaluationResult(
        model_name=f"{model_name} (t={threshold:.2f})",
        dataset_name="test",
        metrics=metrics,
        confusion_matrix=cm,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

def plot_roc_curves_comparison(
    results: List[EvaluationResult],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on the same axes.
    
    Args:
        results: List of EvaluationResult objects.
        save_path: Path to save the figure (optional).
        figsize: Figure size in inches.
        
    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for result, color in zip(results, colors):
        if result.y_proba is None:
            continue
        
        fpr, tpr, _ = roc_curve(result.y_true, result.y_proba)
        roc_auc = result.metrics.get("roc_auc", 0)
        
        ax.plot(
            fpr, tpr,
            color=color,
            lw=2,
            label=f"{result.model_name} (AUC = {roc_auc:.4f})",
        )
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC comparison plot saved to {save_path}")
    
    return fig