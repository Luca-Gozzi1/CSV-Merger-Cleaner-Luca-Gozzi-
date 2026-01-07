"""
Stacking Ensemble for improved prediction accuracy.

Stacking combines multiple base models by training a meta-model
on their predictions, often achieving 2-5% better accuracy than
any single model alone.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Try to import optional models
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Container for ensemble training results."""
    model: Any
    model_name: str
    training_time: float
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    feature_names: List[str]
    scaler: Optional[StandardScaler]


class EnsembleTrainer:
    """
    Train ensemble models for improved accuracy.
    
    Implements two ensemble strategies:
    1. Stacking: Meta-learner trained on base model predictions
    2. Voting: Weighted average of base model predictions
    
    Attributes:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        scaler: Fitted StandardScaler
        feature_names: List of feature names
    """
    
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        scale_features: bool = True,
    ) -> None:
        """
        Initialize the ensemble trainer.
        
        Args:
            X_train: Training features DataFrame
            y_train: Training target Series
            X_val: Optional validation features
            y_val: Optional validation target
            scale_features: Whether to standardize features
        """
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=self.feature_names,
                index=X_train.index
            )
            if X_val is not None:
                self.X_val = pd.DataFrame(
                    self.scaler.transform(X_val),
                    columns=self.feature_names,
                    index=X_val.index
                )
            else:
                self.X_val = None
        else:
            self.scaler = None
            self.X_train = X_train
            self.X_val = X_val
        
        self.y_train = y_train
        self.y_val = y_val
        
        logger.info(f"EnsembleTrainer initialized with {len(self.feature_names)} features")
    
    def _get_base_estimators(self) -> List[Tuple[str, Any]]:
        """
        Create base estimators for ensemble.
        
        Returns:
            List of (name, estimator) tuples
        """
        estimators = []
        
        # 1. Logistic Regression (linear model - different perspective)
        lr = LogisticRegression(
            C=0.5,
            max_iter=2000,
            random_state=RANDOM_SEED,
            solver='saga',
            class_weight='balanced',
            n_jobs=-1,
        )
        estimators.append(('lr', lr))
        
        # 2. Random Forest (bagging ensemble)
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_SEED,
            class_weight='balanced_subsample',
            n_jobs=-1,
        )
        estimators.append(('rf', rf))
        
        # 3. Gradient Boosting (sklearn version - stable)
        gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )
        estimators.append(('gb', gb))
        
        # 4. XGBoost (if available)
        if HAS_XGBOOST:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                random_state=RANDOM_SEED,
                eval_metric='logloss',
                use_label_encoder=False,
            )
            estimators.append(('xgb', xgb_model))
        
        # 5. LightGBM (if available - often best performer)
        if HAS_LIGHTGBM:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                class_weight='balanced',
                verbose=-1,  # Suppress warnings
                force_col_wise=True,
            )
            estimators.append(('lgb', lgb_model))
        
        logger.info(f"Created {len(estimators)} base estimators: {[e[0] for e in estimators]}")
        return estimators
    
    def train_stacking(
        self,
        cv: int = 5,
        use_proba: bool = True,
    ) -> EnsembleResult:
        """
        Train a stacking ensemble classifier.
        
        Stacking uses predictions from base models as input features
        for a meta-learner (final estimator). This captures complementary
        patterns from different model types.
        
        Args:
            cv: Number of cross-validation folds for base models
            use_proba: Use predicted probabilities (better) vs class labels
        
        Returns:
            EnsembleResult with trained stacking model
        """
        logger.info("Training Stacking Ensemble...")
        start_time = time.time()
        
        # Get base estimators
        estimators = self._get_base_estimators()
        
        # Meta-learner: Logistic Regression (works well for stacking)
        meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
        
        # Create stacking classifier
        stack_method = 'predict_proba' if use_proba else 'predict'
        
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv,
            stack_method=stack_method,
            n_jobs=-1,
            passthrough=False,  # Only use model predictions, not original features
        )
        
        # Train
        logger.info(f"Fitting stacking ensemble with {len(estimators)} base models...")
        stacking.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        logger.info(f"Stacking ensemble trained in {training_time:.2f}s")
        
        # Cross-validation score
        logger.info("Calculating cross-validation scores...")
        cv_scores = cross_val_score(
            stacking, self.X_train, self.y_train,
            cv=3,  # Use 3 folds for speed
            scoring='accuracy',
            n_jobs=-1,
        )
        
        logger.info(f"Stacking CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return EnsembleResult(
            model=stacking,
            model_name="Stacking Ensemble",
            training_time=training_time,
            cv_scores=cv_scores,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            feature_names=self.feature_names,
            scaler=self.scaler,
        )
    
    def train_voting(
        self,
        voting: str = 'soft',
        weights: Optional[List[float]] = None,
    ) -> EnsembleResult:
        """
        Train a voting ensemble classifier.
        
        Voting combines predictions through majority vote (hard) or
        averaged probabilities (soft).
        
        Args:
            voting: 'hard' for majority vote, 'soft' for probability averaging
            weights: Optional weights for each base model
        
        Returns:
            EnsembleResult with trained voting model
        """
        logger.info(f"Training Voting Ensemble ({voting})...")
        start_time = time.time()
        
        # Get base estimators
        estimators = self._get_base_estimators()
        
        # Default weights: give more weight to tree-based models
        if weights is None:
            weights = [1.0] * len(estimators)
            # Increase weight for lgb and xgb if available
            for i, (name, _) in enumerate(estimators):
                if name in ['lgb', 'xgb', 'rf']:
                    weights[i] = 1.5
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1,
        )
        
        # Train
        logger.info(f"Fitting voting ensemble with weights {weights}...")
        voting_clf.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        logger.info(f"Voting ensemble trained in {training_time:.2f}s")
        
        # Cross-validation score
        cv_scores = cross_val_score(
            voting_clf, self.X_train, self.y_train,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
        )
        
        logger.info(f"Voting CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return EnsembleResult(
            model=voting_clf,
            model_name=f"Voting Ensemble ({voting})",
            training_time=training_time,
            cv_scores=cv_scores,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            feature_names=self.feature_names,
            scaler=self.scaler,
        )
    
    def train_lightgbm_optimized(self) -> EnsembleResult:
        """
        Train an optimized LightGBM model.
        
        LightGBM often outperforms XGBoost and Random Forest on tabular data.
        This method uses optimized hyperparameters.
        
        Returns:
            EnsembleResult with trained LightGBM model
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")
        
        logger.info("Training Optimized LightGBM...")
        start_time = time.time()
        
        # Optimized parameters for accuracy
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=12,
            learning_rate=0.03,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_SEED,
            class_weight='balanced',
            verbose=-1,
            force_col_wise=True,
            n_jobs=-1,
        )
        
        # Train with early stopping if validation set available
        if self.X_val is not None and self.y_val is not None:
            lgb_model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),  # Suppress output
                ],
            )
            logger.info(f"LightGBM stopped at iteration {lgb_model.best_iteration_}")
        else:
            lgb_model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        logger.info(f"LightGBM trained in {training_time:.2f}s")
        
        # Cross-validation score
        cv_scores = cross_val_score(
            lgb_model, self.X_train, self.y_train,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
        )
        
        logger.info(f"LightGBM CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return EnsembleResult(
            model=lgb_model,
            model_name="LightGBM Optimized",
            training_time=training_time,
            cv_scores=cv_scores,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            feature_names=self.feature_names,
            scaler=self.scaler,
        )
    
    @staticmethod
    def save_model(result: EnsembleResult, filepath: Path) -> None:
        """Save ensemble model to disk."""
        save_dict = {
            'model': result.model,
            'model_name': result.model_name,
            'scaler': result.scaler,
            'feature_names': result.feature_names,
            'cv_mean': result.cv_mean,
            'cv_std': result.cv_std,
        }
        joblib.dump(save_dict, filepath)
        logger.info(f"Saved ensemble model to {filepath}")
    
    @staticmethod
    def load_model(filepath: Path) -> Tuple[Any, Optional[StandardScaler], List[str]]:
        """Load ensemble model from disk."""
        save_dict = joblib.load(filepath)
        logger.info(f"Loaded ensemble model from {filepath}")
        return (
            save_dict['model'],
            save_dict['scaler'],
            save_dict['feature_names'],
        )


def evaluate_ensemble(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scaler: Optional[StandardScaler] = None,
    model_name: str = "Ensemble",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate ensemble model performance.
    
    Args:
        model: Trained ensemble model
        X: Features DataFrame
        y: Target Series
        scaler: Optional scaler to apply
        model_name: Name for logging
        threshold: Classification threshold
    
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )
    
    # Prepare features
    if scaler is not None:
        X_prepared = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        X_prepared = X
    
    # Get predictions
    y_proba = model.predict_proba(X_prepared)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    y_true = y.values if isinstance(y, pd.Series) else y
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'threshold': threshold,
    }
    
    logger.info(
        f"{model_name}: Accuracy={metrics['accuracy']:.4f}, "
        f"F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}"
    )
    
    return metrics