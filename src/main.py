"""
Main orchestrator for the Supply Chain Explorer pipeline.

This module provides the SupplyChainPipeline class that coordinates
all stages of the data pipeline: loading, validation, preprocessing,
feature engineering, model training, evaluation, and prediction.

Run with: python -m src.main --mode full

Author: Luca Gozzi
Date: November 2025
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.ml.splitter import DataSplitter, get_X_y
from src.ml.trainer import ModelTrainer
from src.ml.evaluator import ModelEvaluator, compare_models, plot_model_comparison, plot_roc_curves_comparison
from src.ml.predictor import load_predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupplyChainPipeline:
    """
    Main orchestrator for the Supply Chain Explorer pipeline.
    
    Coordinates all stages of data processing and ML pipeline:
    1. Data loading
    2. Data validation
    3. Data preprocessing
    4. Feature engineering
    5. Model training
    6. Model evaluation
    7. Prediction
    
    Attributes:
        project_dir: Root directory of the project
        data_dir: Directory containing raw and processed data
        models_dir: Directory for saving trained models
        results_dir: Directory for saving results and reports
        figures_dir: Directory for saving figures and plots
    """
    
    def __init__(self, project_dir: Optional[Path] = None):
        """
        Initialize the pipeline.
        
        Args:
            project_dir: Root directory of the project. Defaults to current directory.
        """
        self.project_dir = project_dir or Path(__file__).parent.parent
        self.data_dir = self.project_dir / "data"
        self.models_dir = self.project_dir / "models"
        self.results_dir = self.project_dir / "results"
        self.figures_dir = self.results_dir / "figures"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized with project_dir: {self.project_dir}")
    
    def _check_models_exist(self) -> bool:
        """
        Check if all trained models exist.
        
        Returns:
            True if all models exist, False otherwise.
        """
        model_names = ["logistic_regression.pkl", "random_forest.pkl", "xgboost.pkl"]
        return all((self.models_dir / name).exists() for name in model_names)
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load the supply chain data.
        
        Returns:
            DataFrame with loaded data.
        """
        logger.info("Loading data...")
        
        # Find CSV file in raw directory
        raw_dir = self.data_dir / "raw"
        csv_files = list(raw_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {raw_dir}")
        
        # Use the first CSV file found (or the main dataset)
        csv_path = csv_files[0]
        for f in csv_files:
            if "DataCo" in f.name or "supply" in f.name.lower():
                csv_path = f
                break
        
        logger.info(f"Using data file: {csv_path.name}")
        loader = DataLoader(csv_path)
        df = loader.load()
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the loaded data.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Dictionary with validation results.
        """
        logger.info("Validating data...")
        validator = DataValidator(df)
        report = validator.validate_all()
        
        # Handle different report types
        if hasattr(report, 'is_valid'):
            is_valid = report.is_valid
        else:
            is_valid = True
        
        logger.info(f"Validation complete. Is valid: {is_valid}")
        return {"is_valid": is_valid, "report": report}
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data.
        
        Args:
            df: DataFrame to preprocess.
            
        Returns:
            Preprocessed DataFrame.
        """
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(df)
        df_processed = preprocessor.preprocess()
        logger.info(f"Preprocessing complete. Shape: {df_processed.shape}")
        return df_processed
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from the data.
        
        Args:
            df: DataFrame for feature engineering.
            
        Returns:
            DataFrame with engineered features.
        """
        logger.info("Engineering features...")
        engineer = FeatureEngineer(df)
        df_features = engineer.engineer_all()
        logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
        return df_features
    
    def _train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models.
        
        Args:
            df: DataFrame with features for training.
            
        Returns:
            Dictionary with trained models.
        """
        logger.info("Training models...")
        
        # Split data - use correct parameter names
        splitter = DataSplitter(
            df,
            date_column="order date (DateOrders)",
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15
        )
        splits = splitter.split(method="time")
        splitter.save_splits(splits, self.data_dir / "processed")
        
        # Get X, y for training and validation
        X_train, y_train = get_X_y(splits.train)
        X_val, y_val = get_X_y(splits.validation)
        
        # Create trainer with training data
        trainer = ModelTrainer(X_train, y_train, X_val, y_val)
        
        trained_results = {}
        
        # Train Logistic Regression
        logger.info("Training logistic_regression...")
        lr_result = trainer.train_logistic_regression()
        ModelTrainer.save_model(lr_result, self.models_dir / "logistic_regression.pkl")
        trained_results["logistic_regression"] = lr_result
        logger.info("logistic_regression trained and saved")
        
        # Train Random Forest
        logger.info("Training random_forest...")
        rf_result = trainer.train_random_forest()
        ModelTrainer.save_model(rf_result, self.models_dir / "random_forest.pkl")
        trained_results["random_forest"] = rf_result
        logger.info("random_forest trained and saved")
        
        # Train XGBoost
        logger.info("Training xgboost...")
        xgb_result = trainer.train_xgboost()
        ModelTrainer.save_model(xgb_result, self.models_dir / "xgboost.pkl")
        trained_results["xgboost"] = xgb_result
        logger.info("xgboost trained and saved")
        
        return trained_results
    
    def _evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate all trained models with optimal threshold tuning.
        
        Returns:
            Dictionary with evaluation results for each model.
        """
        logger.info("Evaluating models...")
        
        # Initialize validation variables (prevents linter warnings)
        X_val: Optional[pd.DataFrame] = None
        y_val: Optional[pd.Series] = None
        use_threshold_tuning = False
        
        # Load test data from saved CSV files
        try:
            processed_dir = self.data_dir / "processed"
            test_path = processed_dir / "test.csv"
            val_path = processed_dir / "validation.csv"
            
            if not test_path.exists():
                logger.warning(f"Test data not found at {test_path}")
                return {}
            
            test_df = pd.read_csv(test_path)
            X_test, y_test = get_X_y(test_df)
            logger.info(f"Loaded test data: {X_test.shape}")
            
            # Load validation data for threshold tuning
            if val_path.exists():
                val_df = pd.read_csv(val_path)
                X_val, y_val = get_X_y(val_df)
                logger.info(f"Loaded validation data for threshold tuning: {X_val.shape}")
                use_threshold_tuning = True
            else:
                logger.warning("Validation data not found, using default threshold 0.5")
                
        except Exception as e:
            logger.warning(f"Could not load test data: {e}")
            return {}
        
        # Import threshold tuning function
        from src.ml.evaluator import find_optimal_threshold, evaluate_with_threshold
        
        results = {}
        results_default = {}  # Store default threshold results for comparison
        model_types = ["logistic_regression", "random_forest", "xgboost"]
        
        # Store optimal thresholds
        optimal_thresholds = {}
        
        for model_type in model_types:
            model_path = self.models_dir / f"{model_type}.pkl"
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue
            
            logger.info(f"Evaluating {model_type}...")
            
            # load_model returns tuple: (model, scaler, feature_names)
            model, scaler, feature_names = ModelTrainer.load_model(model_path)
            
            # Create evaluator with scaler
            evaluator = ModelEvaluator(
                model, 
                model_name=model_type,
                scaler=scaler,
                feature_names=feature_names
            )
            
            # First, evaluate with default threshold (0.5) for comparison
            result_default = evaluator.evaluate(X_test, y_test, dataset_name="test")
            results_default[model_type] = result_default
            
            # Find optimal threshold using validation set
            if use_threshold_tuning and X_val is not None and y_val is not None and result_default.y_proba is not None:
                # Get validation predictions
                X_val_prepared = X_val
                if scaler is not None:
                    X_val_prepared = pd.DataFrame(
                        scaler.transform(X_val),
                        columns=X_val.columns,
                        index=X_val.index
                    )
                
                y_val_proba = model.predict_proba(X_val_prepared)[:, 1]
                
                # Safely convert y_val to numpy array (handles both Series and ndarray)
                y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
                
                optimal_threshold, best_score, threshold_results = find_optimal_threshold(
                    y_val_array, 
                    y_val_proba, 
                    metric='balanced'
                )

                optimal_thresholds[model_type] = optimal_threshold
                
                # Save threshold analysis
                threshold_results.to_csv(
                    self.results_dir / f"threshold_analysis_{model_type}.csv",
                    index=False
                )
                
                # Evaluate on test set with optimal threshold
                result = evaluate_with_threshold(
                    model, X_test, y_test,
                    threshold=optimal_threshold,
                    model_name=model_type,
                    scaler=scaler
                )
                
                logger.info(
                    f"{model_type}: Default threshold=0.5 -> Accuracy={result_default.metrics['accuracy']:.4f}, "
                    f"Optimal threshold={optimal_threshold:.2f} -> Accuracy={result.metrics['accuracy']:.4f}"
                )
            else:
                result = result_default
                optimal_thresholds[model_type] = 0.5
            
            results[model_type] = result
            
            # Save confusion matrix
            evaluator.plot_confusion_matrix(
                result,
                save_path=self.figures_dir / f"cm_{model_type}.png"
            )
        
        # Generate comparison if we have results
        if results:
            # Save results with optimal thresholds
            comparison_df = compare_models(list(results.values()))
            comparison_df.to_csv(self.results_dir / "model_comparison.csv", index=False)
            
            # Also save default threshold results for comparison
            comparison_default_df = compare_models(list(results_default.values()))
            comparison_default_df.to_csv(self.results_dir / "model_comparison_default_threshold.csv", index=False)
            
            # Save optimal thresholds
            thresholds_df = pd.DataFrame([
                {"model": k, "optimal_threshold": v} 
                for k, v in optimal_thresholds.items()
            ])
            thresholds_df.to_csv(self.results_dir / "optimal_thresholds.csv", index=False)
            
            logger.info(f"Optimal thresholds: {optimal_thresholds}")
            
            plot_model_comparison(
                list(results.values()),
                save_path=self.figures_dir / "model_comparison.png"
            )
            
            plot_roc_curves_comparison(
                list(results.values()),
                save_path=self.figures_dir / "roc_comparison.png"
            )
        
        return results

    def _generate_report(self, results: Dict[str, Any]) -> None:
        """
        Generate a pipeline report.
        
        Args:
            results: Dictionary with pipeline results.
        """
        report_path = self.results_dir / "pipeline_report.txt"
        
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("SUPPLY CHAIN EXPLORER - PIPELINE REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Status: {results.get('status', 'unknown')}\n\n")
            
            f.write("STAGES:\n")
            f.write("-" * 40 + "\n")
            for stage_name, stage_info in results.get("stages", {}).items():
                f.write(f"\n{stage_name.upper()}:\n")
                if isinstance(stage_info, dict):
                    for key, value in stage_info.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {stage_info}\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def run_full_pipeline(self, retrain: bool = False) -> Dict[str, Any]:
        """
        Run the full pipeline.
        
        Args:
            retrain: If True, retrain models even if they exist.
            
        Returns:
            Dictionary with pipeline results.
        """
        results = {
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "stages": {}
        }
        
        # Stage 1: Load data
        df = self._load_data()
        results["stages"]["load"] = {"status": "success", "rows": len(df)}
        
        # Stage 2: Validate data
        validation = self._validate_data(df)
        results["stages"]["validate"] = {"status": "success", "is_valid": validation["is_valid"]}
        
        # Stage 3: Preprocess data
        df = self._preprocess_data(df)
        results["stages"]["preprocess"] = {"status": "success", "shape": str(df.shape)}
        
        # Stage 4: Engineer features
        df = self._engineer_features(df)
        results["stages"]["features"] = {"status": "success", "shape": str(df.shape)}
        
        # Stage 5: Train models (if needed)
        if retrain or not self._check_models_exist():
            self._train_models(df)
            results["stages"]["train"] = {"status": "success", "action": "trained"}
        else:
            results["stages"]["train"] = {"status": "skipped", "action": "models exist"}
        
        # Stage 6: Evaluate models
        eval_results = self._evaluate_models()
        results["stages"]["evaluate"] = {
            "status": "success",
            "models_evaluated": len(eval_results)
        }
        
        results["status"] = "success"
        results["end_time"] = datetime.now().isoformat()
        
        # Generate report
        self._generate_report(results)
        
        return results
    
    def predict(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        model_name: str = "random_forest"
    ) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            input_path: Path to input CSV file.
            output_path: Optional path to save predictions.
            model_name: Name of model to use for predictions.
            
        Returns:
            DataFrame with predictions.
        """
        logger.info(f"Making predictions on {input_path}...")
        
        # Load data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows")
        
        # Preprocess - pass df to constructor
        preprocessor = DataPreprocessor(df)
        df = preprocessor.preprocess()
        
        # Engineer features - pass df to constructor
        engineer = FeatureEngineer(df)
        df = engineer.engineer_all()
        
        # Load predictor and make predictions
        predictor = load_predictor(model_name)
        df_predictions = predictor.add_predictions_to_dataframe(df)
        
        # Save if output path provided
        if output_path:
            df_predictions.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        
        return df_predictions


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Supply Chain Explorer Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "evaluate", "predict"],
        default="full",
        help="Pipeline mode to run"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining of models"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input CSV file for prediction mode"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file for predictions"
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        help="Model to use for predictions"
    )
    
    args = parser.parse_args()
    
    pipeline = SupplyChainPipeline()
    
    if args.mode == "full":
        results = pipeline.run_full_pipeline(retrain=args.retrain)
        print(f"Pipeline completed with status: {results['status']}")
    
    elif args.mode == "train":
        df = pipeline._load_data()
        df = pipeline._preprocess_data(df)
        df = pipeline._engineer_features(df)
        pipeline._train_models(df)
        print("Training completed")
    
    elif args.mode == "evaluate":
        results = pipeline._evaluate_models()
        print(f"Evaluated {len(results)} models")
    
    elif args.mode == "predict":
        if not args.input:
            print("Error: --input required for predict mode")
            return
        predictions = pipeline.predict(args.input, args.output, args.model)
        print(f"Generated {len(predictions)} predictions")


if __name__ == "__main__":
    main()