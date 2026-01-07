"""
Script to evaluate all trained models on validation and test sets.

This script:
1. Loads trained models from models/ directory
2. Evaluates each model on validation and test sets
3. Generates visualizations (confusion matrices, ROC curves, etc.)
4. Creates a model comparison report
5. Saves all results to results/ directory

Author: Luca Gozzi
Date: November 2025
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from src.ml.splitter import DataSplitter, get_X_y
from src.ml.trainer import ModelTrainer
from src.ml.evaluator import (
    ModelEvaluator,
    compare_models,
    plot_model_comparison,
    plot_roc_curves_comparison,
)


def main():
    print("=" * 60)
    print("Model Evaluation Pipeline")
    print("=" * 60)
    
    # Setup paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data" / "processed"
    models_dir = project_dir / "models"
    results_dir = project_dir / "results"
    figures_dir = results_dir / "figures"
    
    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        splits = DataSplitter.load_splits(data_dir)
        print(f"   Train: {len(splits.train):,} rows")
        print(f"   Validation: {len(splits.validation):,} rows")
        print(f"   Test: {len(splits.test):,} rows")
    except FileNotFoundError:
        print("   ERROR: Split data not found. Run test_splitting.py first.")
        return
    
    # Step 2: Extract X and y
    print("\n2. Extracting features and target...")
    X_val, y_val = get_X_y(splits.validation)
    X_test, y_test = get_X_y(splits.test)
    print(f"   Validation: {X_val.shape[0]:,} samples, {X_val.shape[1]} features")
    print(f"   Test: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    
    # Step 3: Load trained models
    print("\n3. Loading trained models...")
    
    models_to_evaluate = [
        ("Logistic Regression", models_dir / "logistic_regression.pkl"),
        ("Random Forest", models_dir / "random_forest.pkl"),
        ("XGBoost", models_dir / "xgboost.pkl"),
    ]
    
    loaded_models = []
    for name, path in models_to_evaluate:
        if path.exists():
            model, scaler, feature_names = ModelTrainer.load_model(path)
            loaded_models.append((name, model, scaler, feature_names))
            print(f"   ✓ Loaded {name}")
        else:
            print(f"   ✗ {name} not found at {path}")
    
    if not loaded_models:
        print("   ERROR: No models found. Run test_training.py first.")
        return
    
    # Step 4: Evaluate on validation set
    print("\n4. Evaluating models on VALIDATION set...")
    print("-" * 60)
    
    val_results = []
    for name, model, scaler, feature_names in loaded_models:
        evaluator = ModelEvaluator(model, name, scaler, feature_names)
        result = evaluator.evaluate(X_val, y_val, "validation")
        val_results.append(result)
        
        print(result.summary())
        
        # Save individual plots
        evaluator.plot_confusion_matrix(
            result,
            save_path=figures_dir / f"cm_{name.lower().replace(' ', '_')}_val.png"
        )
        evaluator.plot_roc_curve(
            result,
            save_path=figures_dir / f"roc_{name.lower().replace(' ', '_')}_val.png"
        )
        evaluator.plot_feature_importance(
            top_n=15,
            save_path=figures_dir / f"importance_{name.lower().replace(' ', '_')}.png"
        )
        plt.close('all')  # Free memory
    
    # Step 5: Evaluate on test set
    print("\n5. Evaluating models on TEST set...")
    print("-" * 60)
    
    test_results = []
    for name, model, scaler, feature_names in loaded_models:
        evaluator = ModelEvaluator(model, name, scaler, feature_names)
        result = evaluator.evaluate(X_test, y_test, "test")
        test_results.append(result)
        
        print(result.summary())
        
        # Save individual plots
        evaluator.plot_confusion_matrix(
            result,
            save_path=figures_dir / f"cm_{name.lower().replace(' ', '_')}_test.png"
        )
        evaluator.plot_roc_curve(
            result,
            save_path=figures_dir / f"roc_{name.lower().replace(' ', '_')}_test.png"
        )
        plt.close('all')
    
    # Step 6: Create comparison reports
    print("\n6. Creating comparison reports...")
    
    # Validation comparison
    val_comparison = compare_models(
        val_results,
        save_path=results_dir / "model_comparison_validation.csv"
    )
    print("\n   VALIDATION SET COMPARISON:")
    print(val_comparison.to_string(index=False))
    
    # Test comparison
    test_comparison = compare_models(
        test_results,
        save_path=results_dir / "model_comparison_test.csv"
    )
    print("\n   TEST SET COMPARISON:")
    print(test_comparison.to_string(index=False))
    
    # Step 7: Create comparison visualizations
    print("\n7. Creating comparison visualizations...")
    
    # Model comparison bar chart - validation
    plot_model_comparison(
        val_results,
        save_path=figures_dir / "model_comparison_val.png"
    )
    
    # Model comparison bar chart - test
    plot_model_comparison(
        test_results,
        save_path=figures_dir / "model_comparison_test.png"
    )
    
    # ROC curves comparison - validation
    plot_roc_curves_comparison(
        val_results,
        save_path=figures_dir / "roc_comparison_val.png"
    )
    
    # ROC curves comparison - test
    plot_roc_curves_comparison(
        test_results,
        save_path=figures_dir / "roc_comparison_test.png"
    )
    
    plt.close('all')
    
    # Step 8: Summary
    print("\n8. Summary")
    print("=" * 60)
    
    # Find best model by F1 score on test set
    best_result = max(test_results, key=lambda r: r.metrics.get("f1_score", 0))
    
    print(f"\n   Best model (by F1): {best_result.model_name}")
    print(f"   Test Accuracy: {best_result.metrics['accuracy']:.4f}")
    print(f"   Test Precision: {best_result.metrics['precision']:.4f}")
    print(f"   Test Recall: {best_result.metrics['recall']:.4f}")
    print(f"   Test F1 Score: {best_result.metrics['f1_score']:.4f}")
    print(f"   Test ROC-AUC: {best_result.metrics.get('roc_auc', 'N/A')}")
    
    # Note about data leakage
    print("\n   ⚠️  NOTE ON HIGH ACCURACY:")
    print("   The high accuracy (~97%) suggests data leakage from features")
    print("   that contain information only available after delivery")
    print("   (e.g., 'Days for shipping (real)', 'shipping_time_ratio').")
    print("   This is a known limitation of the dataset and should be")
    print("   discussed in the technical report.")
    
    # Files saved
    print("\n   Files saved:")
    print(f"   - {results_dir / 'model_comparison_validation.csv'}")
    print(f"   - {results_dir / 'model_comparison_test.csv'}")
    print(f"   - {figures_dir}/ (multiple PNG figures)")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()