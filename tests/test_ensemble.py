"""
Quick test script for ensemble models.
Run: python test_ensemble.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.ensemble import EnsembleTrainer, evaluate_ensemble
from src.ml.splitter import get_X_y

def main():
    print("=" * 70)
    print("ENSEMBLE MODEL TESTING")
    print("=" * 70)
    
    # Load data
    data_dir = Path("data/processed")
    
    print("\n1. Loading data...")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    
    # Initialize trainer
    print("\n2. Initializing EnsembleTrainer...")
    trainer = EnsembleTrainer(
        X_train, y_train,
        X_val, y_val,
        scale_features=True
    )
    
    results = {}
    
    # Test 1: Optimized LightGBM
    print("\n" + "-" * 70)
    print("3. Training Optimized LightGBM...")
    print("-" * 70)
    try:
        lgb_result = trainer.train_lightgbm_optimized()
        lgb_metrics = evaluate_ensemble(
            lgb_result.model, X_test, y_test,
            scaler=lgb_result.scaler,
            model_name="LightGBM"
        )
        results['LightGBM'] = lgb_metrics
        print(f"   ✓ LightGBM Accuracy: {lgb_metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"   ✗ LightGBM failed: {e}")
    
    # Test 2: Voting Ensemble
    print("\n" + "-" * 70)
    print("4. Training Voting Ensemble...")
    print("-" * 70)
    try:
        voting_result = trainer.train_voting(voting='soft')
        voting_metrics = evaluate_ensemble(
            voting_result.model, X_test, y_test,
            scaler=voting_result.scaler,
            model_name="Voting"
        )
        results['Voting'] = voting_metrics
        print(f"   ✓ Voting Accuracy: {voting_metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"   ✗ Voting failed: {e}")
    
    # Test 3: Stacking Ensemble
    print("\n" + "-" * 70)
    print("5. Training Stacking Ensemble...")
    print("-" * 70)
    try:
        stacking_result = trainer.train_stacking(cv=5)
        stacking_metrics = evaluate_ensemble(
            stacking_result.model, X_test, y_test,
            scaler=stacking_result.scaler,
            model_name="Stacking"
        )
        results['Stacking'] = stacking_metrics
        print(f"   ✓ Stacking Accuracy: {stacking_metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"   ✗ Stacking failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    
    for name, metrics in results.items():
        print(
            f"{name:<20} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1_score']:>10.4f}"
        )
    
    # Best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n★ Best Model: {best_model[0]} with {best_model[1]['accuracy']:.4f} accuracy")
        
        if best_model[1]['accuracy'] >= 0.70:
            print("✓ TARGET ACHIEVED: Accuracy >= 70%")
        else:
            gap = 0.70 - best_model[1]['accuracy']
            print(f"✗ Gap to 70%: {gap:.4f} ({gap*100:.2f}%)")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv("results/ensemble_comparison.csv")
    print(f"\nResults saved to results/ensemble_comparison.csv")


if __name__ == "__main__":
    main()