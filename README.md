# Supply Chain ML Predictor

Machine Learning for Supply Chain Intelligence: Predicting Delivery Delays Using Classification Models

**Author:** Luca Gozzi  
**Course:** Data Science and Advanced Programming  
**Date:** November 2025 – January 2026  
**License:** MIT  


## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Methodology](#methodology)
8. [Results](#results)
9. [Testing](#testing)
10. [References](#references)

---

## Overview

This project implements a machine learning system for predicting delivery delays in supply chain operations. By analyzing historical shipment data, the system identifies orders at risk of late delivery, enabling proactive intervention before problems materialize.

The solution addresses a critical challenge in modern supply chains: data fragmentation. According to Gartner (2021), over 70% of supply chain leaders identify data silos as their primary operational obstacle, contributing to inventory imbalances estimated at $1 trillion globally per year (McKinsey, 2020).

### Key Metrics

| Metric | Value |
|--------|-------|
| Prediction Accuracy | 69.2% |
| Precision | 84% |
| Test Coverage | 80% |
| Automated Tests | 255 |
| Lines of Code | 6,451 |
| Dataset Size | 180,519 orders |

---

## Problem Statement

Traditional supply chain delay management is inherently reactive: managers learn about delays after they occur, often through customer complaints. This reactive stance limits response options and frequently results in expensive emergency measures.

This project transforms delay management from reactive to proactive by:

1. Predicting delays before they occur using machine learning
2. Identifying high-risk shipments for prioritized monitoring
3. Enabling early intervention through customer notification or shipping upgrades


## Project Structure

```
supply-chain-ml-predictor/
│
├── README.md                   # Project documentation
├── PROPOSAL.md                 # Original project proposal
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration
├── LICENSE                     # MIT License
│
├── src/                        # Source code
│   ├── config.py               # Configuration and hyperparameters
│   ├── main.py                 # Pipeline orchestrator
│   │
│   ├── data/                   # Data processing modules
│   │   ├── loader.py           # CSV loading with encoding detection
│   │   ├── validator.py        # Data validation and quality checks
│   │   └── preprocessor.py     # Cleaning and transformation
│   │
│   ├── features/               # Feature engineering
│   │   ├── engineer.py         # Feature creation
│   │   └── selector.py         # Feature importance ranking
│   │
│   └── ml/                     # Machine learning
│       ├── splitter.py         # Train/validation/test splitting
│       ├── trainer.py          # Model training
│       ├── evaluator.py        # Performance metrics and visualization
│       ├── predictor.py        # Inference on new data
│       └── ensemble.py         # Ensemble methods
│
├── tests/                      # Test suite (255 tests)
│   ├── conftest.py             # Pytest fixtures
│   ├── test_loader.py
│   ├── test_validator.py
│   ├── test_preprocessor.py
│   ├── test_features.py
│   ├── test_selector.py
│   ├── test_splitter.py
│   ├── test_trainer.py
│   ├── test_evaluator.py
│   ├── test_predictor.py
│   ├── test_ensemble.py
│   └── test_main.py
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Processed data files
│
├── models/                     # Trained model artifacts
│
└── results/                    # Output files and visualizations
    └── figures/                # Generated plots
```

---

## Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package installer)
- Git

### Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/supply-chain-ml-predictor.git
cd supply-chain-ml-predictor
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place `DataCoSupplyChainDataset.csv` in the `data/raw/` directory:
   - Source: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

5. Verify installation:

```bash
pytest tests/ -v
```

---

## Usage

### Running the Full Pipeline

Execute the complete ML pipeline (load, validate, preprocess, engineer features, train, evaluate):

```bash
python -m src.main --mode full
```

### Training Models Only

```bash
python -m src.main --mode train
```

### Evaluating Existing Models

```bash
python -m src.main --mode evaluate
```

### Force Retraining

```bash
python -m src.main --mode full --retrain
```

### Expected Output

```
2025-01-05 00:30:15 - INFO - Pipeline initialized
2025-01-05 00:30:15 - INFO - Loading data...
2025-01-05 00:30:18 - INFO - Successfully loaded 180,519 rows
2025-01-05 00:30:18 - INFO - Memory reduced from 293.40 MB to 24.78 MB (91.6% reduction)
2025-01-05 00:30:20 - INFO - Validation complete. Valid: True, Errors: 0, Warnings: 0
2025-01-05 00:30:25 - INFO - Preprocessing complete. 180,519 rows remaining
2025-01-05 00:30:30 - INFO - Feature engineering complete. 64 features total
2025-01-05 00:31:15 - INFO - Training complete for 3 models
2025-01-05 00:31:30 - INFO - Evaluation complete
```

---

## Dataset

### DataCo Supply Chain Dataset

| Attribute | Value |
|-----------|-------|
| Source | Kaggle |
| Records | 180,519 orders |
| Time Period | 2015–2018 |
| Features | 53 columns |

### Key Variables

| Variable | Description |
|----------|-------------|
| Order Id | Unique order identifier |
| Shipping Mode | Shipping method (Standard, First Class, Same Day, Second Class) |
| Days for shipping (scheduled) | Promised delivery time |
| Late_delivery_risk | Target variable (1 = Late, 0 = On-time) |
| Market | Geographic market region |
| Order Item Quantity | Number of items ordered |

### Class Distribution

| Class | Proportion |
|-------|------------|
| Late deliveries (1) | 54.8% |
| On-time deliveries (0) | 45.2% |

---

## Methodology

### Pipeline Architecture

The system follows a sequential pipeline architecture:

1. **Data Loading**: CSV ingestion with automatic encoding detection and memory optimization (91.6% reduction)
2. **Validation**: Schema validation, type checking, logical consistency verification
3. **Preprocessing**: Duplicate removal, missing value imputation (median/mode), outlier winsorization
4. **Feature Engineering**: 64 features created from 35 original columns
5. **Data Splitting**: Stratified 70/15/15 train/validation/test split
6. **Model Training**: Three classification algorithms
7. **Evaluation**: Metrics computation and threshold optimization

### Feature Engineering

Features are organized into categories:

- **Temporal**: day of week, month, quarter, weekend indicator, holiday season
- **Shipping**: mode risk scores (domain-informed encoding)
- **Geographic**: market-based risk assessment, international shipping indicator
- **Interaction**: shipping × market interaction, order complexity

### Models

| Model | Configuration |
|-------|---------------|
| Logistic Regression | L2 regularization (C=0.3), StandardScaler |
| Random Forest | 300 estimators, max depth 25, parallel training |
| XGBoost | 300 estimators, learning rate 0.03, max depth 10 |

### Data Leakage Prevention

During development, initial models achieved 95% accuracy. Investigation revealed several features contained post-delivery information:

- Days for shipping (real)
- shipping_time_ratio
- shipping_lead_time_variance
- order_processing_time

These features were removed, reducing accuracy to 69.2%—representing honest predictive capability using only information available at order time.

---

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 69.21% | 84.26% | 53.94% | 65.78% | 0.732 |
| Random Forest | 69.17% | 83.99% | 54.13% | 65.83% | 0.735 |
| XGBoost | 68.97% | 84.09% | 53.56% | 65.42% | 0.731 |

### Key Findings

1. **Model convergence**: All three models achieve similar performance, suggesting engineered features effectively capture the available predictive signal.

2. **High precision**: When the model predicts a late delivery, it is correct 84% of the time.

3. **Moderate recall**: Approximately 54% of actual late deliveries are identified.

4. **Feature importance**: The engineered feature `shipping_mode_risk` is the strongest predictor.

### Business Impact

For a company shipping 10,000 orders monthly:
- The model flags approximately 3,500 orders as high-risk
- Of these, approximately 3,000 will actually be late (84% precision)
- This enables proactive intervention for thousands of at-risk shipments

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_loader.py -v
```

### Test Coverage Summary

| Module | Coverage |
|--------|----------|
| src/config.py | 100% |
| src/data/loader.py | 91% |
| src/data/preprocessor.py | 93% |
| src/data/validator.py | 91% |
| src/features/engineer.py | 92% |
| src/features/selector.py | 99% |
| src/ml/trainer.py | 91% |
| src/ml/evaluator.py | 74% |
| src/ml/splitter.py | 95% |
| src/ml/predictor.py | 84% |
| **Total** | **80%** |

---

## Code Quality

This project adheres to:

- PEP 8 style guidelines
- Google-style docstrings
- Type hints for all function signatures

Linting commands:

```bash
flake8 src/ --max-line-length=120
black src/ tests/
mypy src/
```

---

## References

1. Gartner (2021). Supply Chain Fragmentation Report.

2. McKinsey & Company (2020). Global Supply Chain Survey.

3. DataCo Supply Chain Dataset. Kaggle. Available at: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

---

## Contact

Luca Gozzi  
Email: luca.gozzi@unil.ch
