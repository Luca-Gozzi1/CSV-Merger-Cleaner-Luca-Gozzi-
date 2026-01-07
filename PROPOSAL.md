# Project Proposal

## CSV Data Explorer for Supply Chain Intelligence

**Category:** Data Science & Supply Chain Analytics  
**Author:** Luca Gozzi  
**Date:** November 2025 - January 2026



## Project Title

**CSV Data Explorer for Supply Chain: Multi-Location Inventory Consolidation, Shipment Tracking & Vendor Performance Analytics**

---

## Final Users and Ultimate Goals

### Final Users

To maximize real-world relevance, this project is designed for professionals who routinely manage and optimize complex, fragmented supply chains:

- Supply chain managers in medium to large enterprises
- Logistics analysts within multinational companies
- Data scientists/operations researchers supporting inventory and shipment optimization
- Technology consulting teams specializing in supply chain digitalization
- Vendor relationship or quality managers evaluating supplier performance

### Ultimate Goal

The project aims to deliver a practical, modular tool that automates CSV integration and analysis across fragmented supply chain systems. By solving tangible issues such as inventory imbalances, delayed shipments, and subjective vendor assessments, the tool enables managers to make faster, data-driven decisions, resulting in:

- Reduced inventory costs and shortages
- Improved shipment punctuality
- Transparent and objective vendor evaluations

### Impact

This solution directly addresses the needs of organizations struggling with data fragmentation, unifying heterogeneous data sources to provide actionable insights. The expected impact includes cost savings, operational efficiency, and enhanced strategic management. As supply chains become more complex and global, this kind of tool is increasingly vital for competitive advantage.

---

## Problem Statement and Motivation

Modern supply chains suffer from severe fragmentation across platforms (ERP, WMS, TMS, and diverse supplier portals), each exporting incompatible CSV files. According to recent research:

- **Over 70% of supply chain leaders** identify data fragmentation as their primary operational obstacle (Gartner, 2021)
- Data fragmentation contributes to **inventory imbalances costing companies up to $1 trillion globally each year** (McKinsey, 2020)
- Delayed shipment identification and subjective vendor evaluation remain persistent challenges

Spreadsheets and CSVs persist as universal integration formats, yet manual consolidation is error-prone and time-consuming. This project applies advanced data science to automate these processes and unlock hidden value from disconnected supply chain data.

---

## Planned Approach and Technologies

The project follows a modular architecture with machine learning capabilities:

### Core Analytics Modules

1. **Module 1: Multi-Location Inventory Consolidator**
   - Standardizes, loads, and merges CSVs from multiple sites
   - Aggregates inventory by SKU
   - Performs ABC (Pareto) analysis
   - Creates actionable recommendations for optimal stock allocation


2. **Module 2: Shipment Tracking Intelligence System**
   - Integrates tracking CSVs
   - Calculates On-Time Delivery Rate (OTDR)
   - Averages supplier delays
   - Highlights severe delays and issues prioritized alerts to managers


3. **Module 3: Vendor Performance Analyzer**
   - Assigns composite performance scores using weighted KPIs (delivery, quality, cost, responsiveness)
   - Classifies vendors by percentiles
   - Suggests improvement strategies

### Machine Learning Extension

- **Shipment Delay Prediction Model** using classification algorithms
- Feature engineering from historical shipment data
- Model evaluation with threshold optimization

### Technology Stack

- **Python** (pandas, numpy, matplotlib/seaborn)
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **OOP & modular programming**
- **Automated testing** (pytest), version control (git), best coding practices
- **Public datasets** (Kaggle, Data.gov)

---

## Expected Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Heterogeneous CSV formats | Automated delimiter, encoding detection, and standardized column mapping |
| Data quality issues | Robust validation and cleaning pipeline (handling missing data, duplicates, inconsistent dates, outliers) |
| Large file processing | Efficient chunked loading, optimized data types |
| Algorithm complexity | Well-documented implementations with unit tests |
| Data leakage in ML | Explicit feature auditing and temporal validation |


## Success Criteria

- [x] Merge/integrate at least three location-based CSVs with accurate aggregation
- [x] Calculate OTDR with reliable accuracy, detect and report delays by supplier
- [x] Generate at least five actionable recommendations for inventory and logistics
- [x] Assign vendor performance tiers matching expected statistical distribution
- [x] **80%+ test coverage**, proven efficient processing, and professional documentation
- [x] **Machine learning model** achieving meaningful prediction accuracy

---

## Final Implementation Summary

The project evolved to focus on **Shipment Delay Prediction** as the primary ML component, achieving:

| Metric | Target | Achieved |
|--------|--------|----------|
| Test Coverage | 80%+ | **80%** ✓ |
| Prediction Accuracy | > Baseline | **69.2%** (14.4% above 54.8% baseline) ✓ |
| Precision | High | **84%** ✓ |
| Lines of Code | 1000+ | **6,451** ✓ |
| Automated Tests | Comprehensive | **255 tests** ✓ |


## Repository Structure

```
supply-chain-ml-predictor/
├── src/
│   ├── data/           # Data loading, validation, preprocessing
│   ├── features/       # Feature engineering and selection
│   └── ml/             # Model training, evaluation, prediction
├── tests/              # Comprehensive test suite
├── data/               # Raw and processed data
├── models/             # Trained model artifacts
├── results/            # Output files and visualizations
└── docs/               # Additional documentation
```

## Contact

**Email:** luca.gozzi@unil.ch  
**License:** MIT

---

## References

1. Gartner (2021). "Supply Chain Fragmentation Report": Over 70% of leaders identify data silos as a top challenge.

2. McKinsey (2020). "Global Supply Chain Survey": Data fragmentation results in annual excess costs estimated at $1 trillion.

3. Kaggle DataCo Supply Chain Dataset: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

4. Data.gov: https://data.gov/
