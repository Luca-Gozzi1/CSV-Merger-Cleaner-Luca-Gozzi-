# CSV Data Explorer for Supply Chain Intelligence
**Category:** Data Science & Supply Chain Analytics
**Author:** Luca Gozzi
**Date:** November 3, 2025
---
## Project Title
CSV Data Explorer for Supply Chain: Multi-Location Inventory Consolidation, Shipment
Tracking & Vendor Performance Analytics
---
## Problem Statement and Motivation
Modern supply chains suQer from severe fragmentation across platforms—ERP, WMS, TMS,
and diverse supplier portals—each exporting incompatible CSV files. According to recent
research, 70% of supply chain leaders identify data fragmentation as their main operational
obstacle, leading to inventory imbalances (costing $1 trillion annually), delayed shipment
detection, and subjective vendor evaluation. CSVs persist as the universal integration format,
yet manual consolidation is error-prone and time-consuming. This project aims to apply
practical data science skills to build an automated tool that addresses critical operation pain
points through advanced CSV analytics.
---
## Planned Approach and Technologies
The project follows a modular architecture:
- **Module 1: Multi-Location Inventory Consolidator**
Standardizes, loads, and merges CSVs from multiple sites, aggregates inventories by SKU,
performs ABC Analysis (Pareto classification), and generates actionable recommendations
for optimal stock distribution.
- **Module 2: Shipment Tracking Intelligence System**
Integrates tracking CSVs, calculates On-Time Delivery Rate (OTDR), average delays per
supplier, highlights severe delays, and delivers prioritized alerts for managers.
- **Module 3: Vendor Performance Analyzer**
Assigns composite performance scores using weighted KPIs (delivery, quality, cost,
responsiveness), classifies vendors by percentiles, and suggests improvements.
**Technology Stack:**
- Python with pandas, numpy, matplotlib/seaborn
- OOP and modular programming
- Automated testing (pytest), version control (git), best coding practices
- Public datasets (Kaggle, Data.gov), potential extensions with scikit-learn for prediction
---
## Expected Challenges and Solutions
- **Heterogeneous CSV Formats:**
*Solution:* Automated delimiter and encoding detection, standardized column mapping.
- **Data Quality Issues:**
*Solution:* Robust validation and cleaning pipeline (handle missing data, duplicates, dates,
outliers).
- **Large File Processing:**
*Solution:* EQicient chunked loading and optimization of data types.
- **Algorithm Complexity (Pareto, statistics):**
*Solution:* Stepwise implementation, leveraging vectorized operations and code reviews.
---
## Success Criteria
- Consolidates at least three location CSVs with correct aggregation.
- Calculates OTDR with reliable accuracy.
- Generates at least five actionable recommendations.
- Assigns vendor tiers matching expected statistical distribution.
- Achieves 80%+ test coverage, eQicient processing, and professional documentation.
---
## Stretch Goals
- Time series forecasting and anomaly detection (e.g., exponential smoothing, Z-scores)
- Supplier risk prediction via machine learning (Random Forest)
- Interactive Streamlit dashboard for non-technical users
- API for integration and Docker containerization
---
## Repository Structure
---
**Contact:** gozziluca02@gmail.com
**License:** MIT