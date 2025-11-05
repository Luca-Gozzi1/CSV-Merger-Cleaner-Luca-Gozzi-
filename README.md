# CSV Data Explorer for Supply Chain Intelligence
**Category:** Data Science & Supply Chain Analytics  
**Author:** Luca Gozzi  
**Date:** November 5, 2025  

## Project Title
**CSV Data Explorer for Supply Chain: Multi location Inventory Consolidation, Shipment Tracking & Vendor Performance Analytics**

## Final Users and Ultimate Goals

**Final Users:**
To maximize real world relevance, this project is designed for professionals who routinely manage and optimize complex, fragmented supply chains. The following user groups represent those who will benefit most from automated CSV integration and actionable analytics:
- Supply chain managers in medium to large enterprises  
- Logistics analysts within multinational companies  
- Data scientists/operations researchers supporting inventory and shipment optimization  
- Technology consulting teams specializing in supply chain digitalization  
- Vendor relationship or quality managers evaluating supplier performance  

**Ultimate Goal:** The project aims to deliver a practical, modular tool that automates CSV integration and analysis across fragmented supply chain systems. By solving tangible issues such as inventory imbalances, delayed shipments, and subjective vendor assessments the tool enables managers to make faster, data-driven decisions, resulting in:
- Reduced inventory costs and shortages
- Improved shipment punctuality
- Transparent and objective vendor evaluations

**Impact:** This solution directly addresses the needs of organizations struggling with data fragmentation, unifying heterogeneous data sources to provide actionable insights. The expected impact includes cost savings, operational efficiency, and enhanced strategic management. As supply chains become more complex and global, this kind of tool is increasingly vital for competitive advantage.

## Problem Statement and Motivation

Modern supply chains suffer from severe fragmentation across platforms (ERP, WMS, TMS, and diverse supplier portals), each exporting incompatible CSV files. According to recent research, **over 70% of supply chain leaders** identify data fragmentation as their primary operational obstacle, contributing to inventory imbalances *costing companies up to $1 trillion globally each year* delayed shipment identification, and subjective vendor evaluation ([Gartner, 2021][1]; [McKinsey, 2020][2]). Spreadsheets and CSVs persist as universal integration formats, yet manual consolidation is error-prone and time-consuming. This project applies advanced data science to automate these processes and unlock hidden value from disconnected supply chain data.

## Planned Approach and Technologies

The project follows a modular architecture:
- **Module 1: Multi-Location Inventory Consolidator**  
  Standardizes, loads, and merges CSVs from multiple sites; aggregates inventory by SKU; performs ABC (Pareto) analysis; creates actionable recommendations for optimal stock allocation.
- **Module 2: Shipment Tracking Intelligence System**  
  Integrates tracking CSVs, calculates On-Time Delivery Rate (OTDR), averages supplier delays, highlights severe delays, and issues prioritized alerts to managers.
- **Module 3: Vendor Performance Analyzer**  
  Assigns composite performance scores using weighted KPIs (delivery, quality, cost, responsiveness); classifies vendors by percentiles; suggests improvement strategies.

**Technology Stack:**  

- Python (pandas, numpy, matplotlib or seaborn)  
- OOP & modular programming  
- Automated testing (`pytest`), version control (`git`), best coding practices  
- Public datasets (e.g., [Kaggle][3], [Data.gov][4]); optional extensions with `scikit-learn` for predictive analytics  

## Expected Challenges and Solutions

- **Heterogeneous CSV formats:**  
  *Solution*: Automated delimiter, encoding detection, and standardized column mapping
- **Data quality issues:**  
  *Solution*: Robust validation and cleaning pipeline (handling missing data, duplicates, inconsistent dates, outliers)
- **Large file processing:**  
  *Solution*: Efficient chunked loading, optimized data types
- **Algorithm complexity (Pareto, statistics):**

## Success Criteria

- Merge/integrate at least three location based CSVs with accurate aggregation
- Calculate OTDR with reliable accuracy, detect and report delays by supplier
- Generate at least five actionable recommendations for inventory and logistics
- Assign vendor performance tiers matching expected statistical distribution
- 80%+ test coverage, proven efficient processing, and professional documentation

## Stretch Goals

- Time series forecasting & anomaly detection (exponential smoothing, Z score outlier detection)
- Supplier risk prediction (Random Forest or similar models)
- Streamlit dashboard for non technical end users
- API and Docker containerization for easy integration

## Repository Structure
**Contact:** [gozziluca02@gmail.com](mailto:gozziluca02@gmail.com)  
**License:** MIT  

### References
[1]: Gartner (2021). "Supply Chain Fragmentation Report": Over 70% of leaders identify data silos as a top challenge.  
[2]: McKinsey (2020). "Global Supply Chain Survey": Data fragmentation results in annual excess costs estimated at $1 trillion.  
[3]: https://www.kaggle.com/  
[4]: https://data.gov/
