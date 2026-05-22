# SL Benfica CLV Segmentation Analysis
Customer Lifetime Value (CLV) analysis and K-means segmentation of SL Benfica supporters.
Part of Master's Thesis: *"Customer Lifetime Value in Professional Football: A Data-Driven Analysis of SL Benfica"*
## Project Overview
This repository contains the complete Python implementation for analyzing supporter lifetime value and behavioral segmentation at Sport Lisboa e Benfica (SL Benfica).
**Key Features:**
- Multi-dimensional RFM (Recency, Frequency, Monetary) analysis across 6 revenue streams
- Weighted RFM scoring using Analytic Hierarchy Process (AHP) weights
- K-means clustering with validation (Silhouette Score, Elbow Method)
- Statistical testing (Kruskal-Wallis H-test)
- Comprehensive visualizations
## Repository Contents
- `analise_clv_benfica.py` - Main analysis script
- Generated outputs:
  - Cluster validation plots (Elbow, Silhouette)
  - CLV distribution visualizations
  - Cluster characteristics heatmaps
  - Segmentation summary tables (Excel)
##  Requirements
Python 3.12+
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
matplotlib >= 3.7
seaborn >= 0.12
scipy >= 1.11
openpyxl >= 3.1
