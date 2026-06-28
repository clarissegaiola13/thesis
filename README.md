# SL Benfica CLV Segmentation Analysis

Replication materials for the Master's thesis:

**Customer Lifetime Value in Professional Football: A Data-Driven Analysis of SL Benfica**  
NOVA IMS — Data-Driven Marketing (Data Science for Marketing), 2026

## Overview

This repository provides the Python pipeline used to:

- preprocess anonymized supporter CRM data;
- compute weighted RFM scores across six revenue categories;
- segment supporters with K-means clustering (*k* = 8);
- validate clusters (Silhouette Score, Elbow Method, Kruskal-Wallis H-test);
- generate thesis figures and summary tables.

## Data availability (GDPR)

The supporter-level CRM dataset is **not** included in this repository (GDPR and club confidentiality).

The script expects a file named `dados_benfica.csv` in the project root:

- format: CSV, semicolon-separated (`;`), UTF-8;
- sample used in the thesis: 20,000 anonymized supporter records, 71 variables;
- minimum required fields: preprocessed RFM inputs (`R_*`, `F_*`, `M_*` for six revenue categories), as described in Thesis Chapter 3.

External users cannot reproduce the exact Benfica results without access to equivalent club data; this repository documents the analytical workflow used in the study.

## Requirements

- Python 3.12+

```bash
pip install -r requirements.txt
```

## Usage

```bash
python analise_clv_benfica.py
```

The script expects `dados_benfica.csv` in the same directory.

## Outputs

Running the script generates:

| File | Description |
|------|-------------|
| `01_elbow_silhouette.png` | Elbow and Silhouette validation plots |
| `02_cluster_summary.xlsx` | Cluster size and CLV summary |
| `03_boxplot_clv_cluster.png` | CLV distribution by cluster |
| `04_barplot_clv_medio.png` | Mean CLV by cluster |
| `05_heatmap_clusters.png` | Cluster characteristics heatmap |
| `06_situacao_por_cluster.png` | Membership status by cluster |
| `07_scatter_rfm_clv.png` | RFM vs CLV scatter plot |
| `08_pie_clusters.png` | Cluster size distribution |
| `09_dataset_com_clusters.xlsx` | Full dataset with cluster labels |
| `10_cluster_detailed_summary.xlsx` | Detailed cluster statistics |

## Main script

- `analise_clv_benfica.py` — complete analytical pipeline

## Citation

If you use this code, please cite the thesis and reference this repository:

```
Gaiola, C. P. H. F. (2026). Customer Lifetime Value in Professional Football:
A Data-Driven Analysis of SL Benfica. Master's thesis, NOVA IMS.
https://github.com/clarissegaiola13/thesis
```
