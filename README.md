# Code and Data for:

## Multimodal learning reshapes the assessment of global Sustainable Development Goals

This repository contains the code and intermediate outputs used in the study titled:

**Multimodal learning reshapes the assessment of global Sustainable Development Goals**,  
submitted to **Proceedings of the National Academy of Sciences (PNAS)**.

---

## 📁 Repository Structure

### 1. Data Sources 
All analyses rely on publicly available datasets obtained from the following sources:

- **Nighttime Satellite data (imagery)**
  - Harmonization of DMSP and VIIRS nighttime light data from 1992-2024 at the global scale:  
    Li, Xuecao; Zhou, Yuyu; zhao, Min; Zhao, Xia (2020). Harmonization of DMSP and VIIRS nighttime light data from 1992-2024 at the global scale. figshare. Dataset. https://doi.org/10.6084/m9.figshare.9828827.v10

- **Socioeconomic indicators (tablular)**
  - World Development Indicators (WDI):  
    Li, W. (2025). Data for Diagnosing Syndromes of Biosphere-Atmosphere-Socioeconomic Change [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14876723

- **Country-level descriptions (textual)**
  - CIA World Factbook:  
    [https://www.cia.gov/the-world-factbook/](https://www.cia.gov/the-world-factbook/countries/)
    
- **SDG Index (SDGi)**
  - Sustainable Development Report 2025 (with indicators):  
    https://sdg-transformation-center-sdsn.hub.arcgis.com/datasets/sdsn::sustainable-development-report-2025-with-indicators/about

---

### 2. Code 

This stage contains scripts for preparing and harmonizing the multimodal inputs used throughout the analysis.
- **Stage 1 & 2: Data Integration, Embedding, Unsupervised Gloabl Sustainability Variation Detection**  
  This stage contains scripts for preparing and harmonizing the multi-modal socioeconomic, textual, and remotely sensed imagery data inputs used throughout the analysis. Integration and embedding of the data and construction of the Global Multimodal Sustainability Index (GMSI)
  - `Stage_1_and_2_multimodal_data_integration_and_embedding.py`

- **Stage 3: Supervised Model Validation and Performance Benchmarking**  
  This stage implements the supervised validation, performance benchmarking of GMSI and optimized model selection
  - `Stage_3_supervised_validation.py`
  
- **Stage 4: Sustainability Variation Clustering, Interlinking with the SDGs and Labeling the Interlinkages**  
  This stage conducts scripts for 0) feature_extraction for clustering; 1) number of clusters optimization; 2) GMM model fitting, 3) interpretive labeling and multi-level SDG performance grading. 
  - `Stage_4_0_feature_extraction.py`
  - `Stage_4_1_cluster_n_opt.py`
  - `Stage_4_2_GMM_clustering.py`
  - `Stage_4_3_clustering_stats_and_results.py`

---

### 3. Outputs

This directory includes intermediate and final outputs retained for completeness but not presented in the Supplementary Document.  
- This file contains multiple tabs about cluster-level SDG performance statistics, including standardized scores, descriptive summaries, and inferential analyses (ANOVA and post hoc Tukey HSD) used to support comparative interpretation and SDG level assignment.
  - `cluster_sdg_results.xlsx`

- This table reports country-level cluster assignments, membership probabilities across clusters, and distance-based diagnostics (Euclidean and Mahalanobis distances) used to quantify assignment confidence and cluster proximity.
  - `cluster_membership_with_distances_k7.csv`

---


### 4. Computing Environment

This study was executed in a GPU-accelerated computing environment designed to support large-scale multimodal model training.

- **Hardware**
  - NVIDIA RTX PRO 6000 Blackwell 96 GB GDDR7 GPU memory with ECC

- **Key Software Version**
  - Python: 3.12
  - AutoGluon: 1.4.0
  - PyTorch: 2.7.1
  - CUDA: 12.8

---


## Citation

If you reference this work, please cite the associated manuscript:

> Multimodal learning reshapes the assessment of global Sustainable Development Goals.

Full citation details will be added upon publication.
