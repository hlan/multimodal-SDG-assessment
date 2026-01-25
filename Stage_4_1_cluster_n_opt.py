# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

OUTDIR = Path("cluster_outputs")
IDX_FEAT_CSV = OUTDIR / "features_index70.csv"
SEED = 2025
np.random.seed(SEED)

idx_feat = pd.read_csv(IDX_FEAT_CSV)

CLUST_FEATS = [
    "index_70__mean_all",
    "index_70__slope_all",
    "index_70__residstd_all",
    "index_70__mean_recent",
    "index_70__slope_recent",
    "index_70__max_rise",
    "index_70__max_drop",
    "index_70__p10", "index_70__p50", "index_70__p90",
]

X = idx_feat[CLUST_FEATS].copy()
mask = ~X.isna().any(axis=1)
Xn = X[mask].reset_index(drop=True)
countries = idx_feat.loc[mask, ["countrycode","countryname"]].reset_index(drop=True)

print(f" N of countries for clustering：{len(Xn)}，discard {len(idx_feat)-len(Xn)} data missing cty")

scaler = StandardScaler()
Z = scaler.fit_transform(Xn)

# ===== try K from 3-20 =====
rows_k = []
for k in range(3, 21):  # K=3..8
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED)
    gmm.fit(Z)
    labels = gmm.predict(Z)
    sil = silhouette_score(Z, labels)
    ch  = calinski_harabasz_score(Z, labels)
    db  = davies_bouldin_score(Z, labels)
    bic = gmm.bic(Z)
    aic = gmm.aic(Z)
    rows_k.append({
        "k":k, "silhouette":sil, 
        "calinski_harabasz":ch, "davies_bouldin":db, 
        "BIC":bic, "AIC":aic
    })

k_report = pd.DataFrame(rows_k).sort_values("k")
k_report.to_csv(OUTDIR/"k_selection_metrics_20.csv", index=False)

print("output: k_selection_metrics.csv：")
print(k_report)
