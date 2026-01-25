# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib  

OUTDIR = Path("cluster_outputs")
IDX_FEAT_CSV = OUTDIR / "features_index70.csv"  
SEED = 2025

K = 7 # selected optimized cluster number

COVARIANCE_TYPE = "full"  

def build_inv_covs(gmm: GaussianMixture):
    Kc = gmm.n_components
    D = gmm.means_.shape[1]
    inv_covs = []
    ctype = gmm.covariance_type

    if ctype == "full":
        for k in range(Kc):
            inv_covs.append(np.linalg.inv(gmm.covariances_[k]))
    elif ctype == "diag":
        for k in range(Kc):
            inv_covs.append(np.diag(1.0 / gmm.covariances_[k]))
    elif ctype == "tied":
        inv_tied = np.linalg.inv(gmm.covariances_)
        inv_covs = [inv_tied for _ in range(Kc)]
    elif ctype == "spherical":
        for k in range(Kc):
            inv_covs.append(np.eye(D) * (1.0 / gmm.covariances_[k]))
    else:
        raise ValueError(f"Unsupported covariance_type: {ctype}")
    return inv_covs


def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)

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
    meta = idx_feat.loc[mask, ["countrycode", "countryname"]].reset_index(drop=True)

    dropped = len(idx_feat) - len(Xn)
    print(f"cty for clustering：{len(Xn)}（discard {dropped} data missing cty）")

    scaler = StandardScaler()
    Z = scaler.fit_transform(Xn)  # (N, D)

    # GMM 
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=COVARIANCE_TYPE,
        random_state=SEED
    )
    gmm.fit(Z)
    labels = gmm.predict(Z)        # (N,)
    proba = gmm.predict_proba(Z)   # (N, K)

    # proba
    res = meta.copy()
    res["cluster"] = labels
    for i in range(K):
        res[f"proba_{i}"] = proba[:, i]


    inv_covs = build_inv_covs(gmm)
    means = gmm.means_                 # (K, D)
    assigned_means = means[labels]     # (N, D)
    diff = Z - assigned_means          # (N, D)

    # euclid_dist
    euclid_dist = np.linalg.norm(diff, axis=1)

    # maha_dist
    maha_dist = np.empty(len(Z), dtype=float)
    for i in range(len(Z)):
        k = labels[i]
        v = diff[i]       # (D,)
        invC = inv_covs[k]  # (D, D)
        maha_dist[i] = np.sqrt(v @ invC @ v)

    res["dist_to_centroid_euclid"] = euclid_dist
    res["dist_to_centroid_mahalanobis"] = maha_dist

    out_members = OUTDIR / "cluster_membership_with_distances_k7.csv"
    res.to_csv(out_members, index=False)

    std_feat_path = OUTDIR / "features_index70_standardized.csv"
    pd.DataFrame(Z, columns=[f"z_{c}" for c in CLUST_FEATS]).to_csv(std_feat_path, index=False)

    out_assign = OUTDIR / "cluster_assignments_k7.csv"
    res_assign = res[["countrycode", "countryname", "cluster"] + [f"proba_{i}" for i in range(K)]]
    res_assign.to_csv(out_assign, index=False)

    joblib.dump(gmm, OUTDIR / "gmm_k7.joblib")
    joblib.dump(scaler, OUTDIR / "scaler_for_gmm_k7.joblib")

    print("\ncluster size：")
    print(res["cluster"].value_counts().sort_index())

    print("\ntop_5 proba cty：")
    for k in range(K):
        topk = res[res["cluster"] == k].sort_values(f"proba_{k}", ascending=False).head(5)
        names = ", ".join(topk["countrycode"].tolist())
        print(f" - Cluster {k}: {names}")

    print(f"\nsaved：\n - {out_members}\n - {std_feat_path}\n - {out_assign}\n - {OUTDIR / 'gmm_k7.joblib'}\n - {OUTDIR / 'scaler_for_gmm_k7.joblib'}")


if __name__ == "__main__":
    main()
